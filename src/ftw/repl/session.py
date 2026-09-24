"""The REPL loop, decoupled from any real terminal (ftw_plan.md §3.6, §7
test_repl_flow.py).

``ReplSession`` takes input as a callable and output as a stream, so the
whole interactive loop — conversation turns, slash commands, EOF handling —
is unit-testable with a scripted input list and no subprocess. ``cli.py``
is the thin layer that wires this to a real terminal (readline, stdin,
stdout) plus worker supervision.

Everything here is async (see bus.py's module docstring for why the whole
call chain moved to asyncio), with ONE deliberate exception: reading a line
from ``input_fn`` runs directly on the main thread — NOT bridged through
``asyncio.to_thread`` — via :func:`_blocking_read`. Two confirmed-empirically
reasons, not guesses:

1. Nothing else needs the event loop running while blocked on a line of
   input (no turn is in flight at the idle prompt; a worker-confirmation
   prompt mid-turn is the one other thing happening, and it's the only
   thing that matters right then too), so blocking the whole loop for
   that duration costs nothing.
2. ``asyncio.to_thread`` actively breaks Ctrl-C here. ``asyncio.run()``'s
   own ``Runner`` installs a SIGINT handler that, on the *first* Ctrl-C,
   only requests cooperative cancellation of the top-level task — which
   has nowhere to land while that task is blocked in a plain synchronous
   call, not an ``await``. Read via ``asyncio.to_thread`` specifically
   makes this worse: cancelling the wrapping Task doesn't stop the
   underlying OS thread (it's already running, and
   ``concurrent.futures.Future.cancel()`` is a no-op on a running future),
   so the thread stays genuinely blocked in the real read — and
   ``asyncio.run()``'s own shutdown (``shutdown_default_executor()``)
   later deadlocks trying to join it. Confirmed by reading
   ``asyncio/runners.py`` directly and reproducing both failure modes in
   isolation before landing this fix.

:func:`_blocking_read` sidesteps both problems: it temporarily restores
Python's plain default SIGINT handler (which raises ``KeyboardInterrupt``
immediately, unconditionally — no cooperative two-stage dance, no thread
to orphan) for the duration of the read, then restores whatever was
active before — so turn execution downstream keeps its own,
already-correct cancellation story (``_run_turn_interruptible``, which
installs its own handler via ``loop.add_signal_handler``).
"""

from __future__ import annotations

import asyncio
import signal
import sys
from typing import Any, Callable, TextIO

from ftw.agent_loop import AgentLoop
from ftw.frames import FrameError
from ftw.protocol import AskEnvelope
from ftw.skills.registry import SkillNotFound

InputFn = Callable[[str], str]


def _blocking_read(input_fn: InputFn, prompt: str) -> str:
    """Runs ``input_fn`` directly on the calling (main) thread, with
    Python's plain default SIGINT handler active for the duration — see
    the module docstring for why this, and not ``asyncio.to_thread``, is
    correct here. Raises ``KeyboardInterrupt`` on a single real Ctrl-C,
    exactly like an ordinary synchronous script; callers decide what that
    means for them (the idle prompt treats it as "quit"; a confirmation
    prompt treats it as "declined")."""
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        return input_fn(prompt)
    finally:
        signal.signal(signal.SIGINT, previous)


def make_ask_answerer(input_fn: InputFn, output: TextIO) -> Callable[[AskEnvelope], Any]:
    """Builds an AgentLoop ``ask_answerer`` callback: the one thing that
    answers ANY question AgentLoop asks — the interceptor's own ASK
    (question now includes the actual action and arguments, not just its
    reason) and a worker-initiated ASK (ftw_plan.md §4 "Mid-call
    interaction") alike, over the same input/output the REPL prompt uses.
    A delegated run's own interceptor ASK surfaces to the REPL exactly
    this way too — this is what makes it answerable at all. Yes/no reads
    as a bool (only an answer of exactly True ever allows an intercepted
    call through); anything else is passed through as free text, for a
    worker asking something other than yes/no. Fails closed on EOF *and*
    on a real Ctrl-C: declines the one action being confirmed here, via
    the same ``_blocking_read`` used by the idle prompt. In practice this
    usually ends the whole turn too, not just this one action — when
    called mid-turn, ``_run_turn_interruptible`` has already registered
    its own ``loop.add_signal_handler(SIGINT, turn_task.cancel)``, and
    that registration's ``signal.set_wakeup_fd()`` plumbing stays live
    underneath ``_blocking_read``'s temporary swap of the plain
    ``signal.signal()`` callback (confirmed by reading
    ``asyncio.unix_events``: the wakeup-fd write happens for the signal
    itself, independent of which Python-level callback is currently
    registered) — so the SAME Ctrl-C that declines here also reaches the
    turn-level handler once this coroutine returns control at its next
    real ``await``, cancelling the turn shortly after. That's a safe,
    tested outcome (the REPL stays fully usable afterward), just a
    broader one than "decline only" — not worth the real coupling a
    narrower fix would need (threading the turn's own Task down into this
    callback)."""

    async def answerer(ask: AskEnvelope) -> Any:
        print(f"{ask.payload.question} ", end="", file=output)
        try:
            answer = _blocking_read(input_fn, "")
        except EOFError:
            print("(EOF — declining)", file=output)
            return False
        except KeyboardInterrupt:
            print("(Ctrl-C — declining)", file=output)
            return False
        stripped = answer.strip()
        if stripped.lower() in ("y", "yes"):
            return True
        if stripped.lower() in ("n", "no", ""):
            return False
        return stripped

    return answerer


class ReplSession:
    def __init__(
        self,
        *,
        agent_loop: AgentLoop,
        input_fn: InputFn,
        output: TextIO = sys.stdout,
        prompt: str = "ftw> ",
    ):
        self.agent_loop = agent_loop
        self._input_fn = input_fn
        self._output = output
        self._prompt = prompt

    async def run(self) -> None:
        while True:
            try:
                line = self._read_line()
            except EOFError:
                return
            if not await self.handle_line(line):
                return

    def _read_line(self) -> str:
        """A real SIGINT while idle here (no turn in flight) ends the
        session, the same path as EOF - not a "cancel and re-prompt", the
        way _run_turn_interruptible handles an actual turn. Plain
        (synchronous, not async) on purpose: see the module docstring for
        why blocking the whole event loop here is correct, and why the
        previous asyncio.to_thread-based approach didn't just fail to
        cancel cleanly but deadlocked asyncio.run()'s own shutdown."""
        try:
            return _blocking_read(self._input_fn, self._prompt)
        except KeyboardInterrupt:
            raise EOFError from None

    async def handle_line(self, line: str) -> bool:
        """Processes one line. Returns False when the session should end."""
        line = line.strip()
        if not line:
            return True
        if line.startswith("/"):
            return await self._handle_command(line)
        reply = await self._run_turn_interruptible(line)
        self._print(reply)
        return True

    async def _run_turn_interruptible(self, line: str) -> str:
        """Runs one turn as its own Task, with a real SIGINT cancelling it
        promptly — the actual fix this whole migration exists to deliver.
        A cancelled turn ends cleanly and the REPL is immediately usable
        again for the next line, rather than the process staying wedged
        until some blocking call eventually times out on its own."""
        loop = asyncio.get_running_loop()
        turn_task = asyncio.ensure_future(self.agent_loop.run_turn(line))
        try:
            loop.add_signal_handler(signal.SIGINT, turn_task.cancel)
        except (NotImplementedError, RuntimeError):
            pass  # signal handlers aren't available on every platform/loop; best-effort
        try:
            return await turn_task
        except asyncio.CancelledError:
            return "cancelled by user (Ctrl-C)"
        finally:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, RuntimeError, ValueError):
                pass

    async def _handle_command(self, line: str) -> bool:
        parts = line.split(maxsplit=1)
        cmd = parts[0]

        if cmd == "/exit":
            return False
        if cmd == "/context":
            self._print(self.agent_loop.workbench.render_report())
            return True
        if cmd == "/clear":
            self.agent_loop.workbench.clear_turns()
            self._print("cleared turn horizon")
            return True
        if cmd == "/mount":
            return await self._cmd_mount(parts)
        if cmd == "/unmount":
            return await self._cmd_unmount(parts)
        if cmd == "/focus":
            return await self._cmd_focus(parts)
        if cmd == "/frames":
            return self._cmd_frames()

        self._print(f"unknown command: {cmd}")
        return True

    # -- skill mounting commands (ftw_plan.md §3.2 Frames) -----------------

    async def _cmd_mount(self, parts: list[str]) -> bool:
        if self.agent_loop.frame_tree is None:
            self._print("no skill store configured")
            return True
        tokens = parts[1].split() if len(parts) > 1 else []
        pinned = "--pin" in tokens
        names = [t for t in tokens if t != "--pin"]
        if not names:
            self._print("usage: /mount [--pin] <skill>")
            return True
        try:
            frame = await self.agent_loop.frame_tree.mount(names[0], owner="user", pinned=pinned)
        except (FrameError, SkillNotFound) as exc:
            self._print(f"error: {exc}")
            return True
        self._print(f"mounted {frame.skill_name!r}" + (" (pinned)" if pinned else ""))
        return True

    async def _cmd_unmount(self, parts: list[str]) -> bool:
        if self.agent_loop.frame_tree is None:
            self._print("no skill store configured")
            return True
        name = parts[1].strip() if len(parts) > 1 else ""
        try:
            if name:
                milestone = await self.agent_loop.frame_tree.unmount(name, by="user")
            else:
                milestone = await self.agent_loop.frame_tree.unmount_focused(by="user")
        except FrameError as exc:
            self._print(f"error: {exc}")
            return True
        self._print(f"unmounted; milestone: {milestone}")
        return True

    async def _cmd_focus(self, parts: list[str]) -> bool:
        if self.agent_loop.frame_tree is None:
            self._print("no skill store configured")
            return True
        name = parts[1].strip() if len(parts) > 1 else ""
        try:
            self.agent_loop.frame_tree.focus(name or None)
        except FrameError as exc:
            self._print(f"error: {exc}")
            return True
        focused = self.agent_loop.frame_tree.focused_skill_name
        self._print(f"focus: {focused}" if focused else "focus: (root session)")
        return True

    def _cmd_frames(self) -> bool:
        if self.agent_loop.frame_tree is None:
            self._print("no skill store configured")
            return True
        self._print(self.agent_loop.frame_tree.render_tree())
        return True

    def _print(self, text: str) -> None:
        print(text, file=self._output)
