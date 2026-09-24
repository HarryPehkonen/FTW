"""The REPL loop, decoupled from any real terminal (ftw_plan.md §3.6, §7
test_repl_flow.py).

``ReplSession`` takes input as a callable and output as a stream, so the
whole interactive loop — conversation turns, slash commands, EOF handling —
is unit-testable with a scripted input list and no subprocess. ``cli.py``
is the thin layer that wires this to a real terminal (readline, stdin,
stdout) plus worker supervision.

Everything here is async (see bus.py's module docstring for why the whole
call chain moved to asyncio), with ONE deliberate exception: reading a line
from ``input_fn`` is bridged through ``asyncio.to_thread`` rather than
rewritten as a native async reader, because ``input()``'s automatic
GNU-readline integration (arrow-key history/editing, already wired up via
cli.py's ``_enable_readline()``) has no clean async-native equivalent short
of a much larger dependency. The accepted trade-off: cancelling the
*awaiting* Task at an idle prompt (no turn in flight) unblocks the asyncio
side, but the underlying thread stays blocked on the real stdin read until
something is actually typed or EOF arrives — a narrower, separate gap from
the one this migration exists to fix (a blocked *bus call during a turn*),
documented alongside it in CLAUDE.md's Known Gaps.
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
    worker asking something other than yes/no. Fails closed on EOF."""

    async def answerer(ask: AskEnvelope) -> Any:
        print(f"{ask.payload.question} ", end="", file=output)
        try:
            answer = await asyncio.to_thread(input_fn, "")
        except EOFError:
            print("(EOF — declining)", file=output)
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
                line = await self._read_line()
            except EOFError:
                return
            if not await self.handle_line(line):
                return

    async def _read_line(self) -> str:
        return await asyncio.to_thread(self._input_fn, self._prompt)

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
