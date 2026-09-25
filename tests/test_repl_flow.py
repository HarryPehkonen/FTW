"""Multi-turn REPL interaction, driven offline (ftw_plan.md §7 test layout,
§3.6 slash commands).

ReplSession takes its input as a callable and its output as a stream, so
the whole interactive loop is testable without a real terminal — no stdin,
no subprocess, no live model. ReplSession.run()/handle_line() are async -
see repl/session.py's module docstring for why (and for the one
deliberate exception: input_fn is still a plain blocking callable, bridged
through asyncio.to_thread).
"""

import io
import os
import signal
import threading
import time

import pytest

from ftw.agent_loop import AgentLoop
from ftw.frames import FrameTree
from ftw.outputs import OutputStore
from ftw.protocol import AskEnvelope, AskPayload
from ftw.providers import ChatMessage, ChatRole, MockModelProvider, ProviderResponse
from ftw.repl.session import ReplSession, make_ask_answerer
from ftw.skills.registry import SkillStore
from ftw.workbench import ContextWorkbench


def assistant(text: str) -> ProviderResponse:
    return ProviderResponse(message=ChatMessage(role=ChatRole.ASSISTANT, content=text))


def async_raising(exc: BaseException):
    """Wraps an exception instance as an async callable that raises it -
    stands in for what used to be
    `lambda call: (_ for _ in ()).throw(exc)`."""

    async def fn(*args, **kwargs):
        raise exc

    return fn


class ScriptedInput:
    """Mimics input(): pops the next scripted line, raises EOFError once
    exhausted — exactly like real stdin hitting EOF."""

    def __init__(self, lines: list[str]):
        self._lines = list(lines)

    def __call__(self, prompt: str = "") -> str:
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


def make_session(responses: list[ProviderResponse], lines: list[str], tmp_path) -> tuple[ReplSession, io.StringIO]:
    loop = AgentLoop(
        workbench=ContextWorkbench(system_anchor="be terse"),
        provider=MockModelProvider(responses),
        output_store=OutputStore(tmp_path),
        dispatch=async_raising(AssertionError("should not dispatch")),
    )
    out = io.StringIO()
    session = ReplSession(agent_loop=loop, input_fn=ScriptedInput(lines), output=out)
    return session, out


class TestConversation:
    async def test_plain_input_is_sent_to_agent_loop_and_printed(self, tmp_path):
        session, out = make_session([assistant("hi there")], ["hello", "/exit"], tmp_path)
        await session.run()
        assert "hi there" in out.getvalue()

    async def test_empty_lines_are_ignored(self, tmp_path):
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider([assistant("hi")]),
            output_store=OutputStore(tmp_path),
            dispatch=async_raising(AssertionError()),
        )
        session = ReplSession(agent_loop=loop, input_fn=ScriptedInput(["", "  ", "hello", "/exit"]), output=io.StringIO())
        await session.run()
        assert len(loop.provider.calls) == 1

    async def test_eof_ends_session_gracefully(self, tmp_path):
        session, out = make_session([], [], tmp_path)
        await session.run()  # must not raise
        assert out.getvalue() == ""


class TestSlashCommands:
    async def test_context_prints_workbench_report(self, tmp_path):
        session, out = make_session([], ["/context", "/exit"], tmp_path)
        await session.run()
        assert "System Anchor" in out.getvalue()

    async def test_clear_empties_turn_horizon(self, tmp_path):
        session, _out = make_session([assistant("ok")], ["hi", "/clear", "/exit"], tmp_path)
        await session.run()
        assert session.agent_loop.workbench.turns == []

    async def test_exit_stops_the_loop(self, tmp_path):
        session, _out = make_session([assistant("should not run")], ["/exit", "hello"], tmp_path)
        await session.run()
        assert session.agent_loop.provider.calls == []  # never reached "hello"

    async def test_unknown_command_reports_itself(self, tmp_path):
        session, out = make_session([], ["/bogus", "/exit"], tmp_path)
        await session.run()
        assert "unknown command" in out.getvalue().lower()
        assert "/bogus" in out.getvalue()
        assert "/help" in out.getvalue()  # points the user at how to find the real list


class TestHelpCommand:
    async def test_lists_every_command_with_a_description(self, tmp_path):
        session, out = make_session([], ["/help", "/exit"], tmp_path)
        await session.run()
        text = out.getvalue()
        # every command /help itself documents must actually appear, each
        # with some descriptive text alongside it, not just a bare list
        for cmd in ("/help", "/context", "/clear", "/mount", "/unmount", "/focus", "/frames", "/exit"):
            assert cmd in text, f"{cmd!r} missing from /help output"
        assert "quit" in text.lower()  # spot-check that descriptions, not just names, are present

    async def test_help_does_not_require_a_frame_tree(self, tmp_path):
        """/mount, /unmount, /focus, /frames are always listed - they each
        already handle "no skill store configured" gracefully on their
        own, so /help shouldn't need a frame_tree to describe them."""
        session, out = make_session([], ["/help", "/exit"], tmp_path)
        await session.run()  # must not raise even with no frame_tree wired
        assert "/mount" in out.getvalue()


def write_skill(root, relpath: str, name: str, description: str, body: str = "do the thing") -> None:
    path = root / relpath / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")


def make_session_with_frames(lines: list[str], tmp_path) -> tuple[ReplSession, io.StringIO]:
    write_skill(tmp_path, "cmake/diagnose_configure", "cmake.diagnose_configure", "Diagnose failing CMake configuration.")
    workbench = ContextWorkbench()
    tree = FrameTree(workbench, SkillStore(tmp_path))
    loop = AgentLoop(
        workbench=workbench,
        provider=MockModelProvider([]),
        output_store=OutputStore(tmp_path / "outputs"),
        dispatch=async_raising(AssertionError("should not dispatch")),
        frame_tree=tree,
    )
    out = io.StringIO()
    return ReplSession(agent_loop=loop, input_fn=ScriptedInput(lines), output=out), out


class TestMountCommands:
    async def test_mount_prints_confirmation_and_updates_the_workbench(self, tmp_path):
        session, out = make_session_with_frames(["/mount cmake.diagnose_configure", "/exit"], tmp_path)
        await session.run()
        assert "mounted" in out.getvalue()
        assert "cmake.diagnose_configure" in out.getvalue()
        assert session.agent_loop.workbench.mounted_skill_tokens > 0

    async def test_mount_without_a_name_prints_usage(self, tmp_path):
        session, out = make_session_with_frames(["/mount", "/exit"], tmp_path)
        await session.run()
        assert "usage" in out.getvalue().lower()

    async def test_mount_unknown_skill_reports_error(self, tmp_path):
        session, out = make_session_with_frames(["/mount nope.nothing", "/exit"], tmp_path)
        await session.run()
        assert "error" in out.getvalue().lower()

    async def test_mount_pin_flag_makes_the_frame_user_owned(self, tmp_path):
        session, _out = make_session_with_frames(
            ["/mount --pin cmake.diagnose_configure", "/exit"], tmp_path
        )
        await session.run()
        frame = session.agent_loop.frame_tree._find_by_skill("cmake.diagnose_configure")  # whitebox check
        assert frame.pinned is True

    async def test_commands_without_a_frame_tree_report_no_skill_store(self, tmp_path):
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider([]),
            output_store=OutputStore(tmp_path),
            dispatch=async_raising(AssertionError()),
        )
        session = ReplSession(agent_loop=loop, input_fn=ScriptedInput(["/mount x", "/exit"]), output=io.StringIO())
        # must not raise even with no frame_tree wired
        await session.run()


class TestUnmountFocusCommands:
    async def test_unmount_with_no_name_targets_focused_frame(self, tmp_path):
        session, out = make_session_with_frames(
            ["/mount cmake.diagnose_configure", "/unmount", "/exit"], tmp_path
        )
        await session.run()
        assert "milestone" in out.getvalue()
        assert session.agent_loop.workbench.mounted_skill_tokens == 0

    async def test_unmount_with_nothing_mounted_reports_error(self, tmp_path):
        session, out = make_session_with_frames(["/unmount", "/exit"], tmp_path)
        await session.run()
        assert "error" in out.getvalue().lower()

    async def test_user_can_unmount_a_pinned_frame(self, tmp_path):
        session, _out = make_session_with_frames(
            ["/mount --pin cmake.diagnose_configure", "/unmount cmake.diagnose_configure", "/exit"], tmp_path
        )
        await session.run()
        assert session.agent_loop.workbench.mounted_skill_tokens == 0

    async def test_focus_no_args_resets_to_root(self, tmp_path):
        session, out = make_session_with_frames(
            ["/mount cmake.diagnose_configure", "/focus", "/exit"], tmp_path
        )
        await session.run()
        assert session.agent_loop.frame_tree.focused_skill_name is None
        assert "root" in out.getvalue().lower()

    async def test_focus_by_name(self, tmp_path):
        session, out = make_session_with_frames(
            ["/mount cmake.diagnose_configure", "/focus cmake.diagnose_configure", "/exit"], tmp_path
        )
        await session.run()
        assert "cmake.diagnose_configure" in out.getvalue()

    async def test_focus_unknown_skill_reports_error(self, tmp_path):
        session, out = make_session_with_frames(["/focus nope.nothing", "/exit"], tmp_path)
        await session.run()
        assert "error" in out.getvalue().lower()


class TestFramesCommand:
    async def test_frames_shows_the_tree(self, tmp_path):
        session, out = make_session_with_frames(
            ["/mount cmake.diagnose_configure", "/frames", "/exit"], tmp_path
        )
        await session.run()
        assert "cmake.diagnose_configure" in out.getvalue()

    async def test_frames_on_empty_tree(self, tmp_path):
        session, out = make_session_with_frames(["/frames", "/exit"], tmp_path)
        await session.run()
        assert "nothing mounted" in out.getvalue().lower()


class TestMakeAskAnswerer:
    """The one callback that answers any question AgentLoop asks: the
    interceptor's own pre-dispatch ASK (ftw_plan.md §3.4) and a
    worker-initiated ASK (§4 "Mid-call interaction") alike. Used both for
    a worker asking something directly and, identically from the REPL's
    point of view, for a delegated run's own interceptor ASK relayed back
    as one."""

    def _ask(self, question: str = "Overwrite build/CMakeCache.txt?") -> AskEnvelope:
        return AskEnvelope(
            source="skill.runner", target="repl.master", payload=AskPayload(question=question, resume_token="tok-1")
        )

    @pytest.mark.parametrize("answer,expected", [("y", True), ("yes", True), ("Y", True), ("n", False), ("", False)])
    async def test_yes_no_answers_become_booleans(self, answer, expected):
        out = io.StringIO()
        answerer = make_ask_answerer(ScriptedInput([answer]), out)
        assert await answerer(self._ask()) is expected
        assert "Overwrite" in out.getvalue()

    async def test_free_text_answer_is_passed_through_as_is(self):
        answerer = make_ask_answerer(ScriptedInput(["/tmp/alt-build-dir"]), io.StringIO())
        assert await answerer(self._ask("Which directory should I use?")) == "/tmp/alt-build-dir"

    async def test_fails_closed_on_eof(self):
        answerer = make_ask_answerer(ScriptedInput([]), io.StringIO())
        assert await answerer(self._ask()) is False

    async def test_fails_closed_on_a_real_sigint(self):
        """A real Ctrl-C while blocked answering a confirmation prompt
        declines that one action - the same locally-recoverable way a
        Ctrl-C during tool dispatch is already handled - rather than
        crashing the whole turn with an uncaught KeyboardInterrupt."""

        def blocking_input_fn(prompt: str = "") -> str:
            time.sleep(10)
            raise AssertionError("should have been interrupted before this returns")

        out = io.StringIO()
        answerer = make_ask_answerer(blocking_input_fn, out)

        def send_sigint_soon():
            time.sleep(0.3)
            os.kill(os.getpid(), signal.SIGINT)

        threading.Thread(target=send_sigint_soon, daemon=True).start()

        start = time.monotonic()
        result = await answerer(self._ask())
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed < 2.0
        assert "Ctrl-C" in out.getvalue()


class TestIdleSigintEndsTheSessionCleanly:
    """A real SIGINT while sitting idle at the prompt (no turn in flight)
    must end the session cleanly instead of hanging forever. The read now
    runs directly on the main thread (see session.py's module docstring
    for why asyncio.to_thread specifically broke this - it isn't just
    "doesn't cancel cleanly", it deadlocks asyncio.run()'s own shutdown
    trying to join the orphaned worker thread), with Python's plain
    default SIGINT handler temporarily active so a single real Ctrl-C
    raises KeyboardInterrupt immediately, exactly like an ordinary
    synchronous script."""

    async def test_real_sigint_while_idle_ends_the_session(self, tmp_path):
        def blocking_input_fn(prompt: str = "") -> str:
            time.sleep(10)
            raise AssertionError("should have been interrupted before this returns")

        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider([]),
            output_store=OutputStore(tmp_path),
            dispatch=async_raising(AssertionError("should not dispatch")),
        )
        session = ReplSession(agent_loop=loop, input_fn=blocking_input_fn, output=io.StringIO())

        def send_sigint_soon():
            time.sleep(0.3)
            os.kill(os.getpid(), signal.SIGINT)

        threading.Thread(target=send_sigint_soon, daemon=True).start()

        start = time.monotonic()
        await session.run()  # must return promptly, not hang until blocking_input_fn ever returns
        elapsed = time.monotonic() - start

        assert elapsed < 2.0
