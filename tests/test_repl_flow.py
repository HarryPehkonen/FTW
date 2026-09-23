"""Multi-turn REPL interaction, driven offline (ftw_plan.md §7 test layout,
§3.6 slash commands).

ReplSession takes its input as a callable and its output as a stream, so
the whole interactive loop is testable without a real terminal — no stdin,
no subprocess, no live model.
"""

import io

import pytest

from ftw.agent_loop import AgentLoop
from ftw.frames import FrameTree
from ftw.outputs import OutputStore
from ftw.providers import ChatMessage, ChatRole, MockModelProvider, ProviderResponse
from ftw.protocol import AskEnvelope, AskPayload
from ftw.repl.session import ReplSession, make_ask_answerer
from ftw.skills.registry import SkillStore
from ftw.workbench import ContextWorkbench


def assistant(text: str) -> ProviderResponse:
    return ProviderResponse(message=ChatMessage(role=ChatRole.ASSISTANT, content=text))


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
        dispatch=lambda call: (_ for _ in ()).throw(AssertionError("should not dispatch")),
    )
    out = io.StringIO()
    session = ReplSession(agent_loop=loop, input_fn=ScriptedInput(lines), output=out)
    return session, out


class TestConversation:
    def test_plain_input_is_sent_to_agent_loop_and_printed(self, tmp_path):
        session, out = make_session([assistant("hi there")], ["hello", "/exit"], tmp_path)
        session.run()
        assert "hi there" in out.getvalue()

    def test_empty_lines_are_ignored(self, tmp_path):
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider([assistant("hi")]),
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
        )
        session = ReplSession(agent_loop=loop, input_fn=ScriptedInput(["", "  ", "hello", "/exit"]), output=io.StringIO())
        session.run()
        assert len(loop.provider.calls) == 1

    def test_eof_ends_session_gracefully(self, tmp_path):
        session, out = make_session([], [], tmp_path)
        session.run()  # must not raise
        assert out.getvalue() == ""


class TestSlashCommands:
    def test_context_prints_workbench_report(self, tmp_path):
        session, out = make_session([], ["/context", "/exit"], tmp_path)
        session.run()
        assert "System Anchor" in out.getvalue()

    def test_clear_empties_turn_horizon(self, tmp_path):
        session, out = make_session([assistant("ok")], ["hi", "/clear", "/exit"], tmp_path)
        session.run()
        assert session.agent_loop.workbench.turns == []

    def test_exit_stops_the_loop(self, tmp_path):
        session, out = make_session([assistant("should not run")], ["/exit", "hello"], tmp_path)
        session.run()
        assert session.agent_loop.provider.calls == []  # never reached "hello"

    def test_unknown_command_reports_itself(self, tmp_path):
        session, out = make_session([], ["/bogus", "/exit"], tmp_path)
        session.run()
        assert "unknown command" in out.getvalue().lower()
        assert "/bogus" in out.getvalue()


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
        dispatch=lambda call: (_ for _ in ()).throw(AssertionError("should not dispatch")),
        frame_tree=tree,
    )
    out = io.StringIO()
    return ReplSession(agent_loop=loop, input_fn=ScriptedInput(lines), output=out), out


class TestMountCommands:
    def test_mount_prints_confirmation_and_updates_the_workbench(self, tmp_path):
        session, out = make_session_with_frames(["/mount cmake.diagnose_configure", "/exit"], tmp_path)
        session.run()
        assert "mounted" in out.getvalue()
        assert "cmake.diagnose_configure" in out.getvalue()
        assert session.agent_loop.workbench.mounted_skill_tokens > 0

    def test_mount_without_a_name_prints_usage(self, tmp_path):
        session, out = make_session_with_frames(["/mount", "/exit"], tmp_path)
        session.run()
        assert "usage" in out.getvalue().lower()

    def test_mount_unknown_skill_reports_error(self, tmp_path):
        session, out = make_session_with_frames(["/mount nope.nothing", "/exit"], tmp_path)
        session.run()
        assert "error" in out.getvalue().lower()

    def test_mount_pin_flag_makes_the_frame_user_owned(self, tmp_path):
        session, out = make_session_with_frames(
            ["/mount --pin cmake.diagnose_configure", "/exit"], tmp_path
        )
        session.run()
        frame = session.agent_loop.frame_tree._find_by_skill("cmake.diagnose_configure")  # whitebox check
        assert frame.pinned is True

    def test_commands_without_a_frame_tree_report_no_skill_store(self, tmp_path):
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider([]),
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
        )
        session = ReplSession(agent_loop=loop, input_fn=ScriptedInput(["/mount x", "/exit"]), output=io.StringIO())
        # must not raise even with no frame_tree wired
        session.run()


class TestUnmountFocusCommands:
    def test_unmount_with_no_name_targets_focused_frame(self, tmp_path):
        session, out = make_session_with_frames(
            ["/mount cmake.diagnose_configure", "/unmount", "/exit"], tmp_path
        )
        session.run()
        assert "milestone" in out.getvalue()
        assert session.agent_loop.workbench.mounted_skill_tokens == 0

    def test_unmount_with_nothing_mounted_reports_error(self, tmp_path):
        session, out = make_session_with_frames(["/unmount", "/exit"], tmp_path)
        session.run()
        assert "error" in out.getvalue().lower()

    def test_user_can_unmount_a_pinned_frame(self, tmp_path):
        session, out = make_session_with_frames(
            ["/mount --pin cmake.diagnose_configure", "/unmount cmake.diagnose_configure", "/exit"], tmp_path
        )
        session.run()
        assert session.agent_loop.workbench.mounted_skill_tokens == 0

    def test_focus_no_args_resets_to_root(self, tmp_path):
        session, out = make_session_with_frames(
            ["/mount cmake.diagnose_configure", "/focus", "/exit"], tmp_path
        )
        session.run()
        assert session.agent_loop.frame_tree.focused_skill_name is None
        assert "root" in out.getvalue().lower()

    def test_focus_by_name(self, tmp_path):
        session, out = make_session_with_frames(
            ["/mount cmake.diagnose_configure", "/focus cmake.diagnose_configure", "/exit"], tmp_path
        )
        session.run()
        assert "cmake.diagnose_configure" in out.getvalue()

    def test_focus_unknown_skill_reports_error(self, tmp_path):
        session, out = make_session_with_frames(["/focus nope.nothing", "/exit"], tmp_path)
        session.run()
        assert "error" in out.getvalue().lower()


class TestFramesCommand:
    def test_frames_shows_the_tree(self, tmp_path):
        session, out = make_session_with_frames(
            ["/mount cmake.diagnose_configure", "/frames", "/exit"], tmp_path
        )
        session.run()
        assert "cmake.diagnose_configure" in out.getvalue()

    def test_frames_on_empty_tree(self, tmp_path):
        session, out = make_session_with_frames(["/frames", "/exit"], tmp_path)
        session.run()
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
    def test_yes_no_answers_become_booleans(self, answer, expected):
        out = io.StringIO()
        answerer = make_ask_answerer(ScriptedInput([answer]), out)
        assert answerer(self._ask()) is expected
        assert "Overwrite" in out.getvalue()

    def test_free_text_answer_is_passed_through_as_is(self):
        answerer = make_ask_answerer(ScriptedInput(["/tmp/alt-build-dir"]), io.StringIO())
        assert answerer(self._ask("Which directory should I use?")) == "/tmp/alt-build-dir"

    def test_fails_closed_on_eof(self):
        answerer = make_ask_answerer(ScriptedInput([]), io.StringIO())
        assert answerer(self._ask()) is False
