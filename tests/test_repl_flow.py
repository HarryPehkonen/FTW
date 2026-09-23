"""Multi-turn REPL interaction, driven offline (ftw_plan.md §7 test layout,
§3.6 slash commands).

ReplSession takes its input as a callable and its output as a stream, so
the whole interactive loop is testable without a real terminal — no stdin,
no subprocess, no live model.
"""

import io

import pytest

from ftw.agent_loop import AgentLoop
from ftw.outputs import OutputStore
from ftw.providers import ChatMessage, ChatRole, MockModelProvider, ProviderResponse
from ftw.protocol import CallEnvelope
from ftw.repl.session import ReplSession, make_confirm
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


class TestMakeConfirm:
    def _call(self) -> CallEnvelope:
        from ftw.protocol import CallPayload

        return CallEnvelope(
            source="repl.master",
            target="worker.tool.shell",
            payload=CallPayload(action="run_command", args={"argv": ["rm", "-rf", "build"]}),
        )

    @pytest.mark.parametrize("answer,expected", [("y", True), ("yes", True), ("Y", True), ("n", False), ("", False), ("whatever", False)])
    def test_confirm_reads_yes_no(self, answer, expected):
        out = io.StringIO()
        confirm = make_confirm(ScriptedInput([answer]), out)
        assert confirm(self._call()) is expected
        assert "run_command" in out.getvalue()

    def test_confirm_fails_closed_on_eof(self):
        confirm = make_confirm(ScriptedInput([]), io.StringIO())
        assert confirm(self._call()) is False
