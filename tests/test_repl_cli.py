"""Worker supervision and end-to-end wiring (ftw_plan.md §7 Phase 1
deliverable: "REPL mounts... runs confirmed shell commands over NNG").

This is the one place a real subprocess and a real ipc:// socket are
exercised — everything else in the suite stays over inproc://. Still zero
network calls and zero live LLM tokens: the model side is MockModelProvider.
"""

import io
import sys

from ftw.providers import ChatMessage, ChatRole, MockModelProvider, ProviderResponse, ToolCall
from ftw.repl.cli import build_repl_session


class ScriptedInput:
    """Mimics input(): pops the next scripted line, raises EOFError once
    exhausted."""

    def __init__(self, lines: list[str]):
        self._lines = list(lines)

    def __call__(self, prompt: str = "") -> str:
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


def assistant_run_command() -> ProviderResponse:
    return ProviderResponse(
        message=ChatMessage(
            role=ChatRole.ASSISTANT,
            content=None,
            tool_calls=[ToolCall(id="c1", name="run_command", arguments={"argv": [sys.executable, "-c", "print('from the real worker')"]})],
        )
    )


def assistant_text(text: str) -> ProviderResponse:
    return ProviderResponse(message=ChatMessage(role=ChatRole.ASSISTANT, content=text))


class TestBuildReplSessionWithRealWorker:
    def test_shell_command_round_trips_through_a_real_subprocess_worker(self, tmp_path):
        provider = MockModelProvider([assistant_run_command(), assistant_text("it printed the line")])
        out = io.StringIO()

        handle = build_repl_session(
            ftw_home=tmp_path,
            provider=provider,
            input_fn=ScriptedInput(["y"]),  # confirms the one shell command
            output=out,
        )
        try:
            reply = handle.session.agent_loop.run_turn("run something")
        finally:
            handle.close()

        assert reply == "it printed the line"
