"""The agent loop: model -> interceptor -> bus -> result cycle (ftw_plan.md
§3.4, and the "loop that drives the model" gap called out during planning).

Shared by the REPL; the delegated skill runner (Phase 3) will reuse it with
a fresh workbench. Everything here runs against MockModelProvider and a
recording fake dispatcher — no bus, no network, no live tokens.
"""

from dataclasses import dataclass

import pytest

from ftw.agent_loop import AgentLoop
from ftw.intercept import ConfirmShellCommands, InterceptDecision, InterceptOutcome, PreCommitInterceptor
from ftw.outputs import OutputStore
from ftw.protocol import CallEnvelope, ErrorEnvelope, ErrorPayload, ResultEnvelope, ResultPayload
from ftw.providers import ChatMessage, ChatRole, MockModelProvider, ProviderResponse, ToolCall
from ftw.workbench import ContextWorkbench


def assistant_text(text: str) -> ProviderResponse:
    return ProviderResponse(message=ChatMessage(role=ChatRole.ASSISTANT, content=text))


def assistant_tool_call(name: str, arguments: dict, *, call_id: str = "call-1") -> ProviderResponse:
    return ProviderResponse(
        message=ChatMessage(
            role=ChatRole.ASSISTANT,
            content=None,
            tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        )
    )


@dataclass
class RecordedDispatch:
    call: CallEnvelope


class RecordingDispatcher:
    """A fake bus: records every CallEnvelope it's given and returns the
    next scripted reply."""

    def __init__(self, replies: list):
        self._replies = list(replies)
        self.calls: list[CallEnvelope] = []

    def __call__(self, call: CallEnvelope):
        self.calls.append(call)
        return self._replies.pop(0)


def make_loop(
    *,
    responses,
    dispatch=None,
    interceptor=None,
    confirm=None,
    max_steps=15,
    output_root,
    on_event=None,
):
    return AgentLoop(
        workbench=ContextWorkbench(system_anchor="be terse"),
        provider=MockModelProvider(responses),
        output_store=OutputStore(output_root),
        dispatch=dispatch or (lambda call: (_ for _ in ()).throw(AssertionError("dispatch should not be called"))),
        interceptor=interceptor or PreCommitInterceptor([]),
        confirm=confirm,
        max_steps=max_steps,
        on_event=on_event,
    )


class TestFinalAnswerNoTools:
    def test_returns_text_and_commits_one_turn(self, tmp_path):
        loop = make_loop(responses=[assistant_text("42")], output_root=tmp_path)

        result = loop.run_turn("what is 6*7?")

        assert result == "42"
        assert len(loop.workbench.turns) == 1
        turn = loop.workbench.turns[0]
        assert turn[0] == ChatMessage(role=ChatRole.USER, content="what is 6*7?")
        assert turn[-1].content == "42"


class TestDispatchedToolCall:
    def test_allowed_call_is_dispatched_and_result_fed_back(self, tmp_path):
        result_reply = ResultEnvelope(
            source="worker.tool.shell",
            target="repl.master",
            payload=ResultPayload(status="ok", summary="exit 0: echo hi", outputs={"exit_code": 0}),
        )
        dispatcher = RecordingDispatcher([result_reply])
        loop = make_loop(
            responses=[
                assistant_tool_call("run_command", {"argv": ["echo", "hi"]}),
                assistant_text("it printed hi"),
            ],
            dispatch=dispatcher,
            interceptor=PreCommitInterceptor([]),  # ALLOW everything, no confirmation needed
            output_root=tmp_path,
        )

        result = loop.run_turn("run echo hi")

        assert result == "it printed hi"
        assert len(dispatcher.calls) == 1
        assert dispatcher.calls[0].payload.action == "run_command"
        assert dispatcher.calls[0].target == "worker.tool.shell"
        assert dispatcher.calls[0].trace_id == loop.trace_id

    def test_tool_result_content_reaches_the_next_prompt(self, tmp_path):
        result_reply = ResultEnvelope(
            source="worker.tool.shell",
            target="repl.master",
            payload=ResultPayload(status="ok", summary="exit 0: echo hi", outputs={"exit_code": 0}),
        )
        dispatcher = RecordingDispatcher([result_reply])
        provider = MockModelProvider(
            [
                assistant_tool_call("run_command", {"argv": ["echo", "hi"]}),
                assistant_text("done"),
            ]
        )
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=provider,
            output_store=OutputStore(tmp_path),
            dispatch=dispatcher,
            interceptor=PreCommitInterceptor([]),
        )

        loop.run_turn("run echo hi")

        second_call_messages = provider.calls[1].messages
        tool_messages = [m for m in second_call_messages if m.role == ChatRole.TOOL]
        assert len(tool_messages) == 1
        assert "exit 0" in tool_messages[0].content


class TestConfirmation:
    def test_ask_decision_dispatches_only_when_confirmed(self, tmp_path):
        result_reply = ResultEnvelope(
            source="worker.tool.shell",
            target="repl.master",
            payload=ResultPayload(status="ok", summary="exit 0"),
        )
        dispatcher = RecordingDispatcher([result_reply])
        loop = make_loop(
            responses=[
                assistant_tool_call("run_command", {"argv": ["ls"]}),
                assistant_text("listed"),
            ],
            dispatch=dispatcher,
            interceptor=PreCommitInterceptor([ConfirmShellCommands()]),
            confirm=lambda call: True,
            output_root=tmp_path,
        )

        result = loop.run_turn("list files")

        assert result == "listed"
        assert len(dispatcher.calls) == 1

    def test_ask_decision_declined_does_not_dispatch(self, tmp_path):
        loop = make_loop(
            responses=[
                assistant_tool_call("run_command", {"argv": ["rm", "-rf", "/"]}),
                assistant_text("ok, not running that"),
            ],
            dispatch=None,
            interceptor=PreCommitInterceptor([ConfirmShellCommands()]),
            confirm=lambda call: False,
            output_root=tmp_path,
        )

        result = loop.run_turn("delete everything")

        assert result == "ok, not running that"

    def test_default_confirm_denies(self, tmp_path):
        """No confirm callback wired up (e.g. non-interactive) must fail
        closed, never silently allow a shell command through."""
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider(
                [assistant_tool_call("run_command", {"argv": ["ls"]}), assistant_text("skipped")]
            ),
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError("must not dispatch")),
            interceptor=PreCommitInterceptor([ConfirmShellCommands()]),
        )
        assert loop.run_turn("do it") == "skipped"

    def test_block_decision_never_calls_confirm_or_dispatch(self, tmp_path):
        class AlwaysBlock:
            def evaluate(self, call):
                return InterceptOutcome(InterceptDecision.BLOCK, reason="path outside sandbox")

        confirm_calls = []
        loop = make_loop(
            responses=[
                assistant_tool_call("run_command", {"argv": ["rm", "-rf", "/"]}),
                assistant_text("blocked"),
            ],
            dispatch=None,
            interceptor=PreCommitInterceptor([AlwaysBlock()]),
            confirm=lambda call: confirm_calls.append(call) or True,
            output_root=tmp_path,
        )

        result = loop.run_turn("delete everything")

        assert result == "blocked"
        assert confirm_calls == []


class TestUnknownTool:
    def test_unknown_tool_name_reported_without_touching_bus_or_interceptor(self, tmp_path):
        loop = make_loop(
            responses=[
                assistant_tool_call("teleport", {"where": "mars"}),
                assistant_text("can't do that"),
            ],
            output_root=tmp_path,
        )
        assert loop.run_turn("teleport me") == "can't do that"


class TestLocalTools:
    def test_pin_updates_workbench_without_dispatch(self, tmp_path):
        loop = make_loop(
            responses=[
                assistant_tool_call("pin", {"key": "repo", "value": "/src"}),
                assistant_text("pinned it"),
            ],
            output_root=tmp_path,
        )
        assert loop.run_turn("remember the repo path") == "pinned it"
        assert loop.workbench.scratchpad["repo"] == "/src"

    def test_pin_over_budget_reports_error_without_crashing(self, tmp_path):
        loop = AgentLoop(
            workbench=ContextWorkbench(scratchpad_budget=2),
            provider=MockModelProvider(
                [
                    assistant_tool_call("pin", {"key": "k", "value": "way more than two tokens of value"}),
                    assistant_text("couldn't pin it"),
                ]
            ),
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
        )
        assert loop.run_turn("pin something huge") == "couldn't pin it"
        assert loop.workbench.scratchpad == {}

    def test_unpin_removes_key(self, tmp_path):
        wb = ContextWorkbench()
        wb.pin("repo", "/src")
        loop = AgentLoop(
            workbench=wb,
            provider=MockModelProvider([assistant_tool_call("unpin", {"key": "repo"}), assistant_text("done")]),
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
        )
        loop.run_turn("forget the repo path")
        assert "repo" not in wb.scratchpad

    def test_read_output_returns_stored_content(self, tmp_path):
        store = OutputStore(tmp_path)
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider(
                [assistant_tool_call("read_output", {"output_id": "will-be-filled"}), assistant_text("read it")]
            ),
            output_store=store,
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
        )
        output_id = store.save(loop.trace_id, "line one\nline two\n")
        loop.provider._responses[0].message.tool_calls[0].arguments["output_id"] = output_id

        assert loop.run_turn("show me the output") == "read it"

    def test_grep_output_filters_lines(self, tmp_path):
        store = OutputStore(tmp_path)
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider(
                [assistant_tool_call("grep_output", {"output_id": "x", "pattern": "Error"}), assistant_text("found it")]
            ),
            output_store=store,
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
        )
        store.save(loop.trace_id, "ok\nError: bad\nok", output_id="x")

        assert loop.run_turn("find errors") == "found it"


class TestStepBudget:
    def test_stops_after_max_steps_without_final_answer(self, tmp_path):
        responses = [assistant_tool_call("pin", {"key": "k", "value": "v"}, call_id=f"c{i}") for i in range(3)]
        provider = MockModelProvider(responses)
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=provider,
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
            max_steps=3,
        )

        result = loop.run_turn("loop forever")

        assert "budget" in result.lower()
        assert len(provider.calls) == 3  # never asked a 4th time


class TestEvents:
    def test_emits_model_and_intercept_and_tool_events_in_order(self, tmp_path):
        result_reply = ResultEnvelope(
            source="worker.tool.shell", target="repl.master", payload=ResultPayload(status="ok", summary="ok")
        )
        dispatcher = RecordingDispatcher([result_reply])
        events = []
        loop = make_loop(
            responses=[
                assistant_tool_call("run_command", {"argv": ["ls"]}),
                assistant_text("done"),
            ],
            dispatch=dispatcher,
            interceptor=PreCommitInterceptor([]),
            output_root=tmp_path,
            on_event=lambda evt: events.append(evt.payload.topic),
        )

        loop.run_turn("list files")

        assert events == [
            "event.model.request",
            "event.model.response",
            "event.intercept.decision",
            "event.tool.result",
            "event.model.request",
            "event.model.response",
        ]


class TestErrorReply:
    def test_error_envelope_from_dispatch_is_reported_to_model(self, tmp_path):
        error_reply = ErrorEnvelope(
            source="worker.tool.shell", target="repl.master", payload=ErrorPayload(code="not_found", message="no such file")
        )
        dispatcher = RecordingDispatcher([error_reply])
        loop = make_loop(
            responses=[
                assistant_tool_call("run_command", {"argv": ["nope"]}),
                assistant_text("that file doesn't exist"),
            ],
            dispatch=dispatcher,
            interceptor=PreCommitInterceptor([]),
            output_root=tmp_path,
        )

        assert loop.run_turn("run nope") == "that file doesn't exist"
