"""The agent loop: model -> interceptor -> bus -> result cycle (ftw_plan.md
§3.4, and the "loop that drives the model" gap called out during planning).

Shared by the REPL; the delegated skill runner (Phase 3) will reuse it with
a fresh workbench. Everything here runs against MockModelProvider and a
recording fake dispatcher — no bus, no network, no live tokens.
"""

import threading
from dataclasses import dataclass

import pytest

from ftw.agent_loop import AgentLoop, DelegatedSuspension
from ftw.bus import DeadlineExceeded
from ftw.frames import FrameTree
from ftw.intercept import ConfirmShellCommands, InterceptDecision, InterceptOutcome, PreCommitInterceptor
from ftw.outputs import OutputStore
from ftw.protocol import (
    AnswerEnvelope,
    AskEnvelope,
    AskPayload,
    CallEnvelope,
    ErrorEnvelope,
    ErrorPayload,
    ResultEnvelope,
    ResultPayload,
)
from ftw.providers import ChatMessage, ChatRole, MockModelProvider, ProviderError, ProviderResponse, ToolCall
from ftw.skills.registry import SkillStore
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
    ask_answerer=None,
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
        ask_answerer=ask_answerer,
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
            ask_answerer=lambda ask: True,
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
            ask_answerer=lambda ask: False,
            output_root=tmp_path,
        )

        result = loop.run_turn("delete everything")

        assert result == "ok, not running that"

    def test_default_ask_answerer_denies(self, tmp_path):
        """No ask_answerer callback wired up (e.g. non-interactive) must
        fail closed, never silently allow a shell command through."""
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

    def test_block_decision_never_calls_ask_answerer_or_dispatch(self, tmp_path):
        class AlwaysBlock:
            def evaluate(self, call):
                return InterceptOutcome(InterceptDecision.BLOCK, reason="path outside sandbox")

        answerer_calls = []
        loop = make_loop(
            responses=[
                assistant_tool_call("run_command", {"argv": ["rm", "-rf", "/"]}),
                assistant_text("blocked"),
            ],
            dispatch=None,
            interceptor=PreCommitInterceptor([AlwaysBlock()]),
            ask_answerer=lambda ask: answerer_calls.append(ask) or True,
            output_root=tmp_path,
        )

        result = loop.run_turn("delete everything")

        assert result == "blocked"
        assert answerer_calls == []


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


def write_skill(root, relpath: str, name: str, description: str, body: str = "do the thing") -> None:
    path = root / relpath / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")


class TestDelegateSkillTool:
    """delegate_skill (ftw_plan.md §3.2 Mode A): dispatched like any other
    bus tool — the caller's context grows only by the RESULT's summary,
    since that's all _reply_to_content ever extracts from a ResultEnvelope."""

    def test_delegate_skill_only_registered_when_target_given(self, tmp_path):
        with_target = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider([]),
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
            skill_runner_target="skill.runner",
        )
        assert "delegate_skill" in {t.name for t in with_target._tool_specs}  # noqa: SLF001

        without_target = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider([]),
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
        )
        assert "delegate_skill" not in {t.name for t in without_target._tool_specs}  # noqa: SLF001

    def test_delegate_skill_dispatches_to_the_configured_target_and_returns_the_summary(self, tmp_path):
        result_reply = ResultEnvelope(
            source="skill.runner",
            target="repl.master",
            payload=ResultPayload(status="ok", summary="Diagnosed and fixed the CMake issue.", outputs={"patch_applied": True}),
        )
        dispatcher = RecordingDispatcher([result_reply])
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider(
                [
                    assistant_tool_call("delegate_skill", {"name": "cmake.diagnose_configure", "brief": "fix the build"}),
                    assistant_text("the sub-task finished"),
                ]
            ),
            output_store=OutputStore(tmp_path),
            dispatch=dispatcher,
            interceptor=PreCommitInterceptor([]),
            skill_runner_target="skill.runner",
        )

        assert loop.run_turn("delegate this") == "the sub-task finished"
        assert len(dispatcher.calls) == 1
        assert dispatcher.calls[0].payload.action == "delegate_skill"
        assert dispatcher.calls[0].target == "skill.runner"
        # the caller's own turn holds only the summary text, never the
        # delegated run's internal steps
        tool_messages = [m for m in loop.workbench.turns[0] if m.role == ChatRole.TOOL]
        assert len(tool_messages) == 1
        assert "Diagnosed and fixed" in tool_messages[0].content


class TestFrameToolsIntegration:
    """find_skill/mount_skill/unmount_skill (ftw_plan.md §3.2
    "Self-Mounting"): local execution (no bus dispatch) but still routed
    through the interceptor, unlike pin/unpin/read_output/grep_output."""

    def make_loop_with_frames(self, tmp_path, responses, *, interceptor=None, ask_answerer=None):
        write_skill(tmp_path, "cmake/diagnose_configure", "cmake.diagnose_configure", "Diagnose failing CMake configuration.")
        store = SkillStore(tmp_path)
        workbench = ContextWorkbench()
        tree = FrameTree(workbench, store)
        loop = AgentLoop(
            workbench=workbench,
            provider=MockModelProvider(responses),
            output_store=OutputStore(tmp_path / "outputs"),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError("must not dispatch")),
            interceptor=interceptor or PreCommitInterceptor([]),
            ask_answerer=ask_answerer,
            frame_tree=tree,
        )
        return loop, tree

    def test_find_skill_returns_matching_names(self, tmp_path):
        loop, _tree = self.make_loop_with_frames(
            tmp_path,
            [
                assistant_tool_call("find_skill", {"query": "cmake configuration"}),
                assistant_text("found it"),
            ],
        )
        assert loop.run_turn("find a skill") == "found it"

    def test_mount_skill_mounts_into_the_shared_workbench(self, tmp_path):
        loop, tree = self.make_loop_with_frames(
            tmp_path,
            [
                assistant_tool_call("mount_skill", {"name": "cmake.diagnose_configure"}),
                assistant_text("mounted"),
            ],
        )
        assert loop.run_turn("mount cmake help") == "mounted"
        assert tree.focused_skill_name == "cmake.diagnose_configure"
        assert loop.workbench.mounted_skill_tokens > 0

    def test_mount_skill_unknown_name_reports_error_without_crashing(self, tmp_path):
        loop, _tree = self.make_loop_with_frames(
            tmp_path,
            [
                assistant_tool_call("mount_skill", {"name": "nope.nothing"}),
                assistant_text("no such skill"),
            ],
        )
        assert loop.run_turn("mount something bogus") == "no such skill"

    def test_unmount_skill_produces_a_milestone(self, tmp_path):
        loop, tree = self.make_loop_with_frames(
            tmp_path,
            [
                assistant_tool_call("mount_skill", {"name": "cmake.diagnose_configure"}),
                assistant_tool_call("unmount_skill", {"name": "cmake.diagnose_configure"}),
                assistant_text("done"),
            ],
        )
        assert loop.run_turn("mount then unmount") == "done"
        assert loop.workbench.mounted_skill_tokens == 0
        assert len(loop.workbench.milestones) == 1

    def test_frame_tools_go_through_the_interceptor(self, tmp_path):
        blocked = InterceptOutcome(InterceptDecision.BLOCK, reason="no mounting allowed right now")

        class BlockMounts:
            def evaluate(self, call):
                if call.payload.action == "mount_skill":
                    return blocked
                return InterceptOutcome(InterceptDecision.ALLOW)

        loop, tree = self.make_loop_with_frames(
            tmp_path,
            [
                assistant_tool_call("mount_skill", {"name": "cmake.diagnose_configure"}),
                assistant_text("blocked as expected"),
            ],
            interceptor=PreCommitInterceptor([BlockMounts()]),
        )

        assert loop.run_turn("try to mount") == "blocked as expected"
        assert tree.focused_skill_name is None  # never actually mounted

    def test_committed_turn_is_tagged_with_the_focused_frame(self, tmp_path):
        """Without this tagging, evict_frame() on unmount would find
        nothing to evict — the whole point of a mount/unmount frame."""
        loop, tree = self.make_loop_with_frames(
            tmp_path,
            [
                assistant_tool_call("mount_skill", {"name": "cmake.diagnose_configure"}),
                assistant_text("mounted and answered"),
            ],
        )
        loop.run_turn("mount cmake and answer")

        frame = tree._find_by_skill("cmake.diagnose_configure")  # noqa: SLF001 - whitebox
        evicted = loop.workbench.evict_frame(frame.id)
        assert len(evicted) == 1  # the committed turn was tagged to the frame that was focused when it committed

    def test_pin_made_while_a_frame_is_focused_is_tagged_to_it(self, tmp_path):
        loop, tree = self.make_loop_with_frames(
            tmp_path,
            [
                assistant_tool_call("mount_skill", {"name": "cmake.diagnose_configure"}),
                assistant_tool_call("pin", {"key": "finding", "value": "missing openssl"}),
                assistant_text("noted"),
            ],
        )
        loop.run_turn("mount and note a finding")

        frame = tree._find_by_skill("cmake.diagnose_configure")  # noqa: SLF001 - whitebox
        evicted_pins = loop.workbench.evict_frame_pins(frame.id)
        assert evicted_pins == {"finding": "missing openssl"}

    def test_turn_committed_with_nothing_focused_is_untagged(self, tmp_path):
        loop, _tree = self.make_loop_with_frames(tmp_path, [assistant_text("just chatting")])
        loop.run_turn("hello")
        assert loop.workbench.evict_frame("anything") == []  # nothing was tagged to any frame
        assert len(loop.workbench.turns) == 1  # the turn is still there, just untagged

    def test_mount_work_and_unmount_within_a_single_turn_leaves_nothing_behind(self, tmp_path):
        """The model's natural self-mount pattern: mount, do the work, and
        unmount all as tool calls within ONE reply — committed as a single
        turn. Per-turn tagging used to tag the whole turn with whatever was
        focused at commit time (i.e. nothing, since the unmount already
        ran) — so evict_frame found nothing to evict and the mounted
        skill's own conversation content leaked into the horizon forever.
        Per-message tagging fixes this: each message is tagged with focus
        as of when it was produced, so the frame's own messages are still
        findable and removable even though the frame was long since
        unfocused by the time the turn committed."""
        loop, tree = self.make_loop_with_frames(
            tmp_path,
            [
                assistant_tool_call("mount_skill", {"name": "cmake.diagnose_configure"}, call_id="c1"),
                assistant_tool_call("unmount_skill", {"name": "cmake.diagnose_configure"}, call_id="c2"),
                assistant_text("mounted, worked, and unmounted, all in one go"),
            ],
        )
        result = loop.run_turn("diagnose the cmake failure end to end")

        assert result == "mounted, worked, and unmounted, all in one go"
        assert tree.focused_skill_name is None
        # the invariant that actually matters here: the mounted skill's own
        # (potentially ~1500-token) body text is fully freed and exactly
        # one milestone was produced — mount/unmount both happening inside
        # this same not-yet-committed turn means evict_frame() (which acts
        # on already-committed history) has nothing of *this* turn's own
        # tool-call transcript to remove, and that's fine: those few lines
        # ("mounted ...", "unmounted; milestone: ...") are small, ordinary
        # conversational content, not the frame's bulk skill text.
        assert loop.workbench.mounted_skill_tokens == 0
        assert len(loop.workbench.milestones) == 1
        assert len(loop.workbench.turns) == 1
        tool_names_in_turn = [m.name for m in loop.workbench.turns[0] if m.role == ChatRole.TOOL]
        assert tool_names_in_turn == ["mount_skill", "unmount_skill"]

    def test_agent_loop_without_frame_tree_reports_frame_tools_as_unknown(self, tmp_path):
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider(
                [assistant_tool_call("mount_skill", {"name": "x"}), assistant_text("no skills here")]
            ),
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
        )
        assert loop.run_turn("mount x") == "no skills here"


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


def submit_result_call(status="ok", summary="done", outputs=None, evidence=None, call_id="s1") -> ProviderResponse:
    args = {"status": status, "summary": summary}
    if outputs is not None:
        args["outputs"] = outputs
    if evidence is not None:
        args["evidence"] = evidence
    return assistant_tool_call("submit_result", args, call_id=call_id)


class TestRunDelegated:
    """The delegated skill runner (ftw_plan.md §3.2 Mode A, §8 Phase 3): a
    fresh, isolated AgentLoop runs a skill to completion and produces one
    structured Result — the caller's own context never absorbs the
    delegated run's intermediate tool calls or deliberation."""

    def make_delegated_loop(self, tmp_path, responses, **kwargs):
        return AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider(responses),
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError("must not dispatch")),
            enable_delegated_completion=True,
            **kwargs,
        )

    def test_submit_result_produces_a_result_payload(self, tmp_path):
        loop = self.make_delegated_loop(
            tmp_path,
            [submit_result_call(summary="Identified missing OpenSSL headers.", outputs={"patch_applied": True}, evidence=["grep -i openssl log"])],
        )

        result = loop.run_delegated("diagnose the cmake failure")

        assert isinstance(result, ResultPayload)
        assert result.status == "ok"
        assert result.summary == "Identified missing OpenSSL headers."
        assert result.outputs == {"patch_applied": True}
        assert result.evidence == ["grep -i openssl log"]
        assert "wall_time_ms" in result.cost

    def test_error_status_is_preserved(self, tmp_path):
        loop = self.make_delegated_loop(tmp_path, [submit_result_call(status="error", summary="could not reproduce")])
        result = loop.run_delegated("diagnose it")
        assert result.status == "error"
        assert result.summary == "could not reproduce"

    def test_tool_calls_before_submit_result_are_handled_normally(self, tmp_path):
        loop = self.make_delegated_loop(
            tmp_path,
            [
                assistant_tool_call("pin", {"key": "finding", "value": "missing openssl"}),
                submit_result_call(summary="done"),
            ],
        )
        result = loop.run_delegated("diagnose it")
        assert result.status == "ok"
        assert loop.workbench.scratchpad == {"finding": "missing openssl"}  # ordinary tool handling still applies

    def test_plain_text_without_submit_result_is_a_forgiving_fallback(self, tmp_path):
        loop = self.make_delegated_loop(tmp_path, [assistant_text("I looked and everything's fine.")])
        result = loop.run_delegated("check it")
        assert result.status == "ok"
        assert result.summary == "I looked and everything's fine."

    def test_step_budget_exceeded_without_submit_result_is_an_error_result(self, tmp_path):
        responses = [assistant_tool_call("pin", {"key": "k", "value": "v"}, call_id=f"c{i}") for i in range(3)]
        loop = self.make_delegated_loop(tmp_path, responses, max_steps=3)
        result = loop.run_delegated("loop forever")
        assert result.status == "error"
        assert "budget" in result.summary.lower()

    def test_run_delegated_without_enable_flag_raises(self, tmp_path):
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider([]),
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
        )
        with pytest.raises(RuntimeError):
            loop.run_delegated("anything")

    def test_submit_result_tool_only_appears_when_enabled(self, tmp_path):
        delegated = self.make_delegated_loop(tmp_path, [submit_result_call()])
        assert "submit_result" in {t.name for t in delegated._tool_specs}  # noqa: SLF001 - whitebox

        interactive = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider([]),
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
        )
        assert "submit_result" not in {t.name for t in interactive._tool_specs}  # noqa: SLF001


class TestDelegatedAskPassthrough:
    """A delegated run's own interceptor ASK (e.g. a shell command it
    proposes needs confirmation) can't block on a local confirm() — there's
    no human at that process. run_delegated suspends instead, returning a
    DelegatedSuspension the caller relays; resume_delegated continues from
    exactly where it paused once an answer arrives (ftw_plan.md §8 Phase 3
    "ASK passthrough to the REPL")."""

    def make_loop(self, tmp_path, responses, dispatch=None, **kwargs):
        return AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider(responses),
            output_store=OutputStore(tmp_path),
            dispatch=dispatch or (lambda call: (_ for _ in ()).throw(AssertionError("must not dispatch before an answer"))),
            interceptor=PreCommitInterceptor([ConfirmShellCommands()]),
            enable_delegated_completion=True,
            **kwargs,
        )

    def test_run_delegated_suspends_instead_of_blocking(self, tmp_path):
        loop = self.make_loop(tmp_path, [assistant_tool_call("run_command", {"argv": ["rm", "-rf", "build"]})])

        outcome = loop.run_delegated("clean the build dir")

        assert isinstance(outcome, DelegatedSuspension)
        assert outcome.ask.payload.resume_token
        assert outcome.ask.payload.question

    def test_resume_with_true_dispatches_and_continues(self, tmp_path):
        dispatcher = RecordingDispatcher([ResultEnvelope(source="w", target="c", payload=ResultPayload(status="ok", summary="exit 0"))])
        loop = self.make_loop(
            tmp_path,
            [assistant_tool_call("run_command", {"argv": ["rm", "-rf", "build"]}), submit_result_call(summary="cleaned")],
            dispatch=dispatcher,
        )
        suspension = loop.run_delegated("clean the build dir")

        outcome = loop.resume_delegated(suspension, True)

        assert isinstance(outcome, ResultPayload)
        assert outcome.status == "ok"
        assert outcome.summary == "cleaned"
        assert len(dispatcher.calls) == 1
        assert dispatcher.calls[0].payload.action == "run_command"

    def test_resume_with_false_declines_and_continues(self, tmp_path):
        loop = self.make_loop(
            tmp_path,
            [assistant_tool_call("run_command", {"argv": ["rm", "-rf", "build"]}), submit_result_call(summary="left it alone")],
        )
        suspension = loop.run_delegated("clean the build dir")

        outcome = loop.resume_delegated(suspension, False)

        assert isinstance(outcome, ResultPayload)
        assert outcome.summary == "left it alone"

    def test_second_call_in_same_batch_runs_after_the_first_is_answered(self, tmp_path):
        dispatcher = RecordingDispatcher([ResultEnvelope(source="w", target="c", payload=ResultPayload(status="ok", summary="exit 0"))])
        response = ProviderResponse(
            message=ChatMessage(
                role=ChatRole.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall(id="a", name="run_command", arguments={"argv": ["rm", "x"]}),
                    ToolCall(id="b", name="pin", arguments={"key": "k", "value": "v"}),
                ],
            )
        )
        loop = self.make_loop(tmp_path, [response, submit_result_call(summary="done")], dispatch=dispatcher)

        suspension = loop.run_delegated("go")
        outcome = loop.resume_delegated(suspension, True)

        assert isinstance(outcome, ResultPayload)
        assert loop.workbench.scratchpad == {"k": "v"}  # the second tool_call in the batch still ran

    def test_a_second_ask_in_a_later_step_produces_a_new_suspension(self, tmp_path):
        loop = self.make_loop(
            tmp_path,
            [
                assistant_tool_call("run_command", {"argv": ["rm", "a"]}, call_id="a"),
                assistant_tool_call("run_command", {"argv": ["rm", "b"]}, call_id="b"),
                submit_result_call(summary="both removed"),
            ],
            dispatch=RecordingDispatcher(
                [
                    ResultEnvelope(source="w", target="c", payload=ResultPayload(status="ok", summary="exit 0")),
                    ResultEnvelope(source="w", target="c", payload=ResultPayload(status="ok", summary="exit 0")),
                ]
            ),
        )

        first = loop.run_delegated("clean up")
        assert isinstance(first, DelegatedSuspension)
        second = loop.resume_delegated(first, True)
        assert isinstance(second, DelegatedSuspension)
        assert second.ask.payload.resume_token != first.ask.payload.resume_token
        third = loop.resume_delegated(second, True)
        assert isinstance(third, ResultPayload)
        assert third.summary == "both removed"

    def test_cost_is_still_reported_after_a_suspend_and_resume(self, tmp_path):
        loop = self.make_loop(
            tmp_path,
            [assistant_tool_call("run_command", {"argv": ["rm", "x"]}), submit_result_call(summary="done")],
            dispatch=RecordingDispatcher([ResultEnvelope(source="w", target="c", payload=ResultPayload(status="ok", summary="exit 0"))]),
        )
        suspension = loop.run_delegated("go")
        outcome = loop.resume_delegated(suspension, True)
        assert "wall_time_ms" in outcome.cost


class TestDelegatedCancellation:
    """Cooperative cancellation: checked between steps, since a blocked
    provider.complete() call can't be preempted."""

    def test_cancel_flag_set_before_starting_aborts_immediately(self, tmp_path):
        flag = threading.Event()
        flag.set()
        provider = MockModelProvider([submit_result_call()])
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=provider,
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
            enable_delegated_completion=True,
        )

        result = loop.run_delegated("go", cancel_flag=flag)

        assert result.status == "error"
        assert "cancel" in result.summary.lower()
        assert provider.calls == []  # never even asked the model

    def test_cancel_flag_set_between_steps_stops_before_the_next_one(self, tmp_path):
        flag = threading.Event()

        class SetsFlagAfterFirstCall(MockModelProvider):
            def complete(self, messages, tools=None):
                response = super().complete(messages, tools)
                flag.set()
                return response

        provider = SetsFlagAfterFirstCall(
            [assistant_tool_call("pin", {"key": "k", "value": "v"}), submit_result_call(summary="should not get here")]
        )
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=provider,
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
            enable_delegated_completion=True,
        )

        result = loop.run_delegated("go", cancel_flag=flag)

        assert result.status == "error"
        assert "cancel" in result.summary.lower()
        assert len(provider.calls) == 1  # stopped before the second step

    def test_resume_also_respects_a_cancel_flag(self, tmp_path):
        flag = threading.Event()
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider([assistant_tool_call("run_command", {"argv": ["x"]})]),
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
            interceptor=PreCommitInterceptor([ConfirmShellCommands()]),
            enable_delegated_completion=True,
        )
        suspension = loop.run_delegated("go", cancel_flag=flag)
        flag.set()

        result = loop.resume_delegated(suspension, True)

        assert result.status == "error"
        assert "cancel" in result.summary.lower()


class TestKeyboardInterruptDuringDispatch:
    """If a KeyboardInterrupt is ever raised while dispatching a bus call
    (whatever the trigger — see bus.py's Requester.call() docstring for
    the current limitation on Ctrl-C specifically getting one raised
    promptly), it must be handled cleanly: notify the worker via
    control_dispatch, report the tool call as cancelled, and let the loop
    continue rather than crashing the whole session."""

    def test_notifies_via_control_dispatch_and_reports_cancelled(self, tmp_path):
        control_calls = []

        def dispatch(call):
            raise KeyboardInterrupt()

        def control_dispatch(envelope):
            control_calls.append(envelope)
            return ResultEnvelope(source="w", target="c", payload=ResultPayload(status="ok", summary="cancel requested"))

        loop = make_loop(
            responses=[
                assistant_tool_call("run_command", {"argv": ["sleep", "100"]}),
                assistant_text("cancelled that for you"),
            ],
            dispatch=dispatch,
            interceptor=PreCommitInterceptor([]),
            output_root=tmp_path,
        )
        loop._control_dispatch = control_dispatch  # noqa: SLF001 - whitebox: no public setter, and shouldn't need one

        result = loop.run_turn("run something long")

        assert result == "cancelled that for you"
        assert len(control_calls) == 1
        assert control_calls[0].payload.target_span_id
        assert control_calls[0].payload.reason == "Ctrl-C"

    def test_still_reports_cancelled_when_no_control_dispatch_is_configured(self, tmp_path):
        loop = make_loop(
            responses=[
                assistant_tool_call("run_command", {"argv": ["sleep", "100"]}),
                assistant_text("okay, stopped"),
            ],
            dispatch=lambda call: (_ for _ in ()).throw(KeyboardInterrupt()),
            interceptor=PreCommitInterceptor([]),
            output_root=tmp_path,
        )
        assert loop.run_turn("run something long") == "okay, stopped"

    def test_a_failing_control_dispatch_does_not_crash_the_loop(self, tmp_path):
        def control_dispatch(envelope):
            raise ConnectionError("worker unreachable")

        loop = make_loop(
            responses=[
                assistant_tool_call("run_command", {"argv": ["sleep", "100"]}),
                assistant_text("gave up cleanly"),
            ],
            dispatch=lambda call: (_ for _ in ()).throw(KeyboardInterrupt()),
            interceptor=PreCommitInterceptor([]),
            output_root=tmp_path,
        )
        loop._control_dispatch = control_dispatch  # noqa: SLF001

        assert loop.run_turn("run something long") == "gave up cleanly"


class TestWorkerInitiatedAskRelay:
    """A worker mid-computation can reply ASK instead of RESULT/ERROR
    (ftw_plan.md §4 "Mid-call interaction"). Distinct from the
    interceptor's own ASK decision (already covered above): here the
    *worker itself*, not the interceptor, needs an answer before it can
    finish. Relayed over the same dispatch/target — a worker's ANSWER
    reply-address is the same address its CALL went to."""

    def test_answers_and_redispatches_until_a_result(self, tmp_path):
        ask = AskEnvelope(
            source="worker.tool.shell", target="repl.master", payload=AskPayload(question="Overwrite existing file?", resume_token="tok-1")
        )
        result = ResultEnvelope(source="worker.tool.shell", target="repl.master", payload=ResultPayload(status="ok", summary="overwrote it"))
        dispatcher = RecordingDispatcher([ask, result])
        loop = make_loop(
            responses=[assistant_tool_call("run_command", {"argv": ["cp", "a", "b"]}), assistant_text("done")],
            dispatch=dispatcher,
            interceptor=PreCommitInterceptor([]),
            output_root=tmp_path,
        )

        assert loop.run_turn("copy a to b") == "done"
        assert len(dispatcher.calls) == 2

        answer_call = dispatcher.calls[1]
        assert isinstance(answer_call, AnswerEnvelope)
        assert answer_call.payload.resume_token == "tok-1"
        assert answer_call.payload.value is False  # default ask_answerer fails closed

    def test_custom_ask_answerer_supplies_the_answer_value(self, tmp_path):
        ask = AskEnvelope(source="w", target="repl.master", payload=AskPayload(question="proceed?", resume_token="tok-2"))
        result = ResultEnvelope(source="w", target="repl.master", payload=ResultPayload(status="ok", summary="proceeded"))
        dispatcher = RecordingDispatcher([ask, result])
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider([assistant_tool_call("run_command", {"argv": ["x"]}), assistant_text("ok")]),
            output_store=OutputStore(tmp_path),
            dispatch=dispatcher,
            interceptor=PreCommitInterceptor([]),
            ask_answerer=lambda ask_env: "yes please",
        )
        loop.run_turn("go")

        answer_call = dispatcher.calls[1]
        assert isinstance(answer_call, AnswerEnvelope)
        assert answer_call.payload.value == "yes please"

    def test_bounded_rounds_prevents_an_infinite_ask_loop(self, tmp_path):
        always_ask = AskEnvelope(source="w", target="repl.master", payload=AskPayload(question="again?", resume_token="tok-3"))
        dispatcher = RecordingDispatcher([always_ask] * 50)  # more than the round cap
        loop = make_loop(
            responses=[assistant_tool_call("run_command", {"argv": ["x"]}), assistant_text("gave up")],
            dispatch=dispatcher,
            interceptor=PreCommitInterceptor([]),
            output_root=tmp_path,
        )
        assert loop.run_turn("go") == "gave up"
        assert len(dispatcher.calls) < 50  # bailed out well before exhausting the script


class TestAnswerMustBeExactlyTrue:
    """A free-text answer to an interceptor ASK ("nope", "cancel", any
    non-empty string) must decline, not approve. bool("nope") is True in
    Python — coercing through bool() would have silently approved a
    command someone was trying to decline."""

    def test_free_text_declines_rather_than_approves(self, tmp_path):
        loop = make_loop(
            responses=[assistant_tool_call("run_command", {"argv": ["rm", "-rf", "/"]}), assistant_text("didn't run it")],
            interceptor=PreCommitInterceptor([ConfirmShellCommands()]),
            ask_answerer=lambda ask: "nope",  # someone trying to decline in words, not "y"/"n"
            output_root=tmp_path,
        )
        assert loop.run_turn("delete everything") == "didn't run it"

    def test_the_string_false_also_declines(self, tmp_path):
        loop = make_loop(
            responses=[assistant_tool_call("run_command", {"argv": ["rm", "-rf", "/"]}), assistant_text("didn't run it")],
            interceptor=PreCommitInterceptor([ConfirmShellCommands()]),
            ask_answerer=lambda ask: "False",  # a non-empty string is truthy in Python; must not be treated as approval
            output_root=tmp_path,
        )
        assert loop.run_turn("delete everything") == "didn't run it"

    def test_exactly_true_still_approves(self, tmp_path):
        dispatcher = RecordingDispatcher(
            [ResultEnvelope(source="w", target="c", payload=ResultPayload(status="ok", summary="exit 0"))]
        )
        loop = make_loop(
            responses=[assistant_tool_call("run_command", {"argv": ["ls"]}), assistant_text("listed")],
            dispatch=dispatcher,
            interceptor=PreCommitInterceptor([ConfirmShellCommands()]),
            ask_answerer=lambda ask: True,
            output_root=tmp_path,
        )
        assert loop.run_turn("list files") == "listed"
        assert len(dispatcher.calls) == 1


class TestAskShowsTheActualCall:
    """The relayed question must show what's actually being approved —
    not just the interceptor's generic reason — so a human isn't asked to
    approve blind."""

    def test_ask_question_includes_the_tool_name_and_arguments(self, tmp_path):
        seen_questions = []

        def answerer(ask):
            seen_questions.append(ask.payload.question)
            return False

        loop = make_loop(
            responses=[
                assistant_tool_call("run_command", {"argv": ["rm", "-rf", "build"]}),
                assistant_text("held off"),
            ],
            interceptor=PreCommitInterceptor([ConfirmShellCommands()]),
            ask_answerer=answerer,
            output_root=tmp_path,
        )
        loop.run_turn("clean up")

        assert len(seen_questions) == 1
        assert "run_command" in seen_questions[0]
        assert "rm" in seen_questions[0]
        assert "build" in seen_questions[0]

    def test_ask_payload_expected_carries_the_structured_call(self, tmp_path):
        seen_asks = []

        def answerer(ask):
            seen_asks.append(ask)
            return False

        loop = make_loop(
            responses=[assistant_tool_call("run_command", {"argv": ["ls", "-la"]}), assistant_text("held off")],
            interceptor=PreCommitInterceptor([ConfirmShellCommands()]),
            ask_answerer=answerer,
            output_root=tmp_path,
        )
        loop.run_turn("list")

        assert seen_asks[0].payload.expected == {"action": "run_command", "args": {"argv": ["ls", "-la"]}}


class TestProviderErrorHandling:
    """A model-call failure (a 429, a malformed response, ...) must not
    crash the whole turn — it becomes a clear, reported failure instead."""

    class RaisingProvider:
        def __init__(self, exc):
            self._exc = exc

        def complete(self, messages, tools=None):
            raise self._exc

    def test_run_turn_reports_a_provider_error_instead_of_crashing(self, tmp_path):
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=self.RaisingProvider(ProviderError("rate limited (429)")),
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
        )
        result = loop.run_turn("hello")
        assert "model error" in result.lower()
        assert "429" in result

    def test_run_turn_still_commits_whatever_was_already_in_the_turn(self, tmp_path):
        workbench = ContextWorkbench()
        loop = AgentLoop(
            workbench=workbench,
            provider=self.RaisingProvider(ProviderError("boom")),
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
        )
        loop.run_turn("hello")
        assert len(workbench.turns) == 1
        assert workbench.turns[0][0].content == "hello"

    def test_run_delegated_reports_a_provider_error_as_an_error_result(self, tmp_path):
        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=self.RaisingProvider(ProviderError("boom")),
            output_store=OutputStore(tmp_path),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
            enable_delegated_completion=True,
        )
        result = loop.run_delegated("go")
        assert result.status == "error"
        assert "model error" in result.summary.lower()


class TestDeadlineExceededHandling:
    """A bus call that times out must be reported as a tool result, not
    left to crash the turn."""

    def test_dispatch_timeout_is_reported_not_raised(self, tmp_path):
        def timing_out(call):
            raise DeadlineExceeded("timed out waiting for reply")

        loop = make_loop(
            responses=[assistant_tool_call("run_command", {"argv": ["sleep", "100"]}), assistant_text("gave up waiting")],
            dispatch=timing_out,
            interceptor=PreCommitInterceptor([]),
            output_root=tmp_path,
        )
        assert loop.run_turn("run something slow") == "gave up waiting"


class TestCallDeadlinesAreConsistent:
    """Every dispatched CALL needs a deadline comfortably longer than
    whatever it wraps — the shell worker's own 60s command timeout, or a
    delegated run's several model round-trips — not the bus's generic 30s
    default, which used to be shorter than both."""

    def test_run_command_gets_a_deadline_longer_than_the_shell_workers_own_timeout(self, tmp_path):
        captured = []

        def dispatch(call):
            captured.append(call)
            return ResultEnvelope(source="w", target="c", payload=ResultPayload(status="ok", summary="exit 0"))

        loop = make_loop(
            responses=[assistant_tool_call("run_command", {"argv": ["ls"]}), assistant_text("done")],
            dispatch=dispatch,
            interceptor=PreCommitInterceptor([]),
            output_root=tmp_path,
        )
        loop.run_turn("list")

        assert captured[0].deadline_ms is not None
        assert captured[0].deadline_ms > 60_000  # the shell worker's own default command timeout

    def test_delegate_skill_gets_a_generous_deadline(self, tmp_path):
        captured = []

        def dispatch(call):
            captured.append(call)
            return ResultEnvelope(source="w", target="c", payload=ResultPayload(status="ok", summary="done"))

        loop = AgentLoop(
            workbench=ContextWorkbench(),
            provider=MockModelProvider(
                [assistant_tool_call("delegate_skill", {"name": "x", "brief": "go"}), assistant_text("done")]
            ),
            output_store=OutputStore(tmp_path),
            dispatch=dispatch,
            skill_runner_target="skill.runner",
        )
        loop.run_turn("delegate")

        assert captured[0].deadline_ms is not None
        assert captured[0].deadline_ms >= 120_000  # room for several model round-trips
