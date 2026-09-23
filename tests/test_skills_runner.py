"""SkillRunnerWorker: the delegated skill worker, as an independent NNG
service (ftw_plan.md §3.2 Mode A, §8 Phase 3).

A fresh, isolated workbench per call; the caller's own context grows only
by the RESULT envelope this produces, never by the delegated run's
intermediate tool calls or deliberation.
"""

import sys
import threading

from ftw.bus import Replier, Requester
from ftw.outputs import OutputStore
from ftw.protocol import CallEnvelope, CallPayload, ErrorEnvelope, ResultEnvelope
from ftw.providers import ChatMessage, ChatRole, MockModelProvider, ProviderResponse, ToolCall
from ftw.runtime import inproc_address
from ftw.skills.registry import SkillStore
from ftw.skills.runner import SkillRunnerWorker


def write_skill(root, relpath: str, name: str, description: str, body: str = "do the thing", model: str | None = None) -> None:
    path = root / relpath / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    model_line = f"model: {model}\n" if model else ""
    path.write_text(f"---\nname: {name}\ndescription: {description}\n{model_line}---\n\n{body}\n")


def submit_result_response(status="ok", summary="done", outputs=None, evidence=None) -> ProviderResponse:
    args = {"status": status, "summary": summary}
    if outputs is not None:
        args["outputs"] = outputs
    if evidence is not None:
        args["evidence"] = evidence
    return ProviderResponse(
        message=ChatMessage(role=ChatRole.ASSISTANT, content=None, tool_calls=[ToolCall(id="s1", name="submit_result", arguments=args)])
    )


def delegate_skill_call(name: str, brief: str = "do it", inputs: dict | None = None, **kwargs) -> CallEnvelope:
    args = {"name": name, "brief": brief}
    if inputs is not None:
        args["inputs"] = inputs
    return CallEnvelope(source="repl.master", target="skill.runner", payload=CallPayload(action="delegate_skill", args=args), **kwargs)


class TestSkillRunnerWorker:
    def make_worker(self, tmp_path, responses, *, provider_factory=None) -> SkillRunnerWorker:
        write_skill(tmp_path / "skills", "cmake/diagnose_configure", "cmake.diagnose_configure", "Diagnose failing CMake configuration.")
        store = SkillStore(tmp_path / "skills")
        provider = MockModelProvider(responses)
        return SkillRunnerWorker(
            store,
            provider_factory=provider_factory or (lambda tier: provider),
            output_store=OutputStore(tmp_path / "outputs"),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError("must not dispatch")),
        )

    def test_runs_to_completion_and_returns_a_result_envelope(self, tmp_path):
        worker = self.make_worker(tmp_path, [submit_result_response(summary="Found the missing header.")])

        reply = worker.handle(delegate_skill_call("cmake.diagnose_configure"))

        assert isinstance(reply, ResultEnvelope)
        assert reply.payload.status == "ok"
        assert reply.payload.summary == "Found the missing header."
        assert reply.parent_span_id is not None

    def test_only_the_result_crosses_back_nothing_else(self, tmp_path):
        """The caller only ever sees this one envelope — proving the
        deliverable directly: its own context can only grow by this much."""
        worker = self.make_worker(
            tmp_path,
            [
                # several internal steps before completion — none of this
                # should be visible to whoever called handle()
                ProviderResponse(
                    message=ChatMessage(
                        role=ChatRole.ASSISTANT, content=None, tool_calls=[ToolCall(id="p1", name="pin", arguments={"key": "k", "value": "v"})]
                    )
                ),
                submit_result_response(summary="wrapped up", outputs={"patch_applied": True}, evidence=["log line 1", "log line 2"]),
            ],
        )

        reply = worker.handle(delegate_skill_call("cmake.diagnose_configure"))

        assert isinstance(reply, ResultEnvelope)
        # the envelope's own shape is the whole contract: status/summary/outputs/evidence/cost
        assert set(reply.payload.model_dump().keys()) == {"status", "summary", "outputs", "evidence", "cost"}
        assert reply.payload.outputs == {"patch_applied": True}
        assert reply.payload.evidence == ["log line 1", "log line 2"]
        assert "wall_time_ms" in reply.payload.cost

    def test_unknown_skill_returns_not_found_error(self, tmp_path):
        worker = self.make_worker(tmp_path, [])
        reply = worker.handle(delegate_skill_call("nope.nothing"))
        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "not_found"

    def test_missing_name_returns_invalid_args_error(self, tmp_path):
        worker = self.make_worker(tmp_path, [])
        call = CallEnvelope(source="repl.master", target="skill.runner", payload=CallPayload(action="delegate_skill", args={"brief": "x"}))
        reply = worker.handle(call)
        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "invalid_args"

    def test_unsupported_action_returns_error(self, tmp_path):
        worker = self.make_worker(tmp_path, [])
        call = CallEnvelope(source="repl.master", target="skill.runner", payload=CallPayload(action="not_a_real_action", args={}))
        reply = worker.handle(call)
        assert isinstance(reply, ErrorEnvelope)
        assert reply.payload.code == "unsupported_action"

    def test_provider_factory_is_given_the_skills_model_tier(self, tmp_path):
        write_skill(tmp_path / "skills", "git/status_check", "git.status_check", "Check git status.", model="smart")
        store = SkillStore(tmp_path / "skills")
        seen_tiers = []

        def factory(tier):
            seen_tiers.append(tier)
            return MockModelProvider([submit_result_response()])

        worker = SkillRunnerWorker(
            store,
            provider_factory=factory,
            output_store=OutputStore(tmp_path / "outputs"),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
        )
        worker.handle(delegate_skill_call("git.status_check"))

        assert seen_tiers == ["smart"]

    def test_inputs_are_included_in_the_brief(self, tmp_path):
        captured_prompts = []

        class RecordingProvider:
            def complete(self, messages, tools=None):
                captured_prompts.append(messages)
                return submit_result_response()

        worker = self.make_worker(tmp_path, [], provider_factory=lambda tier: RecordingProvider())
        worker.handle(delegate_skill_call("cmake.diagnose_configure", brief="diagnose it", inputs={"repo_dir": "/src/myrepo"}))

        first_prompt = captured_prompts[0]
        assert any("/src/myrepo" in (m.content or "") for m in first_prompt)

    def test_dispatch_is_used_for_bus_tools_inside_the_delegated_run(self, tmp_path):
        from ftw.protocol import ResultPayload

        calls = []

        def dispatch(call):
            calls.append(call)
            return ResultEnvelope(source=call.target, target=call.source, payload=ResultPayload(status="ok", summary="exit 0"))

        write_skill(tmp_path / "skills", "cmake/diagnose_configure", "cmake.diagnose_configure", "Diagnose CMake.")
        store = SkillStore(tmp_path / "skills")
        responses = [
            ProviderResponse(
                message=ChatMessage(
                    role=ChatRole.ASSISTANT, content=None, tool_calls=[ToolCall(id="r1", name="run_command", arguments={"argv": ["ls"]})]
                )
            ),
            submit_result_response(),
        ]
        worker = SkillRunnerWorker(
            store,
            provider_factory=lambda tier: MockModelProvider(responses),
            output_store=OutputStore(tmp_path / "outputs"),
            dispatch=dispatch,
        )
        worker.handle(delegate_skill_call("cmake.diagnose_configure"))

        assert len(calls) == 1
        assert calls[0].payload.action == "run_command"


class TestSkillRunnerOverBus:
    def test_served_over_nng_end_to_end(self, tmp_path, unique_name):
        write_skill(tmp_path / "skills", "cmake/diagnose_configure", "cmake.diagnose_configure", "Diagnose CMake.")
        store = SkillStore(tmp_path / "skills")
        provider = MockModelProvider([submit_result_response(summary="over the bus")])
        worker = SkillRunnerWorker(
            store,
            provider_factory=lambda tier: provider,
            output_store=OutputStore(tmp_path / "outputs"),
            dispatch=lambda call: (_ for _ in ()).throw(AssertionError()),
        )
        addr = inproc_address(unique_name)
        stop = threading.Event()

        with Replier(addr) as rep:
            t = threading.Thread(target=rep.serve_forever, args=(worker.handle, stop), daemon=True)
            t.start()
            with Requester(addr) as req:
                reply = req.call(delegate_skill_call("cmake.diagnose_configure"))
            assert isinstance(reply, ResultEnvelope)
            assert reply.payload.summary == "over the bus"
            stop.set()
