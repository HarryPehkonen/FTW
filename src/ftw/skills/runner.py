"""The delegated skill runner: an isolated out-of-process worker
(ftw_plan.md §3.2 Mode A, §8 Phase 3).

Each CALL gets a fresh, isolated ContextWorkbench holding only the skill's
body — no shared session state, no other frames. The callee runs to a
*structured* completion (AgentLoop.run_delegated, driven by the
``submit_result`` tool) and replies with one compact RESULT envelope. The
caller's own context grows only by that envelope — never by the
delegated run's intermediate tool calls, dispatched-worker traffic, or
deliberation.

Increment scope: a delegated run's own Pre-Commit Interceptor decisions
are resolved with an empty policy by default (every bus tool call is
allowed without confirmation) — there's no local human at this process to
ask. Relaying an interceptor ASK back through the caller to a real human
mid-run is a distinct, larger piece of work than shipped here; a worker
that itself replies ASK (ftw_plan.md §4 "Mid-call interaction") already
works today via AgentLoop's own ask/answer relay (agent_loop.py).
"""

from __future__ import annotations

import argparse
import sys
import threading
from typing import Callable

from ftw.agent_loop import AgentLoop, Dispatch
from ftw.bus import Replier, Requester
from ftw.config import build_provider, load_config
from ftw.intercept import PreCommitInterceptor
from ftw.outputs import OutputStore
from ftw.protocol import CallEnvelope, ErrorEnvelope, ErrorPayload, ResultEnvelope
from ftw.providers import IModelProvider
from ftw.runtime import ipc_address
from ftw.skills.manifest import render_frame_text
from ftw.skills.registry import SkillNotFound, SkillStore
from ftw.workbench import ContextWorkbench

ProviderFactory = Callable[[str | None], IModelProvider]

# Guidance from ftw_plan.md §3.2 Mode A ("caller brief ≤ 300 tokens") — not
# enforced yet, just surfaced so a caller sees it during development.
RECOMMENDED_MAX_BRIEF_TOKENS = 300


class SkillRunnerWorker:
    def __init__(
        self,
        skill_store: SkillStore,
        *,
        provider_factory: ProviderFactory,
        output_store: OutputStore,
        dispatch: Dispatch,
        interceptor: PreCommitInterceptor | None = None,
        default_max_steps: int = 15,
        source_id: str = "skill.runner",
    ):
        self._skills = skill_store
        self._provider_factory = provider_factory
        self._outputs = output_store
        self._dispatch = dispatch
        self._interceptor = interceptor or PreCommitInterceptor([])
        self._default_max_steps = default_max_steps
        self._source_id = source_id

    def handle(self, call: CallEnvelope) -> ResultEnvelope | ErrorEnvelope:
        # Matches the "delegate_skill" tool name agent_loop.py exposes to the
        # model (CallPayload.action is always the tool_call name, same
        # convention as run_command) — not an independent worker-side name.
        if call.payload.action != "delegate_skill":
            return self._error(call, "unsupported_action", f"unsupported action: {call.payload.action!r}")

        name = call.payload.args.get("name")
        if not name:
            return self._error(call, "invalid_args", "args.name is required")
        brief = call.payload.args.get("brief", "")
        inputs = call.payload.args.get("inputs") or {}

        try:
            manifest, body = self._skills.get(name)
        except SkillNotFound as exc:
            return self._error(call, "not_found", str(exc))

        workbench = ContextWorkbench()
        workbench.set_mounted_skill_text(render_frame_text(manifest, body))

        loop = AgentLoop(
            workbench=workbench,
            provider=self._provider_factory(manifest.model),
            output_store=self._outputs,
            dispatch=self._dispatch,
            interceptor=self._interceptor,
            enable_delegated_completion=True,
            max_steps=manifest.budget.max_steps or self._default_max_steps,
            source_id=self._source_id,
            trace_id=call.trace_id,
        )

        full_brief = self._render_brief(brief, inputs)
        result_payload = loop.run_delegated(full_brief)

        return ResultEnvelope(
            source=call.target,
            target=call.source,
            parent_span_id=call.span_id,
            trace_id=call.trace_id,
            payload=result_payload,
        )

    @staticmethod
    def _render_brief(brief: str, inputs: dict) -> str:
        if not inputs:
            return brief
        input_lines = "\n".join(f"- {k}: {v}" for k, v in inputs.items())
        return f"{brief}\n\nInputs:\n{input_lines}"

    @staticmethod
    def _error(call: CallEnvelope, code: str, message: str) -> ErrorEnvelope:
        return ErrorEnvelope(
            source=call.target,
            target=call.source,
            parent_span_id=call.span_id,
            trace_id=call.trace_id,
            payload=ErrorPayload(code=code, message=message),
        )


def run_worker(
    address: str,
    *,
    skills_dir: str,
    output_root: str,
    shell_worker_address: str,
    config_path: str,
    default_tier: str,
    stop_event: threading.Event | None = None,
    num_workers: int = 4,
) -> None:
    """Blocks, serving CALLs, until ``stop_event`` is set."""
    config = load_config(config_path)
    shell_requester = Requester(shell_worker_address)
    worker = SkillRunnerWorker(
        SkillStore(skills_dir),
        provider_factory=lambda tier: build_provider(config, tier or default_tier),
        output_store=OutputStore(output_root),
        dispatch=shell_requester.call,
    )
    try:
        with Replier(address, num_workers=num_workers) as rep:
            rep.serve_forever(worker.handle, stop_event)
    finally:
        shell_requester.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FTW delegated skill runner worker")
    parser.add_argument("--address", default=ipc_address("skill.runner"))
    parser.add_argument("--skills-dir", default="skills")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--shell-worker-address", required=True)
    parser.add_argument("--config", default="ftw.toml")
    parser.add_argument("--default-tier", default="fast")
    args = parser.parse_args(argv)
    print(f"ftw skill runner listening on {args.address}", file=sys.stderr)
    try:
        run_worker(
            args.address,
            skills_dir=args.skills_dir,
            output_root=args.output_root,
            shell_worker_address=args.shell_worker_address,
            config_path=args.config,
            default_tier=args.default_tier,
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
