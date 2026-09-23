"""The delegated skill runner: an isolated out-of-process worker
(ftw_plan.md §3.2 Mode A, §8 Phase 3).

Each CALL gets a fresh, isolated ContextWorkbench holding only the skill's
body — no shared session state, no other frames. The callee runs to a
*structured* completion (AgentLoop.run_delegated, driven by the
``submit_result`` tool) and replies with one compact RESULT envelope. The
caller's own context grows only by that envelope — never by the
delegated run's intermediate tool calls, dispatched-worker traffic, or
deliberation.

**ASK passthrough.** There's no local human at this process to answer a
delegated run's own interceptor ASK. So when one comes up,
``AgentLoop.run_delegated`` suspends instead of blocking, and ``handle()``
replies with an ``AskEnvelope`` instead of a ``ResultEnvelope`` — stashing
just enough state, keyed by resume_token, to pick the run back up exactly
where it paused once the matching ``AnswerEnvelope`` arrives (``handle()``
accepts that too). To the caller, this is indistinguishable from any other
worker replying ASK mid-computation (ftw_plan.md §4 "Mid-call
interaction"), so no protocol extension was needed for it.

**CANCEL.** A second Replier on the control channel (``<address>.ctl``,
matching ftw_plan.md §4's socket table) accepts CANCEL envelopes and sets
a per-span_id flag that a run's loop checks cooperatively between steps —
see agent_loop.py's ``cancel_flag``. Cooperative, not preemptive: an
already-in-flight model call finishes before a cancel takes effect.
"""

from __future__ import annotations

import argparse
import sys
import threading
from typing import Callable

from ftw.agent_loop import AgentLoop, DelegatedSuspension, Dispatch
from ftw.bus import Replier, Requester
from ftw.config import build_provider, load_config
from ftw.intercept import ConfirmShellCommands, PreCommitInterceptor
from ftw.outputs import OutputStore
from ftw.protocol import (
    AnswerEnvelope,
    AnyEnvelope,
    AskEnvelope,
    AskPayload,
    CallEnvelope,
    CancelEnvelope,
    ErrorEnvelope,
    ErrorPayload,
    ResultEnvelope,
    ResultPayload,
)
from ftw.providers import IModelProvider
from ftw.runtime import ipc_address
from ftw.skills.manifest import render_frame_text
from ftw.skills.registry import SkillNotFound, SkillStore
from ftw.workbench import ContextWorkbench

ProviderFactory = Callable[[str | None], IModelProvider]

# Guidance from ftw_plan.md §3.2 Mode A ("caller brief ≤ 300 tokens") — not
# enforced yet, just surfaced so a caller sees it during development.
RECOMMENDED_MAX_BRIEF_TOKENS = 300


class _PendingAsk:
    """What's stashed, keyed by resume_token, between an AskEnvelope going
    out and its matching AnswerEnvelope coming back."""

    __slots__ = ("loop", "suspension", "original_call", "span_id")

    def __init__(self, loop: AgentLoop, suspension: DelegatedSuspension, original_call: CallEnvelope, span_id: str):
        self.loop = loop
        self.suspension = suspension
        self.original_call = original_call
        self.span_id = span_id


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

        self._lock = threading.Lock()
        self._pending: dict[str, _PendingAsk] = {}
        self._cancel_flags: dict[str, threading.Event] = {}

    def handle(self, envelope: AnyEnvelope) -> ResultEnvelope | ErrorEnvelope | AskEnvelope:
        if isinstance(envelope, AnswerEnvelope):
            return self._handle_answer(envelope)
        return self._handle_call(envelope)

    def handle_cancel(self, envelope: AnyEnvelope) -> ResultEnvelope | ErrorEnvelope:
        """The control-channel handler (ftw_plan.md §4: "a busy call
        channel can't block a CANCEL"), served by a separate Replier."""
        if not isinstance(envelope, CancelEnvelope):
            return ErrorEnvelope(
                source=self._source_id,
                target=envelope.source,
                trace_id=envelope.trace_id,
                payload=ErrorPayload(code="unsupported_action", message=f"control channel only accepts CANCEL, got {envelope.type.value}"),
            )
        with self._lock:
            # setdefault, not get: a CANCEL can race the CALL it targets and
            # arrive first, in which case there's nothing to set yet — this
            # pre-creates the (already-set) flag so _handle_call finds it.
            flag = self._cancel_flags.setdefault(envelope.payload.target_span_id, threading.Event())
            flag.set()
        return ResultEnvelope(
            source=self._source_id,
            target=envelope.source,
            trace_id=envelope.trace_id,
            payload=ResultPayload(status="ok", summary="cancel requested"),
        )

    def _handle_call(self, call: CallEnvelope) -> ResultEnvelope | ErrorEnvelope | AskEnvelope:
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

        with self._lock:
            # A CANCEL can race a CALL for the same span_id and arrive first
            # (control and call channels are two separate sockets); reuse
            # whatever's already registered instead of clobbering it with a
            # fresh, unset Event and silently losing that cancellation.
            cancel_flag = self._cancel_flags.setdefault(call.span_id, threading.Event())

        full_brief = self._render_brief(brief, inputs)
        outcome = loop.run_delegated(full_brief, cancel_flag=cancel_flag)
        return self._finish(loop, call, call.span_id, outcome)

    def _handle_answer(self, answer: AnswerEnvelope) -> ResultEnvelope | ErrorEnvelope | AskEnvelope:
        with self._lock:
            entry = self._pending.pop(answer.payload.resume_token, None)
        if entry is None:
            return ErrorEnvelope(
                source=self._source_id,
                target=answer.source,
                trace_id=answer.trace_id,
                payload=ErrorPayload(
                    code="unknown_resume_token", message=f"no pending ask for resume_token {answer.payload.resume_token!r}"
                ),
            )
        outcome = entry.loop.resume_delegated(entry.suspension, answer.payload.value)
        return self._finish(entry.loop, entry.original_call, entry.span_id, outcome)

    def _finish(
        self, loop: AgentLoop, original_call: CallEnvelope, span_id: str, outcome: ResultPayload | DelegatedSuspension
    ) -> ResultEnvelope | AskEnvelope:
        if isinstance(outcome, DelegatedSuspension):
            with self._lock:
                self._pending[outcome.ask.payload.resume_token] = _PendingAsk(loop, outcome, original_call, span_id)
            return AskEnvelope(
                source=original_call.target,
                target=original_call.source,
                trace_id=original_call.trace_id,
                payload=AskPayload(question=outcome.ask.payload.question, resume_token=outcome.ask.payload.resume_token),
            )

        with self._lock:
            self._cancel_flags.pop(span_id, None)
        return ResultEnvelope(
            source=original_call.target,
            target=original_call.source,
            parent_span_id=original_call.span_id,
            trace_id=original_call.trace_id,
            payload=outcome,
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


def control_address(address: str) -> str:
    return f"{address}.ctl"


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
    """Blocks, serving CALLs (and, on a second Replier, CANCELs) until
    ``stop_event`` is set."""
    config = load_config(config_path)
    shell_requester = Requester(shell_worker_address)
    worker = SkillRunnerWorker(
        SkillStore(skills_dir),
        provider_factory=lambda tier: build_provider(config, tier or default_tier),
        output_store=OutputStore(output_root),
        dispatch=shell_requester.call,
        # Same default policy as the interactive REPL (repl/cli.py) — safe
        # to enable now that ASK passthrough exists to relay a delegated
        # run's confirmation prompt back to a real human instead of it
        # just blocking forever with no one there to answer.
        interceptor=PreCommitInterceptor([ConfirmShellCommands()]),
    )
    stop_event = stop_event or threading.Event()
    try:
        with Replier(address, num_workers=num_workers) as rep, Replier(control_address(address), num_workers=1) as ctl_rep:
            ctl_thread = threading.Thread(target=ctl_rep.serve_forever, args=(worker.handle_cancel, stop_event), daemon=True)
            ctl_thread.start()
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
    print(f"ftw skill runner listening on {args.address} (control: {control_address(args.address)})", file=sys.stderr)
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
