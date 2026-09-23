"""The agent loop: model -> interceptor -> bus -> result cycle
(ftw_plan.md §3.4).

This is the one place that cycle is implemented, as a single shared
generator (``_step_loop``) driven differently by interactive and
delegated modes:

* ``run_turn`` (interactive) drives it synchronously — every question the
  generator ``yield``s is answered immediately via ``ask_answerer``, in
  the same call, and the final ``ResultPayload.summary`` becomes the chat
  reply.
* ``run_delegated``/``resume_delegated`` (Mode A, ftw_plan.md §3.2, §8
  Phase 3) drive it by *suspending*: the first question yielded is handed
  back to the caller as a :class:`DelegatedSuspension` instead of being
  answered, and the paused generator itself — Python's own machinery, not
  a hand-rolled state machine — holds everything needed to resume later,
  including from a completely different call, in a different process's
  request handler (``skills/runner.py`` stashes it by resume_token). No
  local human is needed to answer a delegated run's own interceptor ASK;
  it surfaces to whoever's driving the run exactly like a worker's own
  ASK does (see below) — the same mechanism serves both, so nothing
  extra was needed to support it.

Three kinds of tool call a model can make, handled differently on purpose:

* **Local tools** (``pin``, ``unpin``, ``read_output``, ``grep_output``) —
  pure workbench/output-store reads and writes, no host side effects.
  Never pass through the interceptor; there's nothing to confirm.
  ``submit_result`` (delegated mode only) is handled inline in
  ``_step_loop`` rather than as a local tool, since it ends the run.
* **Intercepted calls** — go through
  :class:`~ftw.intercept.PreCommitInterceptor` first. ``ALLOW`` executes
  immediately; ``ASK`` yields an ``AskEnvelope`` (carrying the actual
  action and arguments, not just the interceptor's reason) and only
  executes if the answer is exactly ``True``; ``BLOCK`` never executes
  and never asks. Two flavors:
  - **Bus tools** (``run_command``, ``delegate_skill``) dispatched as a
    CALL over the bus, with a deadline comfortably longer than whatever
    they wrap (see ``DEFAULT_CALL_DEADLINES_MS``) — a ``DeadlineExceeded``
    or a Ctrl-C ``KeyboardInterrupt`` during dispatch is reported back as
    a tool result, not left to crash the turn. If the worker itself
    replies ``ASK`` (not the interceptor — the worker mid-computation
    needing an answer, ftw_plan.md §4 "Mid-call interaction"), that's
    yielded too and redispatched once answered, looping until a
    ``RESULT``/``ERROR`` or a round cap.
  - **Frame tools** (``find_skill``, ``mount_skill``, ``unmount_skill`` —
    ftw_plan.md §3.2 "Self-Mounting") execute locally against the shared
    :class:`~ftw.frames.FrameTree` instead of over the bus, since mounting
    is an in-process mutation of this session's own workbench, not a call
    to an isolated worker. Only present when a ``frame_tree`` is given.
* **Unknown tool names** — reported back to the model as an error, without
  touching the interceptor or the bus at all.

The default ``ask_answerer`` declines everything: with no REPL/human wired
up to answer, failing closed is the only safe default. Only an answer of
exactly ``True`` allows an intercepted call through — a free-text answer
("nope", "cancel", any non-empty string) declines it rather than being
coerced to a truthy bool, the same way ``False``/empty does.

Every message is tagged with whatever frame has focus at the moment it's
actually produced (ftw_plan.md §3.2), not retroactively when the turn
commits: the assistant's own decision to mount/unmount is tagged with
focus as of before that decision takes effect, and each tool result is
tagged with focus as of just after it ran, since a mount_skill/
unmount_skill call can itself change what's focused mid-batch. This is
what makes the mount/unmount invariant hold even when a model mounts,
works, and unmounts all within a single reply — its natural self-mount
pattern, not just the two-separate-turns case.

**Cooperative cancellation** (``cancel_flag``): checked between steps of
a delegated run, since a blocked ``provider.complete()`` call can't be
preempted. Set by ``skills/runner.py`` in response to a ``CANCEL`` on its
control channel; captured once, in the generator's own closure, when the
run starts — a resume doesn't need it re-supplied.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Generator
from uuid import uuid4

from ftw.bus import DeadlineExceeded
from ftw.frames import FrameBudgetExceeded, FrameNotFound, FramePinned, FrameTree, SkillAlreadyMounted
from ftw.intercept import InterceptDecision, PreCommitInterceptor
from ftw.outputs import OutputNotFound, OutputStore
from ftw.protocol import (
    AnswerEnvelope,
    AnswerPayload,
    AnyEnvelope,
    AskEnvelope,
    AskPayload,
    CallEnvelope,
    CallPayload,
    CancelEnvelope,
    CancelPayload,
    ErrorEnvelope,
    EventEnvelope,
    EventPayload,
    ResultEnvelope,
    ResultPayload,
)
from ftw.providers import ChatMessage, ChatRole, IModelProvider, ProviderError, ToolCall, ToolSpec
from ftw.skills.manifest import SkillParseError
from ftw.skills.registry import SkillNotFound
from ftw.workbench import ContextWorkbench, WorkbenchBudgetExceeded

Dispatch = Callable[[AnyEnvelope], AnyEnvelope]  # a CallEnvelope, or an AnswerEnvelope relaying a worker's ASK
AskAnswerer = Callable[[AskEnvelope], Any]
EventSink = Callable[[EventEnvelope], None]
LocalToolHandler = Callable[[dict[str, Any]], str]
CancelFlag = Any  # duck-typed: anything with a no-arg is_set() -> bool, e.g. threading.Event
StepGenerator = Generator[AskEnvelope, Any, ResultPayload]
ToolCallGenerator = Generator[AskEnvelope, Any, str]

MAX_WORKER_ASK_ROUNDS = 10

# Every CALL gets a deadline comfortably longer than whatever it wraps, so
# a slow-but-legitimate command doesn't get mistaken for a hung one.
# run_command: the shell worker's own default command timeout is 60s
# (tools/shell.py). delegate_skill: a delegated run may need several
# model round-trips plus its own tool calls, so it gets much more room.
DEFAULT_CALL_DEADLINES_MS: dict[str, int] = {
    "run_command": 65_000,
    "delegate_skill": 300_000,
}
FALLBACK_CALL_DEADLINE_MS = 30_000


@dataclass
class DelegatedSuspension:
    """Returned by run_delegated/resume_delegated in place of a
    ResultPayload when the run needs external input before it can
    continue. ``ask`` is the only part meant to be read by the caller —
    the paused generator itself carries the rest of the state."""

    ask: AskEnvelope
    _generator: StepGenerator
    _cancel_flag: CancelFlag | None = None


SUBMIT_RESULT_TOOL_SPEC = ToolSpec(
    name="submit_result",
    description=(
        "Finish this delegated task and report the outcome. Call this exactly once, "
        "when the task is complete or has failed and cannot proceed further."
    ),
    parameters={
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "error"]},
            "summary": {"type": "string"},
            "outputs": {"type": "object"},
            "evidence": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["status", "summary"],
    },
)


BUILTIN_TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="run_command",
        description="Run a shell command and return its exit code and output.",
        parameters={
            "type": "object",
            "properties": {
                "argv": {"type": "array", "items": {"type": "string"}},
                "cwd": {"type": "string"},
            },
            "required": ["argv"],
        },
    ),
    ToolSpec(
        name="pin",
        description="Save a key/value fact to the scratchpad so it survives turn eviction.",
        parameters={
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
            "required": ["key", "value"],
        },
    ),
    ToolSpec(
        name="unpin",
        description="Remove a previously pinned key from the scratchpad.",
        parameters={"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
    ),
    ToolSpec(
        name="read_output",
        description="Read more of a tool output referenced by an output_id handle, by line range.",
        parameters={
            "type": "object",
            "properties": {
                "output_id": {"type": "string"},
                "start": {"type": "integer"},
                "end": {"type": "integer"},
            },
            "required": ["output_id"],
        },
    ),
    ToolSpec(
        name="grep_output",
        description="Search a tool output referenced by an output_id handle for lines matching a regex.",
        parameters={
            "type": "object",
            "properties": {
                "output_id": {"type": "string"},
                "pattern": {"type": "string"},
                "max_matches": {"type": "integer"},
            },
            "required": ["output_id", "pattern"],
        },
    ),
]

DELEGATE_SKILL_TOOL_SPEC = ToolSpec(
    name="delegate_skill",
    description=(
        "Run a skill as an isolated sub-task with its own fresh context. Only its structured "
        "Result (a short summary, not its internal steps) comes back into this conversation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "brief": {"type": "string"},
            "inputs": {"type": "object"},
        },
        "required": ["name", "brief"],
    },
)

FRAME_TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="find_skill",
        description="Search the skill catalog by name/description; returns matching skill names, best first.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
    ),
    ToolSpec(
        name="mount_skill",
        description="Mount a skill so its instructions guide the next turns. Nests under the currently focused skill, if any.",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    ),
    ToolSpec(
        name="unmount_skill",
        description="Unmount a skill (or, with no name, the currently focused one), distilling its work into a milestone.",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": []},
    ),
]

DEFAULT_TOOL_TARGETS: dict[str, str] = {"run_command": "worker.tool.shell"}

# The CallEnvelope.target for frame tools — there's no bus worker on the
# other end (mounting mutates this session's own workbench in-process),
# but the interceptor still needs a well-formed CallEnvelope to evaluate.
FRAME_TOOL_TARGET = "self.frames"


class AgentLoop:
    def __init__(
        self,
        *,
        workbench: ContextWorkbench,
        provider: IModelProvider,
        output_store: OutputStore,
        dispatch: Dispatch,
        interceptor: PreCommitInterceptor | None = None,
        ask_answerer: AskAnswerer | None = None,
        control_dispatch: Dispatch | None = None,
        frame_tree: FrameTree | None = None,
        enable_delegated_completion: bool = False,
        skill_runner_target: str | None = None,
        extra_tool_specs: list[ToolSpec] | None = None,
        tool_targets: dict[str, str] | None = None,
        call_deadlines_ms: dict[str, int] | None = None,
        max_steps: int = 15,
        source_id: str = "repl.master",
        trace_id: str | None = None,
        on_event: EventSink | None = None,
    ):
        self.workbench = workbench
        self.provider = provider
        self.frame_tree = frame_tree
        self.trace_id = trace_id or str(uuid4())

        self._outputs = output_store
        self._dispatch = dispatch
        self._interceptor = interceptor or PreCommitInterceptor([])
        self._ask_answerer: AskAnswerer = ask_answerer or (lambda ask: False)  # fail closed
        self._control_dispatch = control_dispatch
        self._tool_specs = list(BUILTIN_TOOL_SPECS) + list(extra_tool_specs or [])
        self._tool_targets = {**DEFAULT_TOOL_TARGETS, **(tool_targets or {})}
        self._call_deadlines_ms = {**DEFAULT_CALL_DEADLINES_MS, **(call_deadlines_ms or {})}
        self._max_steps = max_steps
        self._source_id = source_id
        self._on_event = on_event
        self._delegated_enabled = enable_delegated_completion

        self._local_tools: dict[str, LocalToolHandler] = {
            "pin": self._handle_pin,
            "unpin": self._handle_unpin,
            "read_output": self._handle_read_output,
            "grep_output": self._handle_grep_output,
        }
        if enable_delegated_completion:
            self._tool_specs.append(SUBMIT_RESULT_TOOL_SPEC)

        if skill_runner_target is not None:
            self._tool_specs.append(DELEGATE_SKILL_TOOL_SPEC)
            self._tool_targets["delegate_skill"] = skill_runner_target

        self._intercepted_local_tools: dict[str, LocalToolHandler] = {}
        if frame_tree is not None:
            self._tool_specs += FRAME_TOOL_SPECS
            self._intercepted_local_tools = {
                "find_skill": self._handle_find_skill,
                "mount_skill": self._handle_mount_skill,
                "unmount_skill": self._handle_unmount_skill,
            }

    # -- public entry points --------------------------------------------

    def run_turn(self, user_input: str) -> str:
        gen = self._step_loop(ChatMessage(role=ChatRole.USER, content=user_input))
        answer: Any = None
        while True:
            try:
                ask = gen.send(answer)
            except StopIteration as stop:
                result: ResultPayload = stop.value
                return result.summary
            answer = self._ask_answerer(ask)

    def run_delegated(self, brief: str, *, cancel_flag: CancelFlag | None = None) -> ResultPayload | DelegatedSuspension:
        """Runs to a structured completion instead of a chat reply — the
        model must call submit_result to finish. Only meaningful on an
        AgentLoop built with enable_delegated_completion=True and, in
        practice, its own fresh workbench (ftw_plan.md §3.2 Mode A).

        Returns a DelegatedSuspension instead of a ResultPayload if the
        run needs an answer before it can continue — resume with
        resume_delegated once one arrives."""
        if not self._delegated_enabled:
            raise RuntimeError("this AgentLoop wasn't constructed with enable_delegated_completion=True")
        gen = self._step_loop(ChatMessage(role=ChatRole.USER, content=brief), cancel_flag=cancel_flag)
        return self._drive_delegated(gen, None, cancel_flag=cancel_flag)

    def resume_delegated(self, suspension: DelegatedSuspension, answer_value: Any) -> ResultPayload | DelegatedSuspension:
        """Continues a run exactly where it suspended — the paused
        generator itself is the entire resumable state, so this is just
        sending the answer into it."""
        if not self._delegated_enabled:
            raise RuntimeError("this AgentLoop wasn't constructed with enable_delegated_completion=True")
        return self._drive_delegated(suspension._generator, answer_value, cancel_flag=suspension._cancel_flag)

    def _drive_delegated(
        self, gen: StepGenerator, send_value: Any, *, cancel_flag: CancelFlag | None
    ) -> ResultPayload | DelegatedSuspension:
        try:
            ask = gen.send(send_value)
        except StopIteration as stop:
            return stop.value
        return DelegatedSuspension(ask=ask, _generator=gen, _cancel_flag=cancel_flag)

    # -- the shared step loop --------------------------------------------

    def _step_loop(self, initial_message: ChatMessage, *, cancel_flag: CancelFlag | None = None) -> StepGenerator:
        tagged: list[tuple[str | None, ChatMessage]] = [(self._current_frame_id(), initial_message)]
        plain: list[ChatMessage] = [initial_message]
        start = time.monotonic()
        total_tokens = 0

        for step in range(self._max_steps):
            if self._is_cancelled(cancel_flag):
                return self._commit(tagged, self._cancelled_result(start, total_tokens))

            self._emit("event.model.request", {"step": step, "message_count": len(plain)})
            try:
                response = self.provider.complete(self.workbench.render_prompt(plain), tools=self._tool_specs)
            except ProviderError as exc:
                return self._commit(
                    tagged, ResultPayload(status="error", summary=f"model error: {exc}", cost=self._cost(start, total_tokens))
                )
            total_tokens += self._usage_tokens(response.usage)
            self._emit(
                "event.model.response",
                {
                    "step": step,
                    "has_tool_calls": bool(response.message.tool_calls),
                    "tool_call_count": len(response.message.tool_calls),
                },
            )

            tagged.append((self._current_frame_id(), response.message))
            plain.append(response.message)

            if not response.message.tool_calls:
                return self._commit(
                    tagged,
                    ResultPayload(status="ok", summary=response.message.content or "", cost=self._cost(start, total_tokens)),
                )

            for tool_call in response.message.tool_calls:
                if tool_call.name == "submit_result" and self._delegated_enabled:
                    payload = self._build_submit_result(tool_call.arguments, start, total_tokens)
                    msg = ChatMessage(role=ChatRole.TOOL, tool_call_id=tool_call.id, name=tool_call.name, content="(submitted)")
                    tagged.append((self._current_frame_id(), msg))
                    return self._commit(tagged, payload)

                content = yield from self._run_tool_call(tool_call, cancel_flag=cancel_flag)
                msg = ChatMessage(role=ChatRole.TOOL, tool_call_id=tool_call.id, name=tool_call.name, content=content)
                tagged.append((self._current_frame_id(), msg))
                plain.append(msg)
                # Re-checked here, not just at the top of the step: a
                # suspended ask resumes *inside* _run_tool_call, well past
                # that check, so a cancel arriving while paused would
                # otherwise only be noticed one whole step too late.
                if self._is_cancelled(cancel_flag):
                    return self._commit(tagged, self._cancelled_result(start, total_tokens))

        return self._commit(
            tagged,
            ResultPayload(
                status="error",
                summary=f"step budget exceeded after {self._max_steps} steps",
                cost=self._cost(start, total_tokens),
            ),
        )

    def _commit(self, tagged: list[tuple[str | None, ChatMessage]], payload: ResultPayload) -> ResultPayload:
        self.workbench.add_tagged_turn(tagged)
        return payload

    def _build_submit_result(self, args: dict[str, Any], start: float, total_tokens: int) -> ResultPayload:
        return ResultPayload(
            status=args.get("status", "ok"),
            summary=args.get("summary", ""),
            outputs=args.get("outputs") or {},
            evidence=args.get("evidence") or [],
            cost=self._cost(start, total_tokens),
        )

    @staticmethod
    def _is_cancelled(cancel_flag: CancelFlag | None) -> bool:
        return cancel_flag is not None and cancel_flag.is_set()

    def _cancelled_result(self, start: float, total_tokens: int) -> ResultPayload:
        return ResultPayload(status="error", summary="cancelled by user", cost=self._cost(start, total_tokens))

    @staticmethod
    def _usage_tokens(usage: dict[str, int]) -> int:
        if "total_tokens" in usage:
            return usage["total_tokens"]
        return usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

    @staticmethod
    def _cost(start: float, total_tokens: int) -> dict[str, int]:
        cost = {"wall_time_ms": int((time.monotonic() - start) * 1000)}
        if total_tokens:
            cost["tokens"] = total_tokens
        return cost

    def _current_frame_id(self) -> str | None:
        """Which frame a message produced *right now* belongs to — the one
        with focus, or none if there's no frame tree or nothing mounted."""
        return self.frame_tree.focused_frame_id if self.frame_tree is not None else None

    # -- tool dispatch --------------------------------------------------

    def _run_tool_call(self, tool_call: ToolCall, *, cancel_flag: CancelFlag | None = None) -> ToolCallGenerator:
        if tool_call.name in self._local_tools:
            return self._local_tools[tool_call.name](tool_call.arguments)

        executor = self._intercepted_local_tools.get(tool_call.name)
        target = FRAME_TOOL_TARGET if executor is not None else self._tool_targets.get(tool_call.name)
        if executor is None and target is None:
            return f"error: unknown tool {tool_call.name!r}"

        call = CallEnvelope(
            source=self._source_id,
            target=target,
            trace_id=self.trace_id,
            deadline_ms=self._call_deadlines_ms.get(tool_call.name, FALLBACK_CALL_DEADLINE_MS),
            payload=CallPayload(action=tool_call.name, args=tool_call.arguments),
        )

        outcome = self._interceptor.evaluate(call)
        self._emit(
            "event.intercept.decision",
            {"action": tool_call.name, "decision": outcome.decision.value, "reason": outcome.reason},
        )

        if outcome.decision == InterceptDecision.BLOCK:
            self._emit("event.tool.result", {"action": tool_call.name, "status": "blocked"})
            return f"blocked: {outcome.reason}"

        if outcome.decision == InterceptDecision.ASK:
            resume_token = uuid4().hex
            answer = yield AskEnvelope(
                source=self._source_id,
                target=call.source,
                trace_id=self.trace_id,
                payload=AskPayload(
                    question=self._render_ask_question(tool_call, outcome.reason),
                    resume_token=resume_token,
                    expected={"action": tool_call.name, "args": tool_call.arguments},
                ),
            )
            if self._is_cancelled(cancel_flag):
                # A resumed ask lands here, well past _step_loop's own
                # per-step check — this is the one that actually matters
                # for "cancelled while suspended, waiting on an answer".
                return "cancelled by user"
            # Only an explicit True allows. Anything else — False, an
            # empty answer, or a free-text answer like "nope" or "cancel"
            # — declines, rather than being coerced through bool(), which
            # would treat any non-empty string as an approval.
            if answer is not True:
                self._emit("event.tool.result", {"action": tool_call.name, "status": "declined"})
                return f"declined by user: {outcome.reason}"

        if executor is not None:
            content = executor(tool_call.arguments)
            self._emit("event.tool.result", {"action": tool_call.name, "status": "ok"})
            return content

        try:
            reply = self._dispatch(call)
        except KeyboardInterrupt:
            return self._cancelled_tool_content(call, tool_call.name)
        except DeadlineExceeded as exc:
            return self._timed_out_tool_content(tool_call.name, exc)

        rounds = 0
        while isinstance(reply, AskEnvelope) and rounds < MAX_WORKER_ASK_ROUNDS:
            rounds += 1
            value = yield reply
            if self._is_cancelled(cancel_flag):
                return "cancelled by user"
            answer_env = AnswerEnvelope(
                source=self._source_id,
                target=reply.source,
                trace_id=self.trace_id,
                payload=AnswerPayload(resume_token=reply.payload.resume_token, value=value),
            )
            try:
                reply = self._dispatch(answer_env)
            except KeyboardInterrupt:
                return self._cancelled_tool_content(call, tool_call.name)
            except DeadlineExceeded as exc:
                return self._timed_out_tool_content(tool_call.name, exc)

        self._emit("event.tool.result", {"action": tool_call.name, "status": self._reply_status(reply)})
        return self._reply_to_content(reply)

    def _cancelled_tool_content(self, call: CallEnvelope, tool_name: str) -> str:
        self._send_cancel(call)
        self._emit("event.tool.result", {"action": tool_name, "status": "cancelled"})
        return f"cancelled by user (Ctrl-C) while running {tool_name!r}"

    def _timed_out_tool_content(self, tool_name: str, exc: DeadlineExceeded) -> str:
        self._emit("event.tool.result", {"action": tool_name, "status": "timeout"})
        return f"error: {tool_name} timed out: {exc}"

    @staticmethod
    def _render_ask_question(tool_call: ToolCall, reason: str) -> str:
        args_text = json.dumps(tool_call.arguments, separators=(",", ": "), sort_keys=True)
        base = reason or f"Allow {tool_call.name}?"
        return f"{base} — {tool_call.name}({args_text}) [y/N]"

    def _send_cancel(self, call: CallEnvelope) -> None:
        """Best-effort: tells the target worker to stop an in-flight call
        after a KeyboardInterrupt. Swallows any failure — the original
        KeyboardInterrupt has already been handled (the call is being
        treated as cancelled either way); a failed notification just means
        the worker keeps running to its own completion unaware.

        This branch itself is correct and tested (test_agent_loop.py's
        TestKeyboardInterruptDuringDispatch), but real Ctrl-C doesn't
        reliably reach it today — see bus.Requester.call()'s docstring for
        why pynng's blocking recv() doesn't hand control back to Python in
        time to raise one.

        ``control_dispatch`` is one fixed callable (see Dispatch), bound to
        whichever single worker's control channel matters most — in
        practice the delegated skill runner, since a delegated run is the
        one thing worth interrupting mid-flight and the only worker with a
        control channel today. A CANCEL for a *different* target (e.g.
        run_command, which the shell worker doesn't expose a control
        channel for at all) is a harmless no-op wherever it lands, not a
        real interruption — routing CANCEL per-target the way
        bus.DispatchRouter does for CALL is future work, once more than
        one worker actually supports it."""
        if self._control_dispatch is None:
            return
        try:
            self._control_dispatch(
                CancelEnvelope(
                    source=self._source_id,
                    target=call.target,
                    trace_id=self.trace_id,
                    payload=CancelPayload(target_span_id=call.span_id, reason="Ctrl-C"),
                )
            )
        except Exception:
            pass

    @staticmethod
    def _reply_status(reply: AnyEnvelope) -> str:
        if isinstance(reply, ResultEnvelope):
            return reply.payload.status
        if isinstance(reply, ErrorEnvelope):
            return "error"
        return reply.type.value.lower()

    @staticmethod
    def _reply_to_content(reply: AnyEnvelope) -> str:
        if isinstance(reply, ResultEnvelope):
            parts = [reply.payload.summary]
            excerpt = reply.payload.outputs.get("excerpt")
            if excerpt:
                parts.append(str(excerpt))
            output_id = reply.payload.outputs.get("output_id")
            if output_id:
                parts.append(f"(full output: output_id={output_id})")
            return "\n".join(p for p in parts if p)
        if isinstance(reply, ErrorEnvelope):
            return f"error [{reply.payload.code}]: {reply.payload.message}"
        return f"unexpected reply type: {reply.type.value}"

    # -- local tools ------------------------------------------------------

    def _handle_pin(self, args: dict[str, Any]) -> str:
        key, value = args.get("key"), args.get("value")
        if key is None or value is None:
            return "error: pin requires 'key' and 'value'"
        try:
            self.workbench.pin(key, value, frame_id=self._current_frame_id())
        except WorkbenchBudgetExceeded as exc:
            return f"error: {exc}"
        return f"pinned {key!r}"

    def _handle_unpin(self, args: dict[str, Any]) -> str:
        key = args.get("key")
        if key is None:
            return "error: unpin requires 'key'"
        try:
            self.workbench.unpin(key)
        except KeyError:
            return f"error: no such pin {key!r}"
        return f"unpinned {key!r}"

    def _handle_read_output(self, args: dict[str, Any]) -> str:
        output_id = args.get("output_id")
        if output_id is None:
            return "error: read_output requires 'output_id'"
        try:
            return self._outputs.read(self.trace_id, output_id, args.get("start", 0), args.get("end"))
        except OutputNotFound as exc:
            return f"error: {exc}"

    def _handle_grep_output(self, args: dict[str, Any]) -> str:
        output_id, pattern = args.get("output_id"), args.get("pattern")
        if output_id is None or pattern is None:
            return "error: grep_output requires 'output_id' and 'pattern'"
        try:
            matches = self._outputs.grep(self.trace_id, output_id, pattern, args.get("max_matches", 50))
        except OutputNotFound as exc:
            return f"error: {exc}"
        return "\n".join(matches) if matches else "(no matches)"

    # -- frame tools (find_skill / mount_skill / unmount_skill) -----------

    def _handle_find_skill(self, args: dict[str, Any]) -> str:
        query = args.get("query")
        if not query:
            return "error: find_skill requires 'query'"
        try:
            results = self.frame_tree.find(query, top_k=args.get("top_k", 5))
        except SkillParseError as exc:
            return f"error: {exc}"
        return ", ".join(results) if results else "(no matching skills)"

    def _handle_mount_skill(self, args: dict[str, Any]) -> str:
        name = args.get("name")
        if not name:
            return "error: mount_skill requires 'name'"
        try:
            frame = self.frame_tree.mount(name, owner="model")
        except (SkillNotFound, SkillAlreadyMounted, FrameBudgetExceeded, SkillParseError) as exc:
            return f"error: {exc}"
        return f"mounted {frame.skill_name!r}"

    def _handle_unmount_skill(self, args: dict[str, Any]) -> str:
        name = args.get("name")
        try:
            milestone = self.frame_tree.unmount(name, by="model") if name else self.frame_tree.unmount_focused(by="model")
        except (FrameNotFound, FramePinned) as exc:
            return f"error: {exc}"
        return f"unmounted; milestone: {milestone}"

    # -- observability ----------------------------------------------------

    def _emit(self, topic: str, data: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        self._on_event(
            EventEnvelope(
                source=self._source_id,
                target="events",
                trace_id=self.trace_id,
                payload=EventPayload(topic=topic, data=data),
            )
        )
