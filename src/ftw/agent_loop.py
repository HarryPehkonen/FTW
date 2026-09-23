"""The agent loop: model -> interceptor -> bus -> result cycle
(ftw_plan.md §3.4).

This is the one place that cycle is implemented. The REPL drives it for
interactive turns; the delegated skill runner (Phase 3) will drive the same
class with a fresh, isolated workbench instead of the session's shared one.

Three kinds of tool call a model can make, handled differently on purpose:

* **Local tools** (``pin``, ``unpin``, ``read_output``, ``grep_output``) —
  pure workbench/output-store reads and writes, no host side effects.
  Never pass through the interceptor; there's nothing to confirm.
* **Intercepted calls** — go through
  :class:`~ftw.intercept.PreCommitInterceptor` first. ``ALLOW`` executes
  immediately; ``ASK`` calls the injected ``confirm`` callback and only
  executes on a yes; ``BLOCK`` never executes and never asks. Two flavors:
  - **Bus tools** (``run_command``, ...) dispatched as a CALL over the bus.
  - **Frame tools** (``find_skill``, ``mount_skill``, ``unmount_skill`` —
    ftw_plan.md §3.2 "Self-Mounting") execute locally against the shared
    :class:`~ftw.frames.FrameTree` instead of over the bus, since mounting
    is an in-process mutation of this session's own workbench, not a call
    to an isolated worker. Only present when a ``frame_tree`` is given.
* **Unknown tool names** — reported back to the model as an error, without
  touching the interceptor or the bus at all.

The default ``confirm`` denies everything: with no REPL/human wired up to
answer an ASK, failing closed is the only safe default.
"""

from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from ftw.frames import FrameBudgetExceeded, FrameNotFound, FramePinned, FrameTree, SkillAlreadyMounted
from ftw.intercept import InterceptDecision, PreCommitInterceptor
from ftw.outputs import OutputNotFound, OutputStore
from ftw.protocol import (
    AnyEnvelope,
    CallEnvelope,
    CallPayload,
    ErrorEnvelope,
    EventEnvelope,
    EventPayload,
    ResultEnvelope,
)
from ftw.providers import ChatMessage, ChatRole, IModelProvider, ToolCall, ToolSpec
from ftw.skills.registry import SkillNotFound
from ftw.workbench import ContextWorkbench, WorkbenchBudgetExceeded

Dispatch = Callable[[CallEnvelope], AnyEnvelope]
Confirm = Callable[[CallEnvelope], bool]
EventSink = Callable[[EventEnvelope], None]
LocalToolHandler = Callable[[dict[str, Any]], str]


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
        confirm: Confirm | None = None,
        frame_tree: FrameTree | None = None,
        extra_tool_specs: list[ToolSpec] | None = None,
        tool_targets: dict[str, str] | None = None,
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
        self._confirm: Confirm = confirm or (lambda call: False)  # fail closed
        self._tool_specs = list(BUILTIN_TOOL_SPECS) + list(extra_tool_specs or [])
        self._tool_targets = {**DEFAULT_TOOL_TARGETS, **(tool_targets or {})}
        self._max_steps = max_steps
        self._source_id = source_id
        self._on_event = on_event

        self._local_tools: dict[str, LocalToolHandler] = {
            "pin": self._handle_pin,
            "unpin": self._handle_unpin,
            "read_output": self._handle_read_output,
            "grep_output": self._handle_grep_output,
        }

        self._intercepted_local_tools: dict[str, LocalToolHandler] = {}
        if frame_tree is not None:
            self._tool_specs += FRAME_TOOL_SPECS
            self._intercepted_local_tools = {
                "find_skill": self._handle_find_skill,
                "mount_skill": self._handle_mount_skill,
                "unmount_skill": self._handle_unmount_skill,
            }

    def run_turn(self, user_input: str) -> str:
        turn: list[ChatMessage] = [ChatMessage(role=ChatRole.USER, content=user_input)]

        for step in range(self._max_steps):
            self._emit("event.model.request", {"step": step, "message_count": len(turn)})
            response = self.provider.complete(self.workbench.render_prompt(turn), tools=self._tool_specs)
            self._emit(
                "event.model.response",
                {"step": step, "has_tool_calls": bool(response.message.tool_calls), "tool_call_count": len(response.message.tool_calls)},
            )
            turn.append(response.message)

            if not response.message.tool_calls:
                self.workbench.add_turn(turn, frame_id=self._current_frame_id())
                return response.message.content or ""

            for tool_call in response.message.tool_calls:
                content = self._run_tool_call(tool_call)
                turn.append(
                    ChatMessage(role=ChatRole.TOOL, tool_call_id=tool_call.id, name=tool_call.name, content=content)
                )

        final = f"[step budget exceeded after {self._max_steps} steps without a final answer]"
        turn.append(ChatMessage(role=ChatRole.ASSISTANT, content=final))
        self.workbench.add_turn(turn, frame_id=self._current_frame_id())
        return final

    def _current_frame_id(self) -> str | None:
        """Which frame a turn or pin committed *right now* belongs to — the
        one with focus, or none if there's no frame tree or nothing
        mounted. Turns and pins are tagged when they're committed, not
        retroactively, so a mount/unmount that happens earlier in the same
        turn is what determines this, not what's focused later."""
        return self.frame_tree.focused_frame_id if self.frame_tree is not None else None

    # -- tool dispatch --------------------------------------------------

    def _run_tool_call(self, tool_call: ToolCall) -> str:
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

        if outcome.decision == InterceptDecision.ASK and not self._confirm(call):
            self._emit("event.tool.result", {"action": tool_call.name, "status": "declined"})
            return f"declined by user: {outcome.reason}"

        if executor is not None:
            content = executor(tool_call.arguments)
            self._emit("event.tool.result", {"action": tool_call.name, "status": "ok"})
            return content

        reply = self._dispatch(call)
        self._emit("event.tool.result", {"action": tool_call.name, "status": self._reply_status(reply)})
        return self._reply_to_content(reply)

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
        results = self.frame_tree.find(query, top_k=args.get("top_k", 5))
        return ", ".join(results) if results else "(no matching skills)"

    def _handle_mount_skill(self, args: dict[str, Any]) -> str:
        name = args.get("name")
        if not name:
            return "error: mount_skill requires 'name'"
        try:
            frame = self.frame_tree.mount(name, owner="model")
        except (SkillNotFound, SkillAlreadyMounted, FrameBudgetExceeded) as exc:
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
