"""The frame tree: multiple and nested mounted skills over one Context
Workbench (ftw_plan.md §3.2 "Frames").

A mount is a stack frame, not a flat flag. Turns and pins made while a
skill is mounted are tagged with its frame id (workbench.py); unmounting
evicts exactly that frame's subtree — nothing more, nothing less — and
replaces it with one distilled milestone. The core, explicitly-tested
invariant: ``workbench_after_unmount == workbench_before_mount +
milestone``.

Nesting rule: a mount always becomes a child of whichever frame currently
has focus (``None`` = the root session). The newly mounted frame then takes
focus itself — "the most recently mounted" frame, per the plan. This is
uniform for both the model's self-mounting and the user's manual /mount:
the model naturally nests a helper skill under whatever it's focused on,
and a user who wants a fresh top-level mount instead just runs `/focus`
(with no argument) first to return focus to the root.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from ftw.providers import ChatMessage, ChatRole, IModelProvider
from ftw.skills.manifest import SkillManifest, render_frame_text
from ftw.skills.registry import SkillStore
from ftw.tokens import count_tokens
from ftw.workbench import ContextWorkbench

Owner = Literal["model", "user"]
Summarizer = Callable[[SkillManifest, str], Awaitable["str | None"]]


class FrameError(Exception):
    pass


class SkillAlreadyMounted(FrameError):
    pass


class FrameNotFound(FrameError):
    pass


class FramePinned(FrameError):
    pass


class FrameBudgetExceeded(FrameError):
    pass


@dataclass
class Frame:
    id: str
    skill_name: str
    manifest: SkillManifest
    body: str
    owner: Owner
    pinned: bool
    parent_id: str | None
    children_ids: list[str] = field(default_factory=list)


def _render_transcript(turns: list[list[ChatMessage]]) -> str:
    lines: list[str] = []
    for turn in turns:
        for msg in turn:
            if msg.role == ChatRole.USER:
                lines.append(f"user: {msg.content}")
            elif msg.role == ChatRole.ASSISTANT:
                if msg.tool_calls:
                    names = ", ".join(tc.name for tc in msg.tool_calls)
                    lines.append(f"assistant: (called {names})")
                if msg.content:
                    lines.append(f"assistant: {msg.content}")
            elif msg.role == ChatRole.TOOL:
                lines.append(f"tool[{msg.name}]: {msg.content}")
    return "\n".join(lines)


def _deterministic_summary(manifest: SkillManifest, turns: list[list[ChatMessage]]) -> str:
    """The fallback used when there's no summarizer, or it fails, or it's
    over budget — always available, never itself a point of failure."""
    tool_names: list[str] = []
    for turn in turns:
        for msg in turn:
            if msg.role == ChatRole.ASSISTANT:
                tool_names.extend(tc.name for tc in msg.tool_calls)
    used = list(dict.fromkeys(tool_names))  # de-dup, keep first-seen order
    if used:
        return f"{manifest.name}: {len(turns)} turn(s); used {', '.join(used)}."
    return f"{manifest.name}: {len(turns)} turn(s), no tool calls."


def make_llm_summarizer(provider: IModelProvider) -> Summarizer:
    """Builds a Summarizer that asks the model to distill a frame's work
    into at most 3 lines. Any failure here (network, malformed reply) just
    returns None — FrameTree's own deterministic fallback still applies,
    so an unmount is never blocked by a flaky summarization call."""

    async def summarize(manifest: SkillManifest, transcript: str) -> str | None:
        prompt = [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content=(
                    "Summarize the following completed work into at most 3 short, concrete "
                    "lines: what was done, what was found, what changed. No preamble."
                ),
            ),
            ChatMessage(role=ChatRole.USER, content=f"Skill: {manifest.name}\n\n{transcript or '(no activity)'}"),
        ]
        try:
            response = await provider.complete(prompt)
        except Exception:  # noqa: BLE001 - best-effort summary; caller falls back to _deterministic_summary
            return None
        text = (response.message.content or "").strip()
        return text or None

    return summarize


class FrameTree:
    def __init__(
        self,
        workbench: ContextWorkbench,
        skill_store: SkillStore,
        *,
        summarizer: Summarizer | None = None,
    ):
        self._workbench = workbench
        self._skills = skill_store
        self._summarizer = summarizer
        self._frames: dict[str, Frame] = {}
        self._focus_id: str | None = None

    # -- mounting -----------------------------------------------------------

    async def mount(self, skill_name: str, *, owner: Owner = "model", pinned: bool = False) -> Frame:
        if self._find_by_skill(skill_name) is not None:
            raise SkillAlreadyMounted(f"{skill_name!r} is already mounted")

        manifest, body = self._skills.get(skill_name)  # raises SkillNotFound

        candidate_tokens = count_tokens(render_frame_text(manifest, body))
        current_tokens = sum(
            count_tokens(render_frame_text(f.manifest, f.body)) for f in self._frames.values()
        )
        budget = self._workbench.mounted_skill_budget
        if current_tokens + candidate_tokens > budget:
            mounted = ", ".join(f.skill_name for f in self._frames.values()) or "(none)"
            raise FrameBudgetExceeded(
                f"mounting {skill_name!r} would need {current_tokens + candidate_tokens} tokens, over the "
                f"{budget}-token Mounted Skill budget. Currently mounted: {mounted}"
            )

        frame = Frame(
            id=uuid4().hex,
            skill_name=skill_name,
            manifest=manifest,
            body=body,
            owner=owner,
            pinned=pinned,
            parent_id=self._focus_id,
        )
        self._frames[frame.id] = frame
        if frame.parent_id is not None:
            self._frames[frame.parent_id].children_ids.append(frame.id)
        self._focus_id = frame.id
        self._sync_mounted_text()
        return frame

    # -- unmounting -----------------------------------------------------------

    async def unmount(self, skill_name: str, *, by: Owner = "model") -> str:
        frame = self._require_by_skill(skill_name)
        if by == "model":
            pinned = self._find_pinned_in_subtree(frame)
            if pinned is not None:
                raise FramePinned(f"{pinned.skill_name!r} is pinned; only the user can unmount it")

        milestone = await self._unmount_subtree(frame)
        self._workbench.add_milestone(milestone)
        self._sync_mounted_text()
        return milestone

    async def unmount_focused(self, *, by: Owner = "model") -> str:
        if self._focus_id is None:
            raise FrameNotFound("nothing is mounted/focused")
        return await self.unmount(self._frames[self._focus_id].skill_name, by=by)

    async def _unmount_subtree(self, frame: Frame) -> str:
        child_summaries = [await self._unmount_subtree(self._frames[cid]) for cid in list(frame.children_ids)]

        evicted_turns = self._workbench.evict_frame(frame.id)
        self._workbench.evict_frame_pins(frame.id)
        own_summary = await self._summarize(frame.manifest, evicted_turns)

        milestone = own_summary
        if child_summaries:
            milestone += " " + " ".join(f"[{s}]" for s in child_summaries)

        if frame.parent_id is not None:
            self._frames[frame.parent_id].children_ids.remove(frame.id)
        if self._focus_id == frame.id:
            self._focus_id = frame.parent_id
        del self._frames[frame.id]

        return milestone

    async def _summarize(self, manifest: SkillManifest, turns: list[list[ChatMessage]]) -> str:
        if self._summarizer is not None:
            try:
                result = await self._summarizer(manifest, _render_transcript(turns))
            except Exception:  # noqa: BLE001 - best-effort summary; falls through to _deterministic_summary below
                result = None
            if result:
                return result
        return _deterministic_summary(manifest, turns)

    def _find_pinned_in_subtree(self, frame: Frame) -> Frame | None:
        if frame.pinned:
            return frame
        for child_id in frame.children_ids:
            found = self._find_pinned_in_subtree(self._frames[child_id])
            if found is not None:
                return found
        return None

    # -- focus ------------------------------------------------------------

    def focus(self, skill_name: str | None) -> None:
        if skill_name is None:
            self._focus_id = None
            return
        self._focus_id = self._require_by_skill(skill_name).id

    @property
    def focused_skill_name(self) -> str | None:
        return self._frames[self._focus_id].skill_name if self._focus_id is not None else None

    @property
    def focused_frame_id(self) -> str | None:
        """The frame id turns/pins should be tagged with right now — what
        ``agent_loop.py`` passes to ``workbench.add_turn``/``pin``."""
        return self._focus_id

    # -- lookup / search ------------------------------------------------------

    def find(self, query: str, *, top_k: int = 5) -> list[tuple[str, str]]:
        return self._skills.find(query, top_k=top_k)

    def mounted_frames(self) -> list[Frame]:
        """Every currently mounted frame, in mount order — a parent always
        precedes its children, since a frame can only ever nest under one
        that's already mounted. sessions.py uses this to serialize the
        current mount structure; nothing else needs frames in bulk like
        this today."""
        return list(self._frames.values())

    def clear(self) -> None:
        """Hard-resets every mounted frame — no summarizer call, no
        milestone, no pinned-frame protection. Unlike unmount(), which
        always distills a frame's work before removing it, this is for
        when the frame tree is about to be replaced wholesale (sessions.py's
        load()): the frames being cleared here aren't finishing, they're
        being discarded in favor of whatever's being restored next, so
        there's nothing here worth summarizing."""
        self._frames = {}
        self._focus_id = None
        self._sync_mounted_text()

    def _find_by_skill(self, skill_name: str) -> Frame | None:
        return next((f for f in self._frames.values() if f.skill_name == skill_name), None)

    def _require_by_skill(self, skill_name: str) -> Frame:
        frame = self._find_by_skill(skill_name)
        if frame is None:
            raise FrameNotFound(f"no mounted skill named {skill_name!r}")
        return frame

    # -- reporting ------------------------------------------------------

    def render_tree(self) -> str:
        roots = [f for f in self._frames.values() if f.parent_id is None]
        if not roots:
            return "session (root) — nothing mounted"

        lines = ["session (root)"]

        def walk(frame: Frame, prefix: str, is_last: bool) -> None:
            connector = "└── " if is_last else "├── "
            markers = ("*" if frame.id == self._focus_id else "") + ("[pinned]" if frame.pinned else "")
            suffix = f"  {markers}" if markers else ""
            lines.append(f"{prefix}{connector}{frame.skill_name}{suffix}")
            child_prefix = prefix + ("    " if is_last else "│   ")
            children = [self._frames[cid] for cid in frame.children_ids]
            for i, child in enumerate(children):
                walk(child, child_prefix, i == len(children) - 1)

        for i, root in enumerate(roots):
            walk(root, "", i == len(roots) - 1)
        return "\n".join(lines)

    # -- sync ------------------------------------------------------

    def _sync_mounted_text(self) -> None:
        parts = [render_frame_text(f.manifest, f.body) for f in self._frames.values()]
        self._workbench.set_mounted_skill_text("\n\n".join(parts))
