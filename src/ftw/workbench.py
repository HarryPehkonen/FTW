"""The Context Workbench: an L1 cache with explicit, budgeted zones
(ftw_plan.md §3.3).

Zone order — both in this module and in the rendered prompt — is
deliberately least-volatile-first, most-volatile-last:

    System Anchor -> User Memory -> Mounted Skill -> Milestones
    -> Turn Horizon -> Scratchpad

Mounting/unmounting a skill or adding a milestone only invalidates the
prompt from that zone onward; the scratchpad, which can change every turn,
sits last so it never invalidates anything ahead of it. This ordering is
what makes KV/prefix caching (local runtimes and provider-side prompt
caching alike) actually pay off across turns.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from ftw.providers import ChatMessage, ChatRole
from ftw.tokens import count_tokens

DEFAULT_SYSTEM_ANCHOR_BUDGET = 300
DEFAULT_USER_MEMORY_BUDGET = 500
DEFAULT_MOUNTED_SKILL_BUDGET = 3000
DEFAULT_MILESTONES_BUDGET = 150
DEFAULT_TURN_HORIZON_BUDGET = 4000
DEFAULT_SCRATCHPAD_BUDGET = 300
DEFAULT_MAX_TOTAL_TOKENS = 8000


class WorkbenchBudgetExceeded(Exception):
    """Raised when an explicit addition (a pin, a mount) would exceed its
    zone's budget. Callers never lose data silently — the caller decides
    whether to shrink the addition, evict something else, or surface the
    refusal to the model/user."""


def _message_tokens(message: ChatMessage) -> int:
    total = count_tokens(message.content or "")
    if message.tool_calls:
        total += count_tokens(json.dumps([tc.model_dump() for tc in message.tool_calls]))
    return total


def _turn_tokens(turn: list[ChatMessage]) -> int:
    return sum(_message_tokens(m) for m in turn)


@dataclass
class _TaggedMessage:
    frame_id: str | None
    message: ChatMessage


@dataclass
class _TurnRecord:
    tagged: list[_TaggedMessage]

    @property
    def messages(self) -> list[ChatMessage]:
        return [tm.message for tm in self.tagged]


@dataclass
class _PinRecord:
    frame_id: str | None
    value: str


@dataclass
class ZoneSnapshot:
    name: str
    tokens: int
    budget: int | None
    detail: str = ""


@dataclass
class WorkbenchSnapshot:
    zones: list[ZoneSnapshot]
    total_tokens: int
    max_total_tokens: int


class ContextWorkbench:
    def __init__(
        self,
        *,
        system_anchor: str = "",
        user_memory: str = "",
        max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
        system_anchor_budget: int = DEFAULT_SYSTEM_ANCHOR_BUDGET,
        user_memory_budget: int = DEFAULT_USER_MEMORY_BUDGET,
        mounted_skill_budget: int = DEFAULT_MOUNTED_SKILL_BUDGET,
        milestones_budget: int = DEFAULT_MILESTONES_BUDGET,
        turn_horizon_budget: int = DEFAULT_TURN_HORIZON_BUDGET,
        scratchpad_budget: int = DEFAULT_SCRATCHPAD_BUDGET,
        on_turn_evicted: Callable[[list[ChatMessage]], None] | None = None,
    ):
        self._system_anchor = system_anchor
        self._user_memory = user_memory
        self._max_total_tokens = max_total_tokens

        self._system_anchor_budget = system_anchor_budget
        self._user_memory_budget = user_memory_budget
        self._mounted_skill_budget = mounted_skill_budget
        self._milestones_budget = milestones_budget
        self._turn_horizon_budget = turn_horizon_budget
        self._scratchpad_budget = scratchpad_budget

        self._mounted_skill_text = ""
        self._milestones: list[str] = []
        self._turns: list[_TurnRecord] = []
        self._scratchpad: dict[str, _PinRecord] = {}
        self._on_turn_evicted = on_turn_evicted

    # -- System Anchor / User Memory: static, set at construction ---------

    @property
    def system_anchor_tokens(self) -> int:
        return count_tokens(self._system_anchor)

    @property
    def user_memory_tokens(self) -> int:
        return count_tokens(self._user_memory)

    # -- Mounted Skill: Phase 2's frame tree drives this; the hook exists
    #    now so the workbench's zone/ordering contract is fixed from Phase 1.

    def set_mounted_skill_text(self, text: str) -> None:
        self._mounted_skill_text = text

    @property
    def mounted_skill_tokens(self) -> int:
        return count_tokens(self._mounted_skill_text)

    @property
    def mounted_skill_budget(self) -> int:
        return self._mounted_skill_budget

    # -- Milestones: rolling, oldest evicted first over budget ------------

    def add_milestone(self, text: str) -> None:
        self._milestones.append(text)
        while self._milestones_tokens_of(self._milestones) > self._milestones_budget and len(self._milestones) > 1:
            self._milestones.pop(0)

    @staticmethod
    def _milestones_tokens_of(milestones: list[str]) -> int:
        return sum(count_tokens(m) for m in milestones)

    @property
    def milestones(self) -> list[str]:
        return list(self._milestones)

    @property
    def milestones_tokens(self) -> int:
        return self._milestones_tokens_of(self._milestones)

    # -- Turn Horizon: rolling FIFO, oldest turn evicted first ------------
    #
    # Every MESSAGE is tagged with the frame it belongs to (None = the base
    # session, untagged) — not the turn as a whole. A turn where a mount,
    # some work, and the matching unmount all happen in one reply (the
    # model's natural self-mount pattern) has messages that belong to
    # different frames; tagging the whole turn with whatever's focused at
    # commit time (i.e. after the unmount) meant the frame's own content
    # was never evicted at all. Budget-driven eviction below is
    # oldest-turn-first and frame-agnostic; evict_frame() is the other way
    # content leaves the horizon — an explicit, frame-scoped removal on
    # unmount (ftw_plan.md §3.2 "Eviction is by subtree") that now works at
    # message granularity, leaving the rest of a mixed turn in place.

    def add_tagged_turn(self, tagged_messages: list[tuple[str | None, ChatMessage]]) -> None:
        self._turns.append(_TurnRecord([_TaggedMessage(fid, m) for fid, m in tagged_messages]))
        while self.turn_horizon_tokens > self._turn_horizon_budget and len(self._turns) > 1:
            evicted = self._turns.pop(0)
            self._fire_turn_evicted(evicted.messages)

    def add_turn(self, messages: list[ChatMessage], frame_id: str | None = None) -> None:
        """Convenience wrapper for the common case where an entire turn
        genuinely belongs to one frame (or none)."""
        self.add_tagged_turn([(frame_id, m) for m in messages])

    def clear_turns(self) -> None:
        while self._turns:
            evicted = self._turns.pop(0)
            self._fire_turn_evicted(evicted.messages)

    def reset(self) -> None:
        """Clears turns, milestones, and the scratchpad in one go — used
        when a previously saved session is loaded (sessions.py), so the
        load starts from a clean slate instead of appending on top of
        whatever's already here."""
        self.clear_turns()
        self._milestones = []
        self._scratchpad = {}

    def evict_frame(self, frame_id: str) -> list[list[ChatMessage]]:
        """Removes every message tagged with ``frame_id``, wherever it
        appears — a turn that becomes empty is dropped entirely; one that
        still has other messages keeps them, in their original order.
        Returns the evicted messages, grouped by their original turn, for
        the unmounting frame to summarize."""
        evicted_by_turn: list[list[ChatMessage]] = []
        remaining: list[_TurnRecord] = []
        for record in self._turns:
            keep = [tm for tm in record.tagged if tm.frame_id != frame_id]
            evicted = [tm.message for tm in record.tagged if tm.frame_id == frame_id]
            if evicted:
                evicted_by_turn.append(evicted)
                self._fire_turn_evicted(evicted)
            if keep:
                remaining.append(_TurnRecord(keep))
        self._turns = remaining
        return evicted_by_turn

    def _fire_turn_evicted(self, messages: list[ChatMessage]) -> None:
        if self._on_turn_evicted is not None:
            self._on_turn_evicted(messages)

    @property
    def turns(self) -> list[list[ChatMessage]]:
        return [list(record.messages) for record in self._turns]

    @property
    def tagged_turns(self) -> list[list[tuple[str | None, ChatMessage]]]:
        """Like ``turns``, but keeps each message's frame_id — sessions.py
        needs this to translate the (process-lifetime, not stable across a
        restart) frame id into a portable skill_name label when saving."""
        return [[(tm.frame_id, tm.message) for tm in record.tagged] for record in self._turns]

    @property
    def turn_horizon_tokens(self) -> int:
        return sum(_turn_tokens(record.messages) for record in self._turns)

    # -- Scratchpad: explicit pins, refused (not silently dropped) over budget.
    #    Each pin is tagged with a frame the same way turns are.

    def pin(self, key: str, value: str, frame_id: str | None = None) -> None:
        candidate = dict(self._scratchpad)
        candidate[key] = _PinRecord(frame_id, value)
        if self._scratchpad_tokens_of(candidate) > self._scratchpad_budget:
            raise WorkbenchBudgetExceeded(
                f"pinning {key!r} would exceed the scratchpad budget ({self._scratchpad_budget} tokens)"
            )
        self._scratchpad = candidate

    def unpin(self, key: str) -> None:
        del self._scratchpad[key]  # raises KeyError for an unknown key, deliberately

    def evict_frame_pins(self, frame_id: str) -> dict[str, str]:
        """Removes and returns every pin tagged with ``frame_id``."""
        evicted = {k: r.value for k, r in self._scratchpad.items() if r.frame_id == frame_id}
        for key in evicted:
            del self._scratchpad[key]
        return evicted

    @staticmethod
    def _scratchpad_tokens_of(scratchpad: dict[str, _PinRecord]) -> int:
        return count_tokens("\n".join(f"{k}={r.value}" for k, r in scratchpad.items()))

    @property
    def scratchpad(self) -> dict[str, str]:
        return {k: r.value for k, r in self._scratchpad.items()}

    @property
    def tagged_scratchpad(self) -> dict[str, tuple[str | None, str]]:
        """Like ``scratchpad``, but keeps each pin's frame_id — see
        ``tagged_turns`` for why sessions.py needs this."""
        return {k: (r.frame_id, r.value) for k, r in self._scratchpad.items()}

    @property
    def scratchpad_tokens(self) -> int:
        return self._scratchpad_tokens_of(self._scratchpad)

    # -- Prompt rendering ---------------------------------------------------

    def render_prompt(self, extra_messages: list[ChatMessage] | None = None) -> list[ChatMessage]:
        """``extra_messages`` is the turn currently being built by the agent
        loop, not yet committed via ``add_turn``. It's inserted after the
        committed Turn Horizon but still before the Scratchpad message, so
        the scratchpad — the part most likely to change next turn — stays
        the very last thing in the prompt."""
        sections = [
            self._system_anchor,
            self._user_memory,
            self._mounted_skill_text,
            "\n".join(f"- {m}" for m in self._milestones),
        ]
        system_content = "\n\n".join(s for s in sections if s)

        messages: list[ChatMessage] = []
        if system_content:
            messages.append(ChatMessage(role=ChatRole.SYSTEM, content=system_content))
        for record in self._turns:
            messages.extend(record.messages)
        messages.extend(extra_messages or [])
        if self._scratchpad:
            scratchpad_text = "\n".join(f"{k}={r.value}" for k, r in self._scratchpad.items())
            messages.append(ChatMessage(role=ChatRole.SYSTEM, content=scratchpad_text))
        return messages

    # -- Reporting ------------------------------------------------------

    def snapshot(self) -> WorkbenchSnapshot:
        zones = [
            ZoneSnapshot("System Anchor", self.system_anchor_tokens, self._system_anchor_budget),
            ZoneSnapshot("User Memory", self.user_memory_tokens, self._user_memory_budget),
            ZoneSnapshot(
                "Mounted Skill",
                self.mounted_skill_tokens,
                self._mounted_skill_budget,
                detail="mounted" if self._mounted_skill_text else "(none)",
            ),
            ZoneSnapshot(
                "Milestones", self.milestones_tokens, self._milestones_budget, detail=f"{len(self._milestones)} milestone(s)"
            ),
            ZoneSnapshot(
                "Turn Horizon", self.turn_horizon_tokens, self._turn_horizon_budget, detail=f"{len(self._turns)} turn(s)"
            ),
            ZoneSnapshot(
                "Scratchpad",
                self.scratchpad_tokens,
                self._scratchpad_budget,
                detail=", ".join(f"{k}={r.value}" for k, r in self._scratchpad.items()),
            ),
        ]
        return WorkbenchSnapshot(
            zones=zones,
            total_tokens=sum(z.tokens for z in zones),
            max_total_tokens=self._max_total_tokens,
        )

    def render_report(self, *, bar_width: int = 20) -> str:
        snap = self.snapshot()
        lines = [f"FTW Context Workbench [Total: {snap.total_tokens:,} / {snap.max_total_tokens:,} max tokens]"]
        for i, zone in enumerate(snap.zones):
            branch = "└──" if i == len(snap.zones) - 1 else "├──"
            ratio = min(zone.tokens / zone.budget, 1.0) if zone.budget else 0.0
            filled = int(bar_width * ratio)
            bar = "[" + "=" * filled + " " * (bar_width - filled) + "]"
            detail = f" {zone.detail}" if zone.detail else ""
            lines.append(f"{branch} {zone.name + ':':<15s} {zone.tokens:>6,} tokens  {bar}{detail}")
        return "\n".join(lines)
