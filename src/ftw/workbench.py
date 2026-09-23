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
        self._turns: list[list[ChatMessage]] = []
        self._scratchpad: dict[str, str] = {}
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

    def add_turn(self, messages: list[ChatMessage]) -> None:
        self._turns.append(list(messages))
        while self.turn_horizon_tokens > self._turn_horizon_budget and len(self._turns) > 1:
            evicted = self._turns.pop(0)
            if self._on_turn_evicted is not None:
                self._on_turn_evicted(evicted)

    def clear_turns(self) -> None:
        while self._turns:
            evicted = self._turns.pop(0)
            if self._on_turn_evicted is not None:
                self._on_turn_evicted(evicted)

    @property
    def turns(self) -> list[list[ChatMessage]]:
        return [list(t) for t in self._turns]

    @property
    def turn_horizon_tokens(self) -> int:
        return sum(_turn_tokens(t) for t in self._turns)

    # -- Scratchpad: explicit pins, refused (not silently dropped) over budget

    def pin(self, key: str, value: str) -> None:
        candidate = dict(self._scratchpad)
        candidate[key] = value
        if self._scratchpad_tokens_of(candidate) > self._scratchpad_budget:
            raise WorkbenchBudgetExceeded(
                f"pinning {key!r} would exceed the scratchpad budget ({self._scratchpad_budget} tokens)"
            )
        self._scratchpad = candidate

    def unpin(self, key: str) -> None:
        del self._scratchpad[key]  # raises KeyError for an unknown key, deliberately

    @staticmethod
    def _scratchpad_tokens_of(scratchpad: dict[str, str]) -> int:
        return count_tokens("\n".join(f"{k}={v}" for k, v in scratchpad.items()))

    @property
    def scratchpad(self) -> dict[str, str]:
        return dict(self._scratchpad)

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
        for turn in self._turns:
            messages.extend(turn)
        messages.extend(extra_messages or [])
        if self._scratchpad:
            scratchpad_text = "\n".join(f"{k}={v}" for k, v in self._scratchpad.items())
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
                detail=", ".join(f"{k}={v}" for k, v in self._scratchpad.items()),
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
