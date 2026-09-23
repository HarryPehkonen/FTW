"""Pre-Commit Interceptor Pipeline (ftw_plan.md §3.4).

Runs between the model's action proposal and the framework firing the NNG
CALL. Phase 1 ships one rule — every shell command requires human
confirmation. Path sandboxing, blocklists, and step budgets replace that
blanket rule in Phase 4; the pipeline shape doesn't change, only the rules
in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ftw.protocol import CallEnvelope


class InterceptDecision(str, Enum):
    ALLOW = "ALLOW"
    ASK = "ASK"
    BLOCK = "BLOCK"
    # MODIFY: deferred to Phase 4, when a rule can rewrite a call in place.


@dataclass
class InterceptOutcome:
    decision: InterceptDecision
    reason: str = ""


class InterceptRule(Protocol):
    def evaluate(self, call: CallEnvelope) -> InterceptOutcome: ...


class ConfirmShellCommands:
    """Phase 1's entire policy: any ``run_command`` call must be confirmed
    by a human before it's allowed to fire."""

    def evaluate(self, call: CallEnvelope) -> InterceptOutcome:
        if call.payload.action == "run_command":
            return InterceptOutcome(InterceptDecision.ASK, reason="shell commands require confirmation")
        return InterceptOutcome(InterceptDecision.ALLOW)


class PreCommitInterceptor:
    """Runs its rules in order and stops at the first non-ALLOW decision."""

    def __init__(self, rules: list[InterceptRule]):
        self._rules = list(rules)

    def evaluate(self, call: CallEnvelope) -> InterceptOutcome:
        for rule in self._rules:
            outcome = rule.evaluate(call)
            if outcome.decision != InterceptDecision.ALLOW:
                return outcome
        return InterceptOutcome(InterceptDecision.ALLOW)
