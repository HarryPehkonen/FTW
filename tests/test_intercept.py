"""Pre-Commit Interceptor Pipeline (ftw_plan.md §3.4).

Phase 1 ships exactly one rule: every shell command must be confirmed by a
human. Real policy (path sandboxing, blocklists, budgets) is Phase 4; until
then the human is the policy, and nothing fires on the bus without either
an ALLOW or an explicit yes to an ASK.
"""

from ftw.intercept import ConfirmShellCommands, InterceptDecision, PreCommitInterceptor
from ftw.protocol import CallEnvelope, CallPayload


def shell_call(argv: list[str]) -> CallEnvelope:
    return CallEnvelope(
        source="repl.master",
        target="worker.tool.shell",
        payload=CallPayload(action="run_command", args={"argv": argv}),
    )


def other_call() -> CallEnvelope:
    return CallEnvelope(
        source="repl.master",
        target="worker.memory",
        payload=CallPayload(action="search", args={"query": "cmake"}),
    )


class TestConfirmShellCommandsRule:
    def test_shell_command_requires_confirmation(self):
        rule = ConfirmShellCommands()
        outcome = rule.evaluate(shell_call(["rm", "-rf", "build"]))
        assert outcome.decision == InterceptDecision.ASK
        assert outcome.reason

    def test_non_shell_call_is_allowed(self):
        rule = ConfirmShellCommands()
        outcome = rule.evaluate(other_call())
        assert outcome.decision == InterceptDecision.ALLOW


class TestPreCommitInterceptor:
    def test_empty_pipeline_allows_everything(self):
        pipeline = PreCommitInterceptor([])
        assert pipeline.evaluate(shell_call(["ls"])).decision == InterceptDecision.ALLOW

    def test_pipeline_with_confirm_rule_asks_for_shell_calls(self):
        pipeline = PreCommitInterceptor([ConfirmShellCommands()])
        assert pipeline.evaluate(shell_call(["ls"])).decision == InterceptDecision.ASK

    def test_first_non_allow_decision_short_circuits(self):
        class AlwaysBlock:
            def evaluate(self, call):
                from ftw.intercept import InterceptOutcome

                return InterceptOutcome(InterceptDecision.BLOCK, reason="nope")

        calls = []

        class RecordsIfReached:
            def evaluate(self, call):
                calls.append(call)
                from ftw.intercept import InterceptOutcome

                return InterceptOutcome(InterceptDecision.ALLOW)

        pipeline = PreCommitInterceptor([AlwaysBlock(), RecordsIfReached()])
        outcome = pipeline.evaluate(shell_call(["ls"]))

        assert outcome.decision == InterceptDecision.BLOCK
        assert calls == []  # second rule never ran
