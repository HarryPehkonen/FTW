"""The REPL loop, decoupled from any real terminal (ftw_plan.md §3.6, §7
test_repl_flow.py).

``ReplSession`` takes input as a callable and output as a stream, so the
whole interactive loop — conversation turns, slash commands, EOF handling —
is unit-testable with a scripted input list and no subprocess. ``cli.py``
is the thin layer that wires this to a real terminal (readline, stdin,
stdout) plus worker supervision.
"""

from __future__ import annotations

import sys
from typing import Callable, TextIO

from ftw.agent_loop import AgentLoop
from ftw.protocol import CallEnvelope

InputFn = Callable[[str], str]


def make_confirm(input_fn: InputFn, output: TextIO) -> Callable[[CallEnvelope], bool]:
    """Builds a Pre-Commit Interceptor ``confirm`` callback that asks the
    human over the given input/output — the same channel the REPL prompt
    itself uses. Fails closed (declines) on EOF, exactly like the
    AgentLoop's own default."""

    def confirm(call: CallEnvelope) -> bool:
        print(f"Confirm {call.payload.action} {call.payload.args}? [y/N] ", end="", file=output)
        try:
            answer = input_fn("")
        except EOFError:
            print("(EOF — declining)", file=output)
            return False
        return answer.strip().lower() in ("y", "yes")

    return confirm


class ReplSession:
    def __init__(
        self,
        *,
        agent_loop: AgentLoop,
        input_fn: InputFn,
        output: TextIO = sys.stdout,
        prompt: str = "ftw> ",
    ):
        self.agent_loop = agent_loop
        self._input_fn = input_fn
        self._output = output
        self._prompt = prompt

    def run(self) -> None:
        while True:
            try:
                line = self._input_fn(self._prompt)
            except EOFError:
                return
            if not self.handle_line(line):
                return

    def handle_line(self, line: str) -> bool:
        """Processes one line. Returns False when the session should end."""
        line = line.strip()
        if not line:
            return True
        if line.startswith("/"):
            return self._handle_command(line)
        reply = self.agent_loop.run_turn(line)
        self._print(reply)
        return True

    def _handle_command(self, line: str) -> bool:
        parts = line.split(maxsplit=1)
        cmd = parts[0]

        if cmd == "/exit":
            return False
        if cmd == "/context":
            self._print(self.agent_loop.workbench.render_report())
            return True
        if cmd == "/clear":
            self.agent_loop.workbench.clear_turns()
            self._print("cleared turn horizon")
            return True

        self._print(f"unknown command: {cmd}")
        return True

    def _print(self, text: str) -> None:
        print(text, file=self._output)
