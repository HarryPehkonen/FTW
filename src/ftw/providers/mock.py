"""Scriptable provider for deterministic, offline tests of everything built
on top of IModelProvider (agent loop, workbench, REPL)."""

from __future__ import annotations

from dataclasses import dataclass

from ftw.providers.base import ChatMessage, IModelProvider, ProviderResponse, ToolSpec


@dataclass
class RecordedCall:
    messages: list[ChatMessage]
    tools: list[ToolSpec] | None


class MockModelProvider(IModelProvider):
    """Returns each entry of ``responses`` in order, one per ``complete()``
    call. Raises once exhausted, so a test's script always matches what the
    agent loop actually asked for."""

    def __init__(self, responses: list[ProviderResponse]):
        self._responses = list(responses)
        self.calls: list[RecordedCall] = []

    def complete(self, messages: list[ChatMessage], tools: list[ToolSpec] | None = None) -> ProviderResponse:
        self.calls.append(RecordedCall(messages=messages, tools=tools))
        if not self._responses:
            raise RuntimeError("MockModelProvider: no scripted responses left")
        return self._responses.pop(0)
