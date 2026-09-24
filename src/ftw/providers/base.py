"""Normalized chat/tool-calling types and the IModelProvider interface
(ftw_plan.md §6).

Every provider adapter (DeepSeek, Nous, a local OpenAI-compatible endpoint,
the test mock) speaks this one internal shape. Translating to and from a
given wire format is the adapter's job, not the agent loop's.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: ChatRole
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None  # role=TOOL: which call this answers
    name: str | None = None  # role=TOOL: the tool name, for providers that want it


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments object


class ProviderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: ChatMessage
    usage: dict[str, int] = Field(default_factory=dict)
    raw: dict[str, Any] | None = None


class ProviderError(Exception):
    """Raised for any provider-side failure: HTTP errors, malformed
    responses, auth failures. The agent loop treats every provider the
    same way regardless of transport."""


class IModelProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec] | None = None,
    ) -> ProviderResponse: ...
