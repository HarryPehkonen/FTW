"""Adapter for any OpenAI-compatible chat-completions endpoint: DeepSeek,
the Nous Research Portal, and local runtimes (llama.cpp, vLLM, Ollama)
all speak this same wire format (ftw_plan.md §6).
"""

from __future__ import annotations

import json

import httpx

from ftw.providers.base import (
    ChatMessage,
    ChatRole,
    IModelProvider,
    ProviderError,
    ProviderResponse,
    ToolCall,
    ToolSpec,
)


class OpenAICompatibleProvider(IModelProvider):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self._model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )

    def complete(self, messages: list[ChatMessage], tools: list[ToolSpec] | None = None) -> ProviderResponse:
        payload: dict = {"model": self._model, "messages": [self._encode_message(m) for m in messages]}
        if tools:
            payload["tools"] = [self._encode_tool(t) for t in tools]

        try:
            resp = self._client.post("/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"{self._model}: HTTP {exc.response.status_code}: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self._model}: request failed: {exc}") from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError(f"{self._model}: response was not valid JSON") from exc

        return self._decode_response(data)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenAICompatibleProvider":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- wire encoding/decoding -------------------------------------------

    @staticmethod
    def _encode_message(m: ChatMessage) -> dict:
        if m.role == ChatRole.TOOL:
            return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}

        out: dict = {"role": m.role.value, "content": m.content}
        if m.tool_calls:
            out["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in m.tool_calls
            ]
        return out

    @staticmethod
    def _encode_tool(t: ToolSpec) -> dict:
        return {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }

    @classmethod
    def _decode_response(cls, data: dict) -> ProviderResponse:
        try:
            raw_message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"response missing choices[0].message: {data!r}") from exc

        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=cls._decode_arguments(tc["function"].get("arguments", "{}")),
            )
            for tc in raw_message.get("tool_calls") or []
        ]

        message = ChatMessage(
            role=ChatRole.ASSISTANT,
            content=raw_message.get("content"),
            tool_calls=tool_calls,
        )
        usage = {k: v for k, v in (data.get("usage") or {}).items() if isinstance(v, int)}
        return ProviderResponse(message=message, usage=usage, raw=data)

    @staticmethod
    def _decode_arguments(raw: str) -> dict:
        try:
            parsed = json.loads(raw) if raw else {}
        except ValueError as exc:
            raise ProviderError(f"tool call arguments were not valid JSON: {raw!r}") from exc
        if not isinstance(parsed, dict):
            raise ProviderError(f"tool call arguments must decode to an object: {raw!r}")
        return parsed
