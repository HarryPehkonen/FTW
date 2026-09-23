"""Contract tests for IModelProvider implementations (ftw_plan.md §6).

MockModelProvider is what the rest of the test suite (agent loop, REPL,
workbench) scripts against — zero network calls, zero live LLM tokens.
OpenAICompatibleProvider is tested against httpx.MockTransport, never a
real endpoint.
"""

import httpx
import pytest

from ftw.providers import (
    ChatMessage,
    ChatRole,
    MockModelProvider,
    OpenAICompatibleProvider,
    ProviderError,
    ProviderResponse,
    ToolCall,
    ToolSpec,
)


class TestChatMessage:
    def test_defaults(self):
        msg = ChatMessage(role=ChatRole.USER, content="hi")
        assert msg.tool_calls == []
        assert msg.tool_call_id is None


class TestMockModelProvider:
    def test_returns_scripted_responses_in_order(self):
        r1 = ProviderResponse(message=ChatMessage(role=ChatRole.ASSISTANT, content="first"))
        r2 = ProviderResponse(message=ChatMessage(role=ChatRole.ASSISTANT, content="second"))
        provider = MockModelProvider([r1, r2])

        assert provider.complete([ChatMessage(role=ChatRole.USER, content="a")]) is r1
        assert provider.complete([ChatMessage(role=ChatRole.USER, content="b")]) is r2

    def test_records_calls_for_assertions(self):
        provider = MockModelProvider([ProviderResponse(message=ChatMessage(role=ChatRole.ASSISTANT, content="ok"))])
        tools = [ToolSpec(name="run_command", description="run a shell command", parameters={"type": "object"})]
        messages = [ChatMessage(role=ChatRole.USER, content="do it")]

        provider.complete(messages, tools=tools)

        assert len(provider.calls) == 1
        assert provider.calls[0].messages == messages
        assert provider.calls[0].tools == tools

    def test_raises_when_scripted_responses_exhausted(self):
        provider = MockModelProvider([])
        with pytest.raises(RuntimeError, match="no scripted responses left"):
            provider.complete([ChatMessage(role=ChatRole.USER, content="a")])


class TestOpenAICompatibleProviderRequest:
    def test_sends_openai_shaped_request_with_auth_header(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            captured["body"] = request.read()
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "hi there"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3},
                },
            )

        provider = OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            api_key="secret-key",
            model="fast-model",
            transport=httpx.MockTransport(handler),
        )
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content="be terse"),
            ChatMessage(role=ChatRole.USER, content="hello"),
        ]

        provider.complete(messages)

        req = captured["request"]
        assert req.url.path == "/v1/chat/completions"
        assert req.headers["authorization"] == "Bearer secret-key"
        import json

        body = json.loads(captured["body"])
        assert body["model"] == "fast-model"
        assert body["messages"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hello"},
        ]
        assert "tools" not in body

    def test_omits_auth_header_when_no_api_key(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["request"] = request
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

        provider = OpenAICompatibleProvider(
            base_url="http://localhost:8080/v1",
            api_key=None,
            model="local-model",
            transport=httpx.MockTransport(handler),
        )
        provider.complete([ChatMessage(role=ChatRole.USER, content="hi")])

        assert "authorization" not in captured["request"].headers

    def test_encodes_tool_specs_and_prior_tool_calls(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = request.read()
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "done"}}]})

        provider = OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            api_key="k",
            model="m",
            transport=httpx.MockTransport(handler),
        )
        messages = [
            ChatMessage(role=ChatRole.USER, content="run echo"),
            ChatMessage(
                role=ChatRole.ASSISTANT,
                content=None,
                tool_calls=[ToolCall(id="call-1", name="run_command", arguments={"argv": ["echo", "hi"]})],
            ),
            ChatMessage(role=ChatRole.TOOL, tool_call_id="call-1", name="run_command", content="hi\n"),
        ]
        tools = [ToolSpec(name="run_command", description="Run a shell command.", parameters={"type": "object"})]

        provider.complete(messages, tools=tools)

        import json

        body = json.loads(captured["body"])
        assert body["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command.",
                    "parameters": {"type": "object"},
                },
            }
        ]
        assistant_msg = body["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] is None
        assert assistant_msg["tool_calls"] == [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "run_command", "arguments": '{"argv": ["echo", "hi"]}'},
            }
        ]
        tool_msg = body["messages"][2]
        assert tool_msg == {"role": "tool", "tool_call_id": "call-1", "content": "hi\n"}


class TestOpenAICompatibleProviderResponse:
    def _provider(self, response_json: dict) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            api_key="k",
            model="m",
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json=response_json)),
        )

    def test_decodes_plain_text_response(self):
        provider = self._provider(
            {
                "choices": [{"message": {"role": "assistant", "content": "the answer is 4"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 6},
            }
        )
        result = provider.complete([ChatMessage(role=ChatRole.USER, content="2+2?")])

        assert result.message.role == ChatRole.ASSISTANT
        assert result.message.content == "the answer is 4"
        assert result.message.tool_calls == []
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 6}

    def test_decodes_tool_call_response(self):
        provider = self._provider(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-9",
                                    "type": "function",
                                    "function": {
                                        "name": "run_command",
                                        "arguments": '{"argv": ["ls"]}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )
        result = provider.complete([ChatMessage(role=ChatRole.USER, content="list files")])

        assert result.message.content is None
        assert len(result.message.tool_calls) == 1
        call = result.message.tool_calls[0]
        assert call.id == "call-9"
        assert call.name == "run_command"
        assert call.arguments == {"argv": ["ls"]}

    def test_raises_provider_error_on_http_failure(self):
        provider = OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            api_key="bad-key",
            model="m",
            transport=httpx.MockTransport(lambda req: httpx.Response(401, json={"error": "unauthorized"})),
        )
        with pytest.raises(ProviderError):
            provider.complete([ChatMessage(role=ChatRole.USER, content="hi")])

    def test_raises_provider_error_on_malformed_body(self):
        provider = OpenAICompatibleProvider(
            base_url="https://api.example.com/v1",
            api_key="k",
            model="m",
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"nonsense": True})),
        )
        with pytest.raises(ProviderError):
            provider.complete([ChatMessage(role=ChatRole.USER, content="hi")])
