import asyncio
import json

import httpx
import pytest

from app.infrastructure.vllm_chat_client import VLLMChatClient, VLLMChatError

SCHEMA = {
    "type": "object",
    "properties": {"zones": {"type": "array", "items": {"type": "string"}}},
    "required": ["zones"],
}

STREAM_BODY = (
    'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
    'data: {"choices":[{"delta":{"reasoning":"thinking hard"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"Итого"}}]}\n\n'
    'data: {"choices":[]}\n\n'
    'data: {"choices":[{"delta":{"content":": 120 зданий"}}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


@pytest.fixture
def captured(monkeypatch):
    """Route the client's httpx transport to a stub, capturing every request."""
    requests: list[httpx.Request] = []
    responses: dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responses["response"]

    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return requests, responses


def _body(request: httpx.Request) -> dict:
    return json.loads(request.content)


def test_stream_chat_yields_content_and_skips_reasoning(captured):
    requests, responses = captured
    responses["response"] = httpx.Response(200, content=STREAM_BODY.encode("utf-8"))

    async def run() -> list[str]:
        async with VLLMChatClient(
            "http://vllm:8001", default_model="gpt-oss-20b"
        ) as client:
            return [
                delta
                async for delta in client.stream_chat(
                    [{"role": "user", "content": "hi"}], temperature=0.7
                )
            ]

    assert asyncio.run(run()) == ["Итого", ": 120 зданий"]

    request = requests[0]
    assert str(request.url) == "http://vllm:8001/v1/chat/completions"
    payload = _body(request)
    assert payload["stream"] is True
    assert payload["model"] == "gpt-oss-20b"
    assert payload["temperature"] == 0.7
    assert payload["reasoning_effort"] == "low"


def test_complete_json_sends_schema_and_parses_content(captured):
    requests, responses = captured
    responses["response"] = httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": '{"zones": ["residential"]}', "reasoning": "…"}}
            ]
        },
    )

    async def run() -> dict:
        async with VLLMChatClient(
            "http://vllm:8001", default_model="gpt-oss-20b"
        ) as client:
            return await client.complete_json(
                [{"role": "user", "content": "hi"}], schema=SCHEMA
            )

    assert asyncio.run(run()) == {"zones": ["residential"]}

    payload = _body(requests[0])
    assert payload["stream"] is False
    assert payload["temperature"] == 0.0
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == SCHEMA


def test_base_url_with_v1_suffix_is_not_duplicated(captured):
    requests, responses = captured
    responses["response"] = httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    async def run() -> None:
        async with VLLMChatClient(
            "http://vllm:8001/v1/", default_model="gpt-oss-20b"
        ) as client:
            await client.complete_json([{"role": "user", "content": "hi"}], schema=SCHEMA)

    asyncio.run(run())
    assert str(requests[0].url) == "http://vllm:8001/v1/chat/completions"


def test_error_status_raises(captured):
    _, responses = captured
    responses["response"] = httpx.Response(404, text='{"error": "model not found"}')

    async def run() -> None:
        async with VLLMChatClient(
            "http://vllm:8001", default_model="missing"
        ) as client:
            await client.complete_json([{"role": "user", "content": "hi"}], schema=SCHEMA)

    with pytest.raises(VLLMChatError) as exc:
        asyncio.run(run())
    assert exc.value.status == 404


def test_non_object_json_raises(captured):
    _, responses = captured
    responses["response"] = httpx.Response(200, json={"choices": [{"message": {"content": "[1, 2]"}}]})

    async def run() -> None:
        async with VLLMChatClient(
            "http://vllm:8001", default_model="gpt-oss-20b"
        ) as client:
            await client.complete_json([{"role": "user", "content": "hi"}], schema=SCHEMA)

    with pytest.raises(VLLMChatError):
        asyncio.run(run())


def test_missing_base_url_raises():
    with pytest.raises(RuntimeError, match="LLM_API"):
        VLLMChatClient("", default_model="gpt-oss-20b")
