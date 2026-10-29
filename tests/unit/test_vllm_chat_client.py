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
    responses: dict[str, httpx.Response | list[httpx.Response]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = responses["response"]
        return response.pop(0) if isinstance(response, list) else response

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
    assert payload["priority"] == -100
    assert payload["model"] == "gpt-oss-20b"
    assert payload["temperature"] == 0.7
    assert "reasoning_effort" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


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
    assert payload["priority"] == -60
    assert payload["temperature"] == 0.0
    assert "reasoning_effort" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == SCHEMA


def test_base_url_with_v1_suffix_is_not_duplicated(captured):
    requests, responses = captured
    responses["response"] = httpx.Response(
        200, json={"choices": [{"message": {"content": "{}"}}]}
    )

    async def run() -> None:
        async with VLLMChatClient(
            "http://vllm:8001/v1/", default_model="gpt-oss-20b"
        ) as client:
            await client.complete_json(
                [{"role": "user", "content": "hi"}], schema=SCHEMA
            )

    asyncio.run(run())
    assert str(requests[0].url) == "http://vllm:8001/v1/chat/completions"


@pytest.mark.parametrize("model", ["qwen3.8-27b", "Qwen/Qwen3.8-27B-FP8", "deployment-alias"])
def test_qwen_requests_direct_answers_without_matching_the_model_name(captured, model):
    requests, responses = captured
    responses["response"] = httpx.Response(200, json={"choices": [{"message": {"content": '{"zones": []}'}}]})

    async def run():
        async with VLLMChatClient("http://vllm:8001", default_model="old-model") as client:
            await client.complete_json([{"role": "user", "content": "hi"}], schema=SCHEMA, model=model)

    asyncio.run(run())
    payload = _body(requests[0])
    assert payload["model"] == model
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in payload


def test_explicit_reasoning_effort_remains_supported(captured):
    requests, responses = captured
    responses["response"] = httpx.Response(200, json={"choices": [{"message": {"content": '{"zones": []}'}}]})

    async def run():
        async with VLLMChatClient("http://vllm:8001", default_model="qwen3.8-27b") as client:
            await client.complete_json(
                [{"role": "user", "content": "hi"}], schema=SCHEMA, reasoning_effort="low"
            )

    asyncio.run(run())
    payload = _body(requests[0])
    assert payload["reasoning_effort"] == "low"
    assert "chat_template_kwargs" not in payload


@pytest.mark.parametrize("request_priority,chat_priority", [(-60, -100), (-20, -80), (0, 0)])
def test_configured_priorities_reach_both_request_types(captured, request_priority, chat_priority):
    requests, responses = captured
    responses["response"] = [
        httpx.Response(200, json={"choices": [{"message": {"content": '{"zones": []}'}}]}),
        httpx.Response(200, content=STREAM_BODY.encode("utf-8")),
    ]

    async def run():
        async with VLLMChatClient(
            "http://vllm:8001", default_model="qwen3.8-27b",
            request_priority=request_priority, chat_priority=chat_priority,
        ) as client:
            await client.complete_json([], schema=SCHEMA)
            assert [delta async for delta in client.stream_chat([])] == ["Итого", ": 120 зданий"]

    asyncio.run(run())
    assert [_body(request)["priority"] for request in requests] == [request_priority, chat_priority]


def test_error_status_raises(captured):
    _, responses = captured
    responses["response"] = httpx.Response(404, text='{"error": "model not found"}')

    async def run() -> None:
        async with VLLMChatClient(
            "http://vllm:8001", default_model="missing"
        ) as client:
            await client.complete_json(
                [{"role": "user", "content": "hi"}], schema=SCHEMA
            )

    with pytest.raises(VLLMChatError) as exc:
        asyncio.run(run())
    assert exc.value.status == 404


def test_non_object_json_raises(captured):
    _, responses = captured
    responses["response"] = httpx.Response(
        200, json={"choices": [{"message": {"content": "[1, 2]"}}]}
    )

    async def run() -> None:
        async with VLLMChatClient(
            "http://vllm:8001", default_model="gpt-oss-20b"
        ) as client:
            await client.complete_json(
                [{"role": "user", "content": "hi"}], schema=SCHEMA
            )

    with pytest.raises(VLLMChatError):
        asyncio.run(run())


def test_missing_base_url_raises():
    with pytest.raises(RuntimeError, match="VLLM_BASE_URL"):
        VLLMChatClient("", default_model="gpt-oss-20b")


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
@pytest.mark.parametrize("content", [None, "", "  "])
def test_complete_json_recovers_structured_answer_from_reasoning(
    captured, field, content
):
    requests, responses = captured
    responses["response"] = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": content,
                        field: '{"zones": ["residential"]}',
                    },
                }
            ]
        },
    )

    async def run():
        async with VLLMChatClient(
            "http://vllm:8001", default_model="gpt-oss-20b"
        ) as client:
            return await client.complete_json(
                [{"role": "user", "content": "количество 500"}], schema=SCHEMA
            )

    assert asyncio.run(run()) == {"zones": ["residential"]}
    assert len(requests) == 1


def test_recovered_answer_reaches_generation_target_extraction(captured):
    from app.logic.chat.param_extraction import extract_generation_targets

    requests, responses = captured
    responses["response"] = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": None,
                        "reasoning": json.dumps({
                            "zones": [],
                            "total_residents": 500,
                            "facade_style_name_ru": None,
                            "facade_style_prompt": None,
                        }),
                    },
                }
            ]
        },
    )

    async def run():
        async with VLLMChatClient(
            "http://vllm:8001", default_model="gpt-oss-20b"
        ) as client:
            return await extract_generation_targets(
                client, user_query="количество жителей 500", la_per_person=30.0
            )

    targets = asyncio.run(run())
    assert targets.error is None
    assert targets.total_residents == 500
    assert len(requests) == 1
    assert _body(requests[0])["priority"] == -60


def test_chat_title_uses_the_semi_blocking_priority(captured):
    from app.logic.chat.chat_title import make_chat_title

    requests, responses = captured
    responses["response"] = httpx.Response(
        200, json={"choices": [{"message": {"content": '{"title": "Генерация жилья на 500 жителей"}'}}]}
    )

    async def run():
        async with VLLMChatClient("http://vllm:8001", default_model="qwen3.8-27b") as client:
            return await make_chat_title(client, user_query="количество жителей 500")

    assert asyncio.run(run()) == "Генерация жилья на 500 жителей"
    assert _body(requests[0])["priority"] == -60


@pytest.mark.parametrize(
    "reasoning,finish_reason",
    [
        ("Thinking about the request", "stop"),
        ('Let me think: {"zones": []}', "stop"),
        ('{"unrelated": "object"}', "stop"),
        ('{"zones": [123]}', "stop"),
        ('[{"zones": []}]', "stop"),
        ('{"zones": []}', "length"),
        ('{"zones": []}', None),
    ],
)
def test_complete_json_retries_unusable_reasoning(captured, reasoning, finish_reason):
    requests, responses = captured
    responses["response"] = [
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": finish_reason,
                        "message": {"content": None, "reasoning": reasoning},
                    }
                ]
            },
        ),
        httpx.Response(
            200, json={"choices": [{"message": {"content": '{"zones": []}'}}]}
        ),
    ]

    async def run():
        async with VLLMChatClient(
            "http://vllm:8001", default_model="gpt-oss-20b"
        ) as client:
            return await client.complete_json(
                [{"role": "user", "content": "hi"}], schema=SCHEMA
            )

    assert asyncio.run(run()) == {"zones": []}
    assert len(requests) == 2
    assert _body(requests[0]) == _body(requests[1])
    assert [_body(request)["priority"] for request in requests] == [-60, -60]


def test_complete_json_bounds_empty_content_retries(captured):
    requests, responses = captured
    responses["response"] = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": None, "reasoning": "private reasoning text"},
                }
            ]
        },
    )

    async def run():
        async with VLLMChatClient(
            "http://vllm:8001", default_model="gpt-oss-20b"
        ) as client:
            return await client.complete_json(
                [{"role": "user", "content": "hi"}], schema=SCHEMA
            )

    with pytest.raises(VLLMChatError) as exc:
        asyncio.run(run())
    assert len(requests) == 2
    assert "finish_reason='stop'" in str(exc.value)
    assert "private reasoning text" not in str(exc.value)


@pytest.fixture
def unreachable(monkeypatch):
    """Make every request fail the way an unreachable vLLM host does."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed", request=request)

    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)


def test_complete_json_wraps_transport_failure(unreachable):
    async def run() -> None:
        async with VLLMChatClient(
            "http://vllm:8001", default_model="gpt-oss-20b"
        ) as client:
            await client.complete_json(
                [{"role": "user", "content": "hi"}], schema=SCHEMA
            )

    with pytest.raises(VLLMChatError) as exc:
        asyncio.run(run())
    assert exc.value.status == 0
    assert "ConnectError" in str(exc.value)


def test_stream_chat_wraps_transport_failure(unreachable):
    async def run() -> list[str]:
        async with VLLMChatClient(
            "http://vllm:8001", default_model="gpt-oss-20b"
        ) as client:
            return [
                delta
                async for delta in client.stream_chat(
                    [{"role": "user", "content": "hi"}]
                )
            ]

    with pytest.raises(VLLMChatError) as exc:
        asyncio.run(run())
    assert exc.value.status == 0


def test_non_json_body_raises(captured):
    _, responses = captured
    responses["response"] = httpx.Response(200, text="<html>502 Bad Gateway</html>")

    async def run() -> None:
        async with VLLMChatClient(
            "http://vllm:8001", default_model="gpt-oss-20b"
        ) as client:
            await client.complete_json(
                [{"role": "user", "content": "hi"}], schema=SCHEMA
            )

    with pytest.raises(VLLMChatError):
        asyncio.run(run())
