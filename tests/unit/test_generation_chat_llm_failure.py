"""The chat stream must always terminate with an event, never with a dead socket.

The frontend can only report "поток завершился без результата" when the SSE
response closes with no ``result``/``error``/``done`` on it — which is what an
unhandled exception inside the generator produces. These tests pin the two
guards: an LLM that cannot be reached is reported as an ``error`` event, and any
unexpected failure is still turned into ``error`` + ``done`` by the router.
"""
import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sse_starlette.sse import AppStatus

from app.logic.chat import generation_chat
from app.logic.chat.generation_chat import stream_generation_chat
from app.logic.chat.param_extraction import ExtractedTargets
from app.routers import generation_chat_routers
from app.utils import auth

LLM_DOWN = "vllm /v1/chat/completions is unreachable: ConnectError: All connection attempts failed"


class _FakeBuilder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"generated_buildings": {"type": "FeatureCollection", "features": []}}


class _FakeLLM:
    async def stream_chat(self, messages, **kwargs):
        yield "неважно"
        return


def _collect(**overrides) -> list[dict]:
    defaults = dict(
        builder=_FakeBuilder(),
        llm_client=_FakeLLM(),
        chat_storage_client=None,
        token="user-token",
        user_id=None,
        user_query="5000",
        scenario_id=843,
        year=2026,
        source="User",
        la_per_person=30.0,
    )
    defaults.update(overrides)

    async def _run() -> list[dict]:
        return [event async for event in stream_generation_chat(**defaults)]

    return asyncio.run(_run())


def test_unreachable_llm_is_reported_not_asked_again(monkeypatch):
    async def _extract(llm_client, *, user_query, la_per_person, model=None):
        return ExtractedTargets(raw={"error": LLM_DOWN}, error=LLM_DOWN)

    monkeypatch.setattr(generation_chat, "extract_generation_targets", _extract)
    builder = _FakeBuilder()

    events = _collect(builder=builder)

    assert [event["type"] for event in events] == ["error", "done"]
    assert events[0]["stage"] == "param_extraction"
    assert events[0]["detail"] == LLM_DOWN
    assert "Языковая модель недоступна" in events[0]["message"]
    # A clarification here would ask for parameters the user already gave.
    assert builder.calls == []


def _sse_events(text: str) -> list[str]:
    return [
        line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if line.startswith("event:")
    ]


def _chat_client(monkeypatch, stream):
    # sse_starlette keeps a process-wide exit event bound to the first event loop.
    monkeypatch.setattr(AppStatus, "should_exit_event", None)
    monkeypatch.setattr(generation_chat_routers, "stream_generation_chat", stream)
    monkeypatch.setattr(generation_chat_routers, "chat_llm_configured", lambda: True)
    monkeypatch.setattr(
        generation_chat_routers, "build_chat_storage_client", lambda: None
    )
    monkeypatch.setattr(generation_chat_routers, "optional_object_storage", lambda: None)
    monkeypatch.setattr(generation_chat_routers, "public_base_url", lambda: None)

    class _DummyLLM:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return None

    monkeypatch.setattr(
        generation_chat_routers, "build_vllm_chat_client", lambda temperature: _DummyLLM()
    )

    app = FastAPI()
    app.include_router(generation_chat_routers.generation_chat_router)
    app.dependency_overrides[auth.get_current_user] = lambda: auth.AuthUser(
        token="user-token", user_id="user-1"
    )
    return TestClient(app)


def test_unexpected_failure_still_terminates_the_stream(monkeypatch):
    async def _stream(**kwargs):
        yield {"type": "chat_created", "chat_id": "chat-42", "title": "Жильё"}
        raise RuntimeError("boom")

    client = _chat_client(monkeypatch, _stream)

    response = client.post(
        "/generate/chat/stream",
        data={"user_query": "5000", "scenario_id": 843, "year": 2026, "source": "User"},
    )

    assert response.status_code == 200
    assert _sse_events(response.text) == ["chat_created", "error", "done"]
    payloads = [
        json.loads(line.split(":", 1)[1].strip())
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]
    assert payloads[1]["stage"] == "stream"
    assert "RuntimeError: boom" == payloads[1]["detail"]
    # The chat was created before the failure — the client needs its id back.
    assert payloads[2]["chat_id"] == "chat-42"


@pytest.mark.parametrize("path", ["/generate/chat/stream", "/generate/chat/stream/3d"])
def test_territory_id_reaches_the_chat_stream(monkeypatch, path):
    received: list[dict] = []

    async def _stream(**kwargs):
        received.append(kwargs)
        yield {"type": "done", "chat_id": None, "assistant_message_id": None}

    client = _chat_client(monkeypatch, _stream)
    monkeypatch.setattr(generation_chat_routers, "facade_jobs_configured", lambda: True)

    response = client.post(
        path,
        data={"user_query": "5000", "territory_id": 47},
        files={"blocks_file": ("blocks.geojson", b'{"type": "FeatureCollection", "features": []}')},
    )

    assert response.status_code == 200
    assert received[0]["territory_id"] == 47
