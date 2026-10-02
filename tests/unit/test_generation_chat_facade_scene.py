"""Facade events of ``/generate/chat/stream/3d``: a ready library scene, a queued job or an error."""

import json

from test_generation_chat_llm_failure import _chat_client, _sse_events

from app.exceptions.http_exception_wrapper import http_exception
from app.routers import generation_chat_routers

BUILDINGS = {"type": "FeatureCollection", "features": []}

SCENE_READY = {
    "status": "ready",
    "result_id": "0123456789abcdef0123456789abcdef",
    "glb_url": "/facade-scenes/0123456789abcdef0123456789abcdef.glb",
    "origin": {"lon": 30.05, "lat": 60.05},
    "facade_style": "Кирпичный",
    "style_by_zone": {"residential": "brick"},
    "source": "library",
    "stats": {
        "buildings": 3,
        "wall_instances": 24,
        "floor_instances": 144,
        "template_count": 2,
        "nearest_substitutions": 0,
    },
}

QUEUED = {
    "status": "queued",
    "job_id": "job-1",
    "status_url": "http://facade-jobs/jobs/job-1",
    "facade_style": "Песчаник",
}


async def _stream(**kwargs):
    yield {
        "type": "result",
        "content": BUILDINGS,
        "summary": {},
        "facade_style": "Кирпичный",
        "facade_style_prompt": "red brick facade",
    }
    yield {"type": "done", "chat_id": "chat-1", "assistant_message_id": None}


def _post(monkeypatch, produce):
    calls: list[dict] = []

    async def _produce(buildings, **kwargs):
        calls.append({"buildings": buildings, **kwargs})
        return await produce()

    client = _chat_client(monkeypatch, _stream)
    monkeypatch.setattr(
        generation_chat_routers.orchestration, "produce_chat_facade_scene", _produce
    )
    response = client.post(
        "/generate/chat/stream/3d",
        data={
            "user_query": "кирпичные дома",
            "scenario_id": 843,
            "year": 2026,
            "source": "User",
        },
    )
    return response, calls


def _payloads(text: str) -> list[dict]:
    return [
        json.loads(line.split(":", 1)[1].strip())
        for line in text.splitlines()
        if line.startswith("data:")
    ]


def test_ready_library_scene_arrives_as_facade_scene_before_done(monkeypatch):
    async def produce():
        return SCENE_READY

    response, calls = _post(monkeypatch, produce)

    assert response.status_code == 200
    assert _sse_events(response.text) == ["result", "facade_scene", "done"]
    assert _payloads(response.text)[1] == SCENE_READY
    assert calls == [
        {
            "buildings": BUILDINGS,
            "requested_by": "user-1",
            "facade_style": "red brick facade",
            "facade_style_name_ru": "Кирпичный",
        }
    ]


def test_queued_scene_arrives_as_facade_job(monkeypatch):
    async def produce():
        return QUEUED

    response, _ = _post(monkeypatch, produce)

    assert _sse_events(response.text) == ["result", "facade_job", "done"]
    assert _payloads(response.text)[1] == QUEUED


def test_facade_failure_is_an_error_event_and_the_stream_still_ends(monkeypatch):
    async def produce():
        raise http_exception(422, "Free-text facade styles need facade-jobs.")

    response, _ = _post(monkeypatch, produce)

    assert _sse_events(response.text) == ["result", "error", "done"]
    assert _payloads(response.text)[1]["stage"] == "facade_job"
