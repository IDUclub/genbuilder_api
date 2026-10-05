"""Facade scene of ``/generate/chat/stream/3d``: events, and the GLB link in history.

The scene is produced inside the chat stream, after the summary and before the
assistant turn is persisted, so a ready GLB is linked from the chat history the
same way the stored geo layers are.
"""

import asyncio
import json

from test_generation_chat_layers import (  # noqa: F401 - _stub_extraction is an autouse fixture
    _FakeBuilder,
    _FakeChatStorage,
    _FakeLLM,
    _assistant_message,
    _stub_extraction,
)
from test_generation_chat_llm_failure import _chat_client, _sse_events

from app.exceptions.http_exception_wrapper import http_exception
from app.infrastructure.object_storage import LocalStorage
from app.logic.chat.generation_chat import stream_generation_chat
from app.routers import generation_chat_routers

SCENE_ID = "0123456789abcdef0123456789abcdef"

SCENE_READY = {
    "status": "ready",
    "result_id": SCENE_ID,
    "glb_url": f"/facade-scenes/{SCENE_ID}.glb",
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

SCENE_FILE = {
    "name": "facade_scene",
    "title": "3D-модель застройки",
    "role": "result",
    "url": f"/facade-scenes/{SCENE_ID}.glb",
    "download_url": None,
    "filename": f"{SCENE_ID}.glb",
    "mime_type": "model/gltf-binary",
    "source_service": "genbuilder",
}


def _producer(outcome):
    calls: list[dict] = []

    async def produce(buildings, **kwargs):
        calls.append({"buildings": buildings, **kwargs})
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return produce, calls


def _collect(**overrides) -> list[dict]:
    defaults = dict(
        builder=_FakeBuilder(),
        llm_client=_FakeLLM(),
        chat_storage_client=None,
        token="user-token",
        user_id=None,
        user_query="Построй кирпичные дома на 5000 человек",
        scenario_id=198,
        year=2024,
        source="OSM",
        la_per_person=30.0,
        facade_style="Кирпичный",
        enable_facade_styles=True,
    )
    defaults.update(overrides)

    async def _run() -> list[dict]:
        return [event async for event in stream_generation_chat(**defaults)]

    return asyncio.run(_run())


def _types(events):
    return [event["type"] for event in events]


def _strip(event):
    return {k: v for k, v in event.items() if k != "type"}


def _file_parts(message):
    return [part["payload"] for part in message.get("parts", []) if part["kind"] == "file"]


# --- the stream -----------------------------------------------------------------


def test_ready_scene_follows_the_summary_and_precedes_done():
    produce, calls = _producer(SCENE_READY)

    events = _collect(facade_scene_producer=produce)

    types = _types(events)
    assert types[-3:] == ["facade_scene", "file", "done"]
    last_token = max(i for i, t in enumerate(types) if t == "token")
    assert last_token < types.index("facade_scene")
    assert _strip(events[-3]) == SCENE_READY
    assert _strip(events[-2]) == SCENE_FILE
    result = next(event for event in events if event["type"] == "result")
    assert calls == [
        {
            "buildings": result["content"],
            "facade_style": result["facade_style_prompt"],
            "facade_style_name_ru": "Кирпичный",
        }
    ]


def test_ready_scene_glb_is_linked_from_the_chat_history(tmp_path):
    produce, _ = _producer(SCENE_READY)
    storage = _FakeChatStorage()

    _collect(
        facade_scene_producer=produce,
        chat_storage_client=storage,
        user_id="user-1",
        object_storage=LocalStorage(str(tmp_path)),
    )

    message = _assistant_message(storage)
    files = _file_parts(message)
    assert [part["name"] for part in files] == ["buildings", "facade_scene"]
    assert files[-1] == {
        "url": SCENE_FILE["url"],
        "name": "facade_scene",
        "title": "3D-модель застройки",
        "filename": SCENE_FILE["filename"],
        "mime_type": "model/gltf-binary",
        "source_service": "genbuilder",
    }
    # ``origin`` places the GLB on the map when the chat is reopened.
    assert message["metadata"]["facade_scene"] == SCENE_READY


def test_glb_is_linked_even_without_geo_layer_storage():
    produce, _ = _producer(SCENE_READY)
    storage = _FakeChatStorage()

    _collect(facade_scene_producer=produce, chat_storage_client=storage, user_id="user-1")

    files = _file_parts(_assistant_message(storage))
    assert [part["name"] for part in files] == ["facade_scene"]


def test_queued_job_is_kept_in_metadata_without_a_file_part():
    produce, _ = _producer(QUEUED)
    storage = _FakeChatStorage()

    events = _collect(
        facade_scene_producer=produce, chat_storage_client=storage, user_id="user-1"
    )

    assert _types(events)[-2:] == ["facade_job", "done"]
    assert _strip(events[-2]) == QUEUED
    message = _assistant_message(storage)
    assert _file_parts(message) == []
    assert message["metadata"]["facade_scene"] == QUEUED


def test_facade_failure_is_an_error_and_the_turn_is_still_persisted():
    produce, _ = _producer(
        http_exception(422, "Free-text facade styles need facade-jobs.")
    )
    storage = _FakeChatStorage()

    events = _collect(
        facade_scene_producer=produce, chat_storage_client=storage, user_id="user-1"
    )

    assert _types(events)[-2:] == ["error", "done"]
    assert events[-2]["stage"] == "facade_job"
    assert events[-1]["assistant_message_id"] is not None
    assert "facade_scene" not in _assistant_message(storage)["metadata"]


def test_unexpected_facade_failure_does_not_lose_the_turn():
    produce, _ = _producer(RuntimeError("library exploded"))
    storage = _FakeChatStorage()

    events = _collect(
        facade_scene_producer=produce, chat_storage_client=storage, user_id="user-1"
    )

    assert _types(events)[-2:] == ["error", "done"]
    assert events[-2] == {
        "type": "error",
        "stage": "facade_job",
        "detail": "library exploded",
    }
    assert events[-1]["assistant_message_id"] is not None


def test_regular_chat_never_produces_a_scene():
    produce, calls = _producer(SCENE_READY)

    events = _collect(facade_scene_producer=produce, enable_facade_styles=False)

    assert "facade_scene" not in _types(events)
    assert calls == []


# --- the router -----------------------------------------------------------------


def _post(monkeypatch, path):
    received: list[dict] = []
    orchestration_calls: list[dict] = []

    async def _stream(**kwargs):
        received.append(kwargs)
        producer = kwargs["facade_scene_producer"]
        if producer is not None:
            scene = await producer(
                {"type": "FeatureCollection", "features": []},
                facade_style="red brick facade",
                facade_style_name_ru="Кирпичный",
            )
            yield {"type": "facade_scene", **scene}
        yield {"type": "done", "chat_id": "chat-1", "assistant_message_id": None}

    async def _produce(buildings, **kwargs):
        orchestration_calls.append({"buildings": buildings, **kwargs})
        return SCENE_READY

    client = _chat_client(monkeypatch, _stream)
    monkeypatch.setattr(
        generation_chat_routers.orchestration, "produce_chat_facade_scene", _produce
    )
    response = client.post(
        path,
        data={
            "user_query": "кирпичные дома",
            "scenario_id": 843,
            "year": 2026,
            "source": "User",
        },
    )
    return response, received, orchestration_calls


def test_3d_route_produces_the_scene_for_the_caller(monkeypatch):
    response, received, calls = _post(monkeypatch, "/generate/chat/stream/3d")

    assert response.status_code == 200
    assert _sse_events(response.text) == ["facade_scene", "done"]
    payload = json.loads(
        next(
            line.split(":", 1)[1]
            for line in response.text.splitlines()
            if line.startswith("data:")
        )
    )
    assert payload == SCENE_READY
    assert received[0]["enable_facade_styles"] is True
    assert calls == [
        {
            "buildings": {"type": "FeatureCollection", "features": []},
            "requested_by": "user-1",
            "facade_style": "red brick facade",
            "facade_style_name_ru": "Кирпичный",
        }
    ]


def test_regular_route_has_no_scene_producer(monkeypatch):
    response, received, calls = _post(monkeypatch, "/generate/chat/stream")

    assert _sse_events(response.text) == ["done"]
    assert received[0]["facade_scene_producer"] is None
    assert calls == []
