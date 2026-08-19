import asyncio
import json

import pytest

from app.infrastructure.object_storage import LocalStorage, ObjectStorageError
from app.logic.chat import generation_chat
from app.logic.chat.generation_chat import stream_generation_chat
from app.logic.chat.param_extraction import ExtractedTargets
from app.logic.geo_layers import SLOT_BLOCKS_INPUT, object_key

PUBLIC_BASE_URL = "http://10.32.1.46:8200"
ANSWER = "Сгенерировано 1 здание."

ZONES_LAYER = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"zone": "residential", "functional_zone_id": 11},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[30.0, 60.0], [30.1, 60.0], [30.1, 60.1], [30.0, 60.0]]],
            },
        }
    ],
}

GENERATED = {
    "generated_buildings": {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "zone": "residential",
                    "living_area": 120.0,
                    "residents_number": 4,
                },
                "geometry": {"type": "Point", "coordinates": [30.05, 60.05]},
            }
        ],
    },
    "selected_features": {"type": "FeatureCollection", "features": []},
}


def _block(zone: str, x: float) -> dict:
    return {
        "type": "Feature",
        "properties": {"zone": zone},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[x, 60.0], [x + 0.01, 60.0], [x + 0.01, 60.01], [x, 60.0]]
            ],
        },
    }


class _FakeBuilder:
    def __init__(self, result=None, exc=None):
        self._result = result if result is not None else GENERATED
        self._exc = exc
        self.calls: list[dict] = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeLLM:
    async def stream_chat(self, messages, model=None, temperature=None):
        for delta in ("Сгенерировано ", "1 здание."):
            yield delta


class _FakeChatStorage:
    def __init__(self):
        self.messages: list[dict] = []

    async def create_chat(self, user_id, **kwargs):
        return {"chat_id": "chat-1", "title": kwargs.get("title")}

    async def get_chat(self, user_id, chat_id):
        return {"messages": []}

    async def add_message(self, user_id, chat_id, **kwargs):
        self.messages.append(kwargs)
        return {"message_id": f"msg-{len(self.messages)}"}


class _FakeZones:
    def __init__(self, layer=None, exc=None):
        self._layer = layer if layer is not None else ZONES_LAYER
        self._exc = exc
        self.calls: list[dict] = []

    async def prepare_zones_layer(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return self._layer


class _BrokenStorage(LocalStorage):
    def put_json(self, payload, object_key):
        raise ObjectStorageError("bucket is on fire")


@pytest.fixture(autouse=True)
def _stub_extraction(monkeypatch):
    """The extractor is an LLM call; pin it so the stream is deterministic."""

    async def _extract(llm_client, *, user_query, la_per_person, model=None):
        return ExtractedTargets(
            targets_by_zone={"residents": {"residential": 5000, "business": 200}},
            functional_zone_types=[],
            raw={},
        )

    monkeypatch.setattr(generation_chat, "extract_generation_targets", _extract)


def _collect(**overrides) -> list[dict]:
    defaults = dict(
        builder=_FakeBuilder(),
        llm_client=_FakeLLM(),
        chat_storage_client=None,
        token="user-token",
        user_id=None,
        user_query="Построй жильё на 5000 человек",
        scenario_id=198,
        year=2024,
        source="OSM",
        la_per_person=30.0,
        public_base_url=PUBLIC_BASE_URL,
    )
    defaults.update(overrides)

    async def _run() -> list[dict]:
        return [event async for event in stream_generation_chat(**defaults)]

    return asyncio.run(_run())


def _types(events):
    return [event["type"] for event in events]


def _of_type(events, event_type):
    return [event for event in events if event["type"] == event_type]


def _assistant_message(storage_client):
    return [m for m in storage_client.messages if m["role"] == "assistant"][0]


def _result_id(descriptor):
    return descriptor["url"].rsplit("/", 1)[-1]


def test_scenario_mode_streams_zones_before_generation(tmp_path):
    events = _collect(
        zones_service=_FakeZones(), object_storage=LocalStorage(str(tmp_path))
    )

    types = _types(events)
    assert types.index("zones") < types.index("progress") < types.index("result")
    zone_event = _of_type(events, "zones")[0]
    assert zone_event["source"] == "scenario"
    assert zone_event["content"] == ZONES_LAYER


def test_scenario_zones_are_fetched_with_the_caller_token():
    zones = _FakeZones()

    _collect(zones_service=zones)

    assert zones.calls == [
        {
            "scenario_id": 198,
            "year": 2024,
            "source": "OSM",
            "token": "user-token",
            "functional_zone_types": ["residential", "business"],
        }
    ]


def test_zones_descriptor_is_a_live_query_not_a_stored_object():
    events = _collect(zones_service=_FakeZones())

    descriptor = _of_type(events, "file")[0]
    assert descriptor["name"] == "functional_zones"
    assert descriptor["url"].startswith(f"{PUBLIC_BASE_URL}/layers/functional_zones?")
    assert "scenario_id=198" in descriptor["url"]
    assert "functional_zone_types=residential" in descriptor["url"]
    assert descriptor["download_url"] is None


def test_buildings_are_stored_and_linked_after_the_result(tmp_path):
    storage = LocalStorage(str(tmp_path))

    events = _collect(object_storage=storage)

    types = _types(events)
    assert types.index("result") < types.index("file")
    descriptor = _of_type(events, "file")[0]
    assert descriptor["url"] == f"{PUBLIC_BASE_URL}/files/buildings/{_result_id(descriptor)}"

    raw = b"".join(storage.open_stream(object_key(_result_id(descriptor), "buildings")))
    assert json.loads(raw.decode("utf-8")) == _of_type(events, "result")[0]["content"]


def test_nothing_is_stored_without_a_storage_backend():
    events = _collect(object_storage=None)

    assert _of_type(events, "file") == []
    assert _of_type(events, "result")


def test_blocks_file_mode_uses_the_filtered_blocks_as_backdrop(tmp_path):
    uploaded = {
        "type": "FeatureCollection",
        "features": [_block("residential", 30.0), _block("recreation", 30.5)],
    }

    events = _collect(
        scenario_id=None,
        year=None,
        source=None,
        blocks_geojson=uploaded,
        zones_service=_FakeZones(),
        object_storage=LocalStorage(str(tmp_path)),
    )

    zone_event = _of_type(events, "zones")[0]
    assert zone_event["source"] == "blocks_file"
    zones = [f["properties"]["zone"] for f in zone_event["content"]["features"]]
    assert zones == ["residential"]


def test_blocks_file_mode_never_links_the_scenario_zones(tmp_path):
    """The file defines the territory, so a scenario link would describe something else."""
    zones = _FakeZones()

    events = _collect(
        blocks_geojson={
            "type": "FeatureCollection",
            "features": [_block("residential", 30.0)],
        },
        zones_service=zones,
        object_storage=LocalStorage(str(tmp_path)),
    )

    names = [descriptor["name"] for descriptor in _of_type(events, "file")]
    assert "functional_zones" not in names
    assert zones.calls == []


def test_uploaded_blocks_are_stored_exactly_as_uploaded(tmp_path):
    """History keeps the user's own file, dropped features included."""
    storage = LocalStorage(str(tmp_path))
    uploaded = {
        "type": "FeatureCollection",
        "features": [_block("residential", 30.0), _block("recreation", 30.5)],
    }

    events = _collect(blocks_geojson=uploaded, object_storage=storage)

    descriptor = [
        d for d in _of_type(events, "file") if d["name"] == SLOT_BLOCKS_INPUT
    ][0]
    key = object_key(_result_id(descriptor), SLOT_BLOCKS_INPUT)
    assert json.loads(b"".join(storage.open_stream(key)).decode("utf-8")) == uploaded


def test_both_slots_share_one_result_id(tmp_path):
    events = _collect(
        blocks_geojson={
            "type": "FeatureCollection",
            "features": [_block("residential", 30.0)],
        },
        object_storage=LocalStorage(str(tmp_path)),
    )

    assert len({_result_id(d) for d in _of_type(events, "file")}) == 1


def test_storage_failure_warns_and_the_stream_still_finishes(tmp_path):
    events = _collect(object_storage=_BrokenStorage(str(tmp_path)))

    warnings = [e for e in _of_type(events, "warning") if e["stage"] == "store_layer"]
    assert warnings and "bucket is on fire" in warnings[0]["detail"]
    assert _of_type(events, "file") == []
    assert _types(events)[-1] == "done"


def test_zones_failure_warns_and_generation_still_runs():
    zones = _FakeZones(exc=RuntimeError("urban_db is down"))

    events = _collect(zones_service=zones)

    warnings = [e for e in _of_type(events, "warning") if e["stage"] == "zones"]
    assert warnings and "urban_db is down" in warnings[0]["detail"]
    assert _of_type(events, "result")
    assert _types(events)[-1] == "done"


def test_history_keeps_the_answer_and_the_layer_links(tmp_path):
    storage_client = _FakeChatStorage()

    events = _collect(
        chat_storage_client=storage_client,
        user_id="user-1",
        zones_service=_FakeZones(),
        object_storage=LocalStorage(str(tmp_path)),
    )

    assistant = _assistant_message(storage_client)
    assert [part["kind"] for part in assistant["parts"]] == ["text", "file", "file"]
    assert assistant["parts"][0]["payload"]["text"] == ANSWER
    urls = [part["payload"]["url"] for part in assistant["parts"][1:]]
    assert urls == [d["url"] for d in _of_type(events, "file")]


def test_persisted_file_parts_never_carry_an_ephemeral_download_url(tmp_path):
    storage_client = _FakeChatStorage()

    _collect(
        chat_storage_client=storage_client,
        user_id="user-1",
        zones_service=_FakeZones(),
        object_storage=LocalStorage(str(tmp_path)),
    )

    assistant = _assistant_message(storage_client)
    for part in assistant["parts"][1:]:
        assert "download_url" not in part["payload"]


def test_history_falls_back_to_plain_content_without_layers():
    storage_client = _FakeChatStorage()

    _collect(chat_storage_client=storage_client, user_id="user-1", object_storage=None)

    assistant = _assistant_message(storage_client)
    assert assistant["content"] == ANSWER
    assert "parts" not in assistant


def test_a_failed_generation_stores_nothing(tmp_path):
    events = _collect(
        builder=_FakeBuilder(exc=RuntimeError("solver diverged")),
        object_storage=LocalStorage(str(tmp_path)),
    )

    assert _types(events) == ["status", "progress", "error", "done"]
    assert _of_type(events, "file") == []
    assert not list(tmp_path.iterdir())
