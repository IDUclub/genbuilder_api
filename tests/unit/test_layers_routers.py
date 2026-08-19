import json
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.infrastructure.object_storage import LocalStorage, get_object_storage
from app.logic.geo_layers import SLOT_BUILDINGS, object_key
from app.routers.layers_routers import get_zones_service, layers_router
from app.utils import auth

RESULT_ID = uuid4().hex
BUILDINGS = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"zone": "residential", "floors_count": 9},
            "geometry": {"type": "Point", "coordinates": [30.3, 59.9]},
        }
    ],
}
ZONES = {"type": "FeatureCollection", "features": []}


class _RecordingZonesService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def prepare_zones_layer(self, **kwargs):
        self.calls.append(kwargs)
        return ZONES


def _client(tmp_path, *, authenticated: bool = True):
    app = FastAPI()
    app.include_router(layers_router)

    storage = LocalStorage(str(tmp_path))
    zones = _RecordingZonesService()

    app.dependency_overrides[get_object_storage] = lambda: storage
    app.dependency_overrides[get_zones_service] = lambda: zones
    if authenticated:
        app.dependency_overrides[auth.verify_token] = lambda: "user-token"

    return TestClient(app), storage, zones


def test_zones_layer_forwards_params_and_caller_token(tmp_path):
    client, _, zones = _client(tmp_path)

    response = client.get(
        "/layers/functional_zones",
        params={
            "scenario_id": 198,
            "year": 2024,
            "source": "OSM",
            "functional_zone_types": ["residential", "business"],
        },
    )

    assert response.status_code == 200
    assert response.json() == ZONES
    assert zones.calls == [
        {
            "scenario_id": 198,
            "year": 2024,
            "source": "OSM",
            "token": "user-token",
            "functional_zone_types": ["residential", "business"],
        }
    ]


def test_zones_layer_without_type_filter_passes_none(tmp_path):
    client, _, zones = _client(tmp_path)

    response = client.get(
        "/layers/functional_zones",
        params={"scenario_id": 198, "year": 2024, "source": "OSM"},
    )

    assert response.status_code == 200
    assert zones.calls[0]["functional_zone_types"] is None


def test_zones_layer_rejects_a_missing_scenario(tmp_path):
    client, _, _ = _client(tmp_path)

    response = client.get("/layers/functional_zones", params={"year": 2024})

    assert response.status_code == 422


def test_zones_layer_requires_a_token(tmp_path):
    client, _, _ = _client(tmp_path, authenticated=False)

    response = client.get(
        "/layers/functional_zones",
        params={"scenario_id": 198, "year": 2024, "source": "OSM"},
    )

    assert response.status_code == 401


def test_file_streams_a_stored_layer(tmp_path):
    client, storage, _ = _client(tmp_path)
    storage.put_json(BUILDINGS, object_key(RESULT_ID, SLOT_BUILDINGS))

    response = client.get(f"/files/{SLOT_BUILDINGS}/{RESULT_ID}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    assert "buildings.geojson" in response.headers["content-disposition"]
    assert json.loads(response.content.decode("utf-8")) == BUILDINGS


def test_file_requires_a_token(tmp_path):
    client, storage, _ = _client(tmp_path, authenticated=False)
    storage.put_json(BUILDINGS, object_key(RESULT_ID, SLOT_BUILDINGS))

    response = client.get(f"/files/{SLOT_BUILDINGS}/{RESULT_ID}")

    assert response.status_code == 401


def test_file_returns_404_for_an_unknown_slot(tmp_path):
    client, _, _ = _client(tmp_path)

    response = client.get(f"/files/secrets/{RESULT_ID}")

    assert response.status_code == 404


def test_file_returns_404_for_an_expired_result(tmp_path):
    client, _, _ = _client(tmp_path)

    response = client.get(f"/files/{SLOT_BUILDINGS}/{uuid4().hex}")

    assert response.status_code == 404


@pytest.mark.parametrize(
    "result_id",
    ["../../etc/passwd", "not-a-uuid", "A" * 32, RESULT_ID + "extra"],
)
def test_file_rejects_a_malformed_result_id(tmp_path, result_id):
    """The id lands in an object key, so anything but uuid4 hex is refused."""
    client, _, _ = _client(tmp_path)

    response = client.get(f"/files/{SLOT_BUILDINGS}/{result_id}")

    assert response.status_code == 404
