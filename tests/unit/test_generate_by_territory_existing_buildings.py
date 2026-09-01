"""``/generate/by_territory`` is the project-less mode: no scenario means no
physical objects to exclude by id, so existing buildings ride along in the body.
"""
import asyncio

import pytest

from app.logic import generation_orchestration as orchestration
from app.schema.dto import TerritoryRequest

BLOCKS = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"zone": "residential"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[31.0, 59.91], [31.01, 59.91], [31.01, 59.92], [31.0, 59.91]]],
            },
        }
    ],
}

EXISTING = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"floors_count": 5},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[31.001, 59.911], [31.002, 59.911], [31.002, 59.912], [31.001, 59.911]]],
            },
        }
    ],
}

EXCLUDED_FEATURE = {
    "type": "Feature",
    "properties": {"is_excluded": True, "floors_count": 5.0},
    "geometry": EXISTING["features"][0]["geometry"],
}


class _FakeBuilder:
    def __init__(self, selected=()):
        self.calls: list[dict] = []
        self._selected = list(selected)

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "generated_buildings": {"type": "FeatureCollection", "features": []},
            "selected_features": {
                "type": "FeatureCollection",
                "features": self._selected,
            },
        }


@pytest.fixture
def builder(monkeypatch):
    fake = _FakeBuilder(selected=[EXCLUDED_FEATURE])
    monkeypatch.setattr(orchestration, "builder", fake)
    return fake


def _run(payload):
    return asyncio.run(orchestration.generate_by_territory(payload))


def test_existing_buildings_reach_the_generator(builder):
    payload = TerritoryRequest.model_validate(
        {"blocks": BLOCKS, "existing_buildings": EXISTING}
    )

    _run(payload)

    passed = builder.calls[0]["existing_buildings"]
    assert [f["geometry"]["type"] for f in passed["features"]] == ["Polygon"]
    assert passed["features"][0]["properties"] == {"floors_count": 5}


def test_no_existing_buildings_is_passed_as_none(builder):
    _run(TerritoryRequest.model_validate({"blocks": BLOCKS}))

    assert builder.calls[0]["existing_buildings"] is None


def test_excluded_buildings_come_back_in_the_response(builder):
    payload = TerritoryRequest.model_validate(
        {"blocks": BLOCKS, "existing_buildings": EXISTING}
    )

    result = _run(payload)

    assert result["features"] == [EXCLUDED_FEATURE]
