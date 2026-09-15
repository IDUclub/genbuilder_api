"""``/generate/by_territory`` places services only for the region the caller names."""
import asyncio

import pytest
from pydantic import ValidationError

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


class _FakeBuilder:
    def __init__(self):
        self.calls: list[dict] = []

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "generated_buildings": {"type": "FeatureCollection", "features": []},
            "selected_features": {"type": "FeatureCollection", "features": []},
        }


@pytest.fixture
def builder(monkeypatch):
    fake = _FakeBuilder()
    monkeypatch.setattr(orchestration, "builder", fake)
    return fake


def test_the_region_reaches_the_generator(builder):
    payload = TerritoryRequest.model_validate({"blocks": BLOCKS, "territory_id": 1})

    asyncio.run(orchestration.generate_by_territory(payload))

    assert builder.calls[0]["territory_id"] == 1


def test_without_a_region_the_generator_gets_none(builder):
    asyncio.run(orchestration.generate_by_territory(TerritoryRequest.model_validate({"blocks": BLOCKS})))

    assert builder.calls[0]["territory_id"] is None


def test_the_region_id_must_be_positive():
    with pytest.raises(ValidationError):
        TerritoryRequest.model_validate({"blocks": BLOCKS, "territory_id": 0})
