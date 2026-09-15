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


def test_unknown_request_fields_are_rejected_instead_of_silently_ignored():
    with pytest.raises(ValidationError) as caught:
        TerritoryRequest.model_validate({"blocks": BLOCKS, "region_id": 1})

    assert caught.value.errors()[0]["type"] == "extra_forbidden"


def test_service_diagnostics_are_preserved_in_the_public_result():
    diagnostics = {
        "territory_id_provided": True,
        "territory_id": 1,
        "normatives_found": 2,
        "services_requested": 2,
        "services_placed": 1,
        "services_unplaced": 1,
        "service_buildings_placed": 1,
        "capacity_requested": 150.0,
        "capacity_placed": 100.0,
        "capacity_unplaced": 50.0,
        "status": "partial",
        "warning": "не все сервисы удалось разместить",
    }

    result = orchestration.merge_generation_result(
        {
            "generated_buildings": {"type": "FeatureCollection", "features": []},
            "selected_features": {"type": "FeatureCollection", "features": []},
            "service_diagnostics": diagnostics,
        }
    )

    assert result["service_diagnostics"] == diagnostics
