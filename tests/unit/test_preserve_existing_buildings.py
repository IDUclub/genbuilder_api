"""``preserve_existing_buildings``: scenario modes load the scenario's buildings
from UrbanDB, cut them out of the territory — and refuse to generate when they
can't be loaded, instead of silently building on top of them."""
import asyncio

import pytest
from fastapi import HTTPException

from app.logic import generation_orchestration as orchestration
from app.schema.dto import FunctionalZonesRequest


def _square(lon, lat, size):
    return {
        "type": "Polygon",
        "coordinates": [[[lon, lat], [lon + size, lat], [lon + size, lat + size], [lon, lat + size], [lon, lat]]],
    }


ZONE_GEOMETRY = _square(31.0, 59.9, 0.01)

HOUSE_INSIDE = {
    "type": "Feature",
    "properties": {
        "physical_object_id": 10,
        "physical_object_type": {"physical_object_type_id": 4, "name": "Жилой дом"},
        "building": {"floors": 5, "properties": {"living_area": 3000}},
    },
    "geometry": _square(31.001, 59.901, 0.0005),
}
HOUSE_OUTSIDE = {
    "type": "Feature",
    "properties": {
        "physical_object_id": 11,
        "physical_object_type": {"physical_object_type_id": 4, "name": "Жилой дом"},
        "building": {"floors": 9},
    },
    "geometry": _square(32.0, 60.5, 0.0005),
}
ROAD = {
    "type": "Feature",
    "properties": {"physical_object_id": 12, "physical_object_type": {"physical_object_type_id": 30}},
    "geometry": {"type": "LineString", "coordinates": [[31.0, 59.9], [31.01, 59.91]]},
}


class _FakeUrbanDb:
    def __init__(self, physical_objects=None, error=None, zones=None):
        self.physical_objects_calls = 0
        self._physical_objects = physical_objects or {"type": "FeatureCollection", "features": []}
        self._error = error
        self._zones = zones or {"type": "FeatureCollection", "features": []}

    async def get_physical_objects(self, scenario_id, token, physical_object_type_id=None):
        self.physical_objects_calls += 1
        if self._error is not None:
            raise self._error
        return self._physical_objects

    async def get_scenario_functional_zones(self, scenario_id, source, year, token):
        return self._zones


class _FakeBuilder:
    def __init__(self, generated=()):
        self.calls: list[dict] = []
        self._generated = list(generated)

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "generated_buildings": {"type": "FeatureCollection", "features": self._generated},
            "selected_features": {"type": "FeatureCollection", "features": []},
        }


def _fc(*features):
    return {"type": "FeatureCollection", "features": list(features)}


def _generate_by_scenario(**overrides):
    kwargs = dict(
        scenario_id=1,
        year=2024,
        source="OSM",
        functional_zone_types=["residential"],
        physical_object_id=None,
        token="t",
        targets_by_zone={"residents": {"residential": 100}},
        generation_parameters=None,
    )
    kwargs.update(overrides)
    return asyncio.run(orchestration.generate_by_scenario(**kwargs))


def test_scenario_without_flag_does_not_load_buildings(monkeypatch):
    urban_db, builder = _FakeUrbanDb(), _FakeBuilder()
    monkeypatch.setattr(orchestration, "urban_db_api", urban_db)
    monkeypatch.setattr(orchestration, "builder", builder)

    _generate_by_scenario(physical_object_id=[10])

    assert urban_db.physical_objects_calls == 0
    assert builder.calls[0]["existing_buildings"] is None
    assert builder.calls[0]["physical_object_ids"] == [10]


def test_scenario_passes_existing_buildings_and_dedupes_ids(monkeypatch):
    urban_db = _FakeUrbanDb(physical_objects=_fc(HOUSE_INSIDE, HOUSE_OUTSIDE, ROAD))
    builder = _FakeBuilder()
    monkeypatch.setattr(orchestration, "urban_db_api", urban_db)
    monkeypatch.setattr(orchestration, "builder", builder)

    _generate_by_scenario(physical_object_id=[10, 12], preserve_existing_buildings=True)

    call = builder.calls[0]
    existing = call["existing_buildings"]["features"]
    assert sorted(f["properties"]["physical_object_id"] for f in existing) == [10, 11]
    assert all(f["properties"]["is_excluded"] for f in existing)
    # the house is already covered by existing_buildings; only the road stays an id exclusion
    assert call["physical_object_ids"] == [12]


def test_scenario_fails_loudly_when_buildings_cannot_be_loaded(monkeypatch):
    urban_db = _FakeUrbanDb(error=RuntimeError("connection reset"))
    builder = _FakeBuilder()
    monkeypatch.setattr(orchestration, "urban_db_api", urban_db)
    monkeypatch.setattr(orchestration, "builder", builder)

    with pytest.raises(HTTPException) as exc_info:
        _generate_by_scenario(preserve_existing_buildings=True)

    assert exc_info.value.status_code == 502
    assert builder.calls == []


def test_blocks_pass_only_buildings_inside_requested_zones(monkeypatch):
    zone = {
        "type": "Feature",
        "properties": {"functional_zone_id": 7, "functional_zone_type": {"name": "residential"}},
        "geometry": ZONE_GEOMETRY,
    }
    urban_db = _FakeUrbanDb(physical_objects=_fc(HOUSE_INSIDE, HOUSE_OUTSIDE), zones=_fc(zone))
    builder = _FakeBuilder()
    monkeypatch.setattr(orchestration, "urban_db_api", urban_db)
    monkeypatch.setattr(orchestration, "builder", builder)

    body = FunctionalZonesRequest.model_validate(
        {"zones": [{"functional_zone_id": 7, "targets_by_zone": {"residents": {"residential": 500}}}]}
    )
    progress = []

    async def _progress(done, total):
        progress.append((done, total))

    asyncio.run(
        orchestration.generate_by_blocks(
            scenario_id=1,
            year=2024,
            source="OSM",
            functional_zone_types=["residential"],
            physical_object_id=None,
            token="t",
            body=body,
            preserve_existing_buildings=True,
            progress=_progress,
        )
    )

    assert progress == [(1, 1)]
    existing = builder.calls[0]["existing_buildings"]["features"]
    assert [f["properties"]["physical_object_id"] for f in existing] == [10]


def test_list_functional_zones_reports_ids_types_and_areas(monkeypatch):
    zones = _fc(
        {
            "type": "Feature",
            "properties": {"functional_zone_id": 7, "functional_zone_type": {"name": "residential_midrise"}},
            "geometry": ZONE_GEOMETRY,
        },
        {
            "type": "Feature",
            "properties": {"functional_zone_id": 8, "functional_zone_type": {"name": "industrial"}},
            "geometry": _square(31.02, 59.9, 0.01),
        },
        {
            "type": "Feature",
            "properties": {"functional_zone_type": {"name": "industrial"}},
            "geometry": _square(31.04, 59.9, 0.01),
        },
    )
    monkeypatch.setattr(orchestration, "urban_db_api", _FakeUrbanDb(zones=zones))

    result = asyncio.run(
        orchestration.list_functional_zones(scenario_id=1, year=2024, source="OSM", token="t")
    )

    assert [z["functional_zone_id"] for z in result["zones"]] == [7, 8]
    first = result["zones"][0]
    assert first["functional_zone_type"] == "residential_midrise"
    assert first["generation_zone"] == "residential"
    # 0.01° x 0.01° at ~59.9°N is roughly 1113 m x 557 m
    assert 550_000 < first["area_m2"] < 700_000
    assert result["totals_by_type"]["industrial"]["count"] == 1

    filtered = asyncio.run(
        orchestration.list_functional_zones(
            scenario_id=1, year=2024, source="OSM", token="t", functional_zone_types=["industrial"]
        )
    )
    assert [z["functional_zone_id"] for z in filtered["zones"]] == [8]


class _FakeZonesService:
    async def prepare_blocks_by_zones(self, *, scenario_id, year, source, token, functional_zone_types, zone_ids):
        from app.schema.dto import BlockFeatureCollection

        return {
            7: BlockFeatureCollection.model_validate(
                _fc({"type": "Feature", "properties": {"block_id": 7, "zone": "residential"}, "geometry": ZONE_GEOMETRY})
            )
        }


@pytest.mark.parametrize("preserve", [False, True])
def test_capacity_estimate_reports_area_living_area_and_existing(monkeypatch, preserve):
    generated = [
        {"type": "Feature", "properties": {"residents_number": 200, "living_area": 4000.0}, "geometry": None},
        {"type": "Feature", "properties": {"residents_number": 100, "living_area": 2000.5}, "geometry": None},
    ]
    builder = _FakeBuilder(generated=generated)
    monkeypatch.setattr(orchestration, "urban_db_api", _FakeUrbanDb(physical_objects=_fc(HOUSE_INSIDE, HOUSE_OUTSIDE)))
    monkeypatch.setattr(orchestration, "zones_service", _FakeZonesService())
    monkeypatch.setattr(orchestration, "builder", builder)
    progress = []

    async def _progress(done, total):
        progress.append((done, total))

    estimates = asyncio.run(
        orchestration.estimate_capacity_by_blocks(
            scenario_id=1,
            year=2024,
            source="OSM",
            functional_zone_types=["residential"],
            functional_zone_ids=[7],
            token="t",
            preserve_existing_buildings=preserve,
            progress=_progress,
        )
    )

    estimate = estimates[7]
    assert progress == [(1, 1)]
    assert estimate["functional_zone_type"] == "residential"
    assert estimate["max_residents"] == 300
    assert estimate["max_living_area"] == 6000.5
    assert 550_000 < estimate["zone_area_m2"] < 700_000
    assert estimate["existing_buildings_count"] == 1
    existing = builder.calls[0]["existing_buildings"]
    if preserve:
        assert [f["properties"]["physical_object_id"] for f in existing["features"]] == [10]
    else:
        assert existing is None
