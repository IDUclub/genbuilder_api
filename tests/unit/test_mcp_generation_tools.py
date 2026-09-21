"""MCP generation tools: no silent default targets, every result carries a
``summary``, the layer is stored instead of inlined, and runs are reproducible."""
import asyncio
import json

import pytest
from mcp import McpError

from app.infrastructure.object_storage import LocalStorage, ObjectStorageError
from app.logic import generation_orchestration as orchestration
from app.logic.geo_layers import SLOT_BUILDINGS, object_key
from app.mcp_server.tools import generation as tools

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

SERVICE_DIAGNOSTICS = {
    "territory_id_provided": True,
    "territory_id": 1,
    "normatives_found": 3,
    "services_requested": 2,
    "services_placed": 1,
    "services_unplaced": 1,
    "service_buildings_placed": 1,
    "capacity_requested": 150.0,
    "capacity_placed": 100.0,
    "capacity_unplaced": 50.0,
    "status": "partial",
    "warning": "не все сервисы размещены",
}

RESULT_FC = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"zone": "residential", "residents_number": 800}, "geometry": None},
        {"type": "Feature", "properties": {"is_excluded": True, "residents_number": 50}, "geometry": None},
    ],
    "service_diagnostics": SERVICE_DIAGNOSTICS,
}


async def _token():
    return "token"


def _scenario_args(**overrides):
    kwargs = dict(scenario_id=1, year=2024, source="OSM", functional_zone_types=["residential"])
    kwargs.update(overrides)
    return kwargs


@pytest.fixture(autouse=True)
def storage(monkeypatch, tmp_path):
    local = LocalStorage(str(tmp_path))
    monkeypatch.setattr(tools, "optional_object_storage", lambda: local)
    monkeypatch.setattr(tools, "public_base_url", lambda: None)
    return local


def _fake_scenario_generation(monkeypatch):
    calls = []

    async def _generate(**kwargs):
        calls.append(kwargs)
        return RESULT_FC

    monkeypatch.setattr(tools, "require_verified_token", _token)
    monkeypatch.setattr(orchestration, "generate_by_scenario", _generate)
    return calls


_TARGETS = {"residents": {"residential": 800}}


def test_scenario_requires_targets_or_use_defaults(monkeypatch):
    async def _no_token():
        raise AssertionError("targets must be validated before auth")

    monkeypatch.setattr(tools, "require_verified_token", _no_token)

    with pytest.raises(McpError) as exc_info:
        asyncio.run(tools.generate_by_scenario(**_scenario_args()))

    assert exc_info.value.error.code == -32602
    assert "use_defaults" in exc_info.value.error.message


def test_scenario_with_use_defaults_reports_service_defaults(monkeypatch):
    calls = []

    async def _generate(**kwargs):
        calls.append(kwargs)
        return RESULT_FC

    monkeypatch.setattr(tools, "require_verified_token", _token)
    monkeypatch.setattr(orchestration, "generate_by_scenario", _generate)

    result = asyncio.run(
        tools.generate_by_scenario(**_scenario_args(use_defaults=True, preserve_existing_buildings=True))
    )

    assert calls[0]["preserve_existing_buildings"] is True
    assert calls[0]["targets_by_zone"]["residents"]["residential"] == 1100
    summary = result["summary"]
    assert summary["targets_source"] == "service_defaults"
    assert summary["existing_buildings_preserved"] is True
    assert summary["residents_total"] == 800
    assert summary["excluded_buildings"] == 1
    assert summary["targets"]["residential"]["residents_deficit"] == 300
    assert "features" not in result
    assert result["result_id"] == result["generation_id"]


def test_territory_requires_targets_or_use_defaults():
    with pytest.raises(McpError) as exc_info:
        asyncio.run(tools.generate_by_territory(blocks=BLOCKS))

    assert exc_info.value.error.code == -32602


def test_territory_with_explicit_targets_reports_request(monkeypatch):
    async def _generate(payload):
        return RESULT_FC

    monkeypatch.setattr(orchestration, "generate_by_territory", _generate)

    result = asyncio.run(
        tools.generate_by_territory(blocks=BLOCKS, targets_by_zone={"residents": {"residential": 800}})
    )

    summary = result["summary"]
    assert summary["targets_source"] == "request"
    assert summary["targets"]["residential"]["residents_deficit"] == 0
    assert "existing_buildings_preserved" not in summary
    assert result["service_diagnostics"] == SERVICE_DIAGNOSTICS


def test_blocks_summary_combines_per_zone_targets(monkeypatch):
    async def _generate(**kwargs):
        return RESULT_FC

    monkeypatch.setattr(tools, "require_verified_token", _token)
    monkeypatch.setattr(orchestration, "generate_by_blocks", _generate)

    result = asyncio.run(
        tools.generate_by_blocks(
            **_scenario_args(
                zones=[
                    {"functional_zone_id": 1, "targets_by_zone": {"residents": {"residential": 600}}},
                    {"functional_zone_id": 2, "targets_by_zone": {"residents": {"residential": 400}}},
                ]
            )
        )
    )

    summary = result["summary"]
    assert summary["targets"]["residential"]["target_residents"] == 1000
    assert summary["targets"]["residential"]["residents_deficit"] == 200
    assert summary["existing_buildings_preserved"] is False


def test_capacity_estimate_returns_zones_and_totals(monkeypatch):
    async def _estimate(**kwargs):
        base = dict(
            functional_zone_type="residential",
            existing_buildings_count=1,
            existing_living_area=100.25,
            existing_residents=5,
        )
        return {
            1: {**base, "functional_zone_id": 1, "zone_area_m2": 1000.1, "max_residents": 10, "max_living_area": 200.2},
            2: {**base, "functional_zone_id": 2, "zone_area_m2": 2000.2, "max_residents": 20, "max_living_area": 400.4},
        }

    monkeypatch.setattr(tools, "require_verified_token", _token)
    monkeypatch.setattr(orchestration, "estimate_capacity_by_blocks", _estimate)

    result = asyncio.run(
        tools.estimate_max_residents_by_blocks(**_scenario_args(functional_zone_ids=[1, 2]))
    )

    assert [z["functional_zone_id"] for z in result["zones"]] == [1, 2]
    assert result["totals"] == {
        "zone_area_m2": 3000.3,
        "max_residents": 30,
        "max_living_area": 600.6,
        "existing_buildings_count": 2,
        "existing_living_area": 200.5,
        "existing_residents": 10,
    }
    assert result["existing_buildings_preserved"] is False


def test_generation_stores_layer_and_returns_link_instead_of_geometry(monkeypatch, storage):
    _fake_scenario_generation(monkeypatch)

    result = asyncio.run(tools.generate_by_scenario(**_scenario_args(targets_by_zone=_TARGETS)))

    assert "features" not in result and "storage_warning" not in result
    result_id = result["result_id"]
    assert result_id == result["generation_id"] and len(result_id) == 32
    assert result["layer"]["url"].endswith(f"/files/buildings/{result_id}")
    stored = json.loads(b"".join(storage.open_stream(object_key(result_id, SLOT_BUILDINGS))))
    assert stored["features"] == RESULT_FC["features"]
    assert stored["summary"] == result["summary"]
    assert stored["seed"] == result["seed"]


def test_include_geometry_inlines_features(monkeypatch):
    _fake_scenario_generation(monkeypatch)

    result = asyncio.run(
        tools.generate_by_scenario(**_scenario_args(targets_by_zone=_TARGETS, include_geometry=True))
    )

    assert result["type"] == "FeatureCollection"
    assert result["features"] == RESULT_FC["features"]
    assert result["result_id"] is not None


class _BrokenStorage:
    def put_json(self, payload, object_key):
        raise ObjectStorageError("bucket is gone")


@pytest.mark.parametrize("unavailable", [None, _BrokenStorage()])
def test_features_are_inlined_when_storage_is_unavailable(monkeypatch, unavailable):
    monkeypatch.setattr(tools, "optional_object_storage", lambda: unavailable)
    _fake_scenario_generation(monkeypatch)

    result = asyncio.run(tools.generate_by_scenario(**_scenario_args(targets_by_zone=_TARGETS)))

    assert result["result_id"] is None and result["layer"] is None
    assert result["features"] == RESULT_FC["features"]
    assert result["storage_warning"]


def test_seed_and_applied_parameters_are_echoed(monkeypatch):
    calls = _fake_scenario_generation(monkeypatch)

    result = asyncio.run(
        tools.generate_by_scenario(
            **_scenario_args(
                targets_by_zone=_TARGETS, seed=42, generation_parameters={"rectangle_finder_step": 7}
            )
        )
    )

    assert calls[0]["generation_parameters"] == {"rectangle_finder_step": 7, "seed": 42}
    assert result["seed"] == 42
    applied = result["applied_parameters"]
    assert applied["generation_parameters"]["seed"] == 42
    assert applied["generation_parameters"]["rectangle_finder_step"] == 7
    assert "INNER_BORDER" in applied["generation_parameters"]
    assert "service_projects_file" not in applied["generation_parameters"]
    assert applied["targets_by_zone"]["residents"]["residential"] == 800
    assert result["duration_s"] >= 0


def test_seed_is_drawn_when_omitted(monkeypatch):
    calls = _fake_scenario_generation(monkeypatch)

    result = asyncio.run(tools.generate_by_scenario(**_scenario_args(targets_by_zone=_TARGETS)))

    assert isinstance(result["seed"], int)
    assert calls[0]["generation_parameters"]["seed"] == result["seed"]


def test_invalid_generation_parameters_fail_before_generation(monkeypatch):
    calls = _fake_scenario_generation(monkeypatch)

    with pytest.raises(McpError) as exc_info:
        asyncio.run(
            tools.generate_by_territory(
                blocks=BLOCKS, targets_by_zone=_TARGETS, generation_parameters={"rectangle_finder_step": "fine"}
            )
        )

    assert exc_info.value.error.code == -32602
    assert calls == []


class _FakeContext:
    def __init__(self):
        self.progress = []

    async def report_progress(self, progress, total=None, message=None):
        self.progress.append((progress, total, message))


def test_blocks_seed_every_zone_and_forward_progress(monkeypatch):
    calls = []

    async def _generate(**kwargs):
        calls.append(kwargs)
        await kwargs["progress"](1, 2)
        await kwargs["progress"](2, 2)
        return RESULT_FC

    monkeypatch.setattr(tools, "require_verified_token", _token)
    monkeypatch.setattr(orchestration, "generate_by_blocks", _generate)
    ctx = _FakeContext()

    result = asyncio.run(
        tools.generate_by_blocks(
            **_scenario_args(
                zones=[
                    {"functional_zone_id": 1, "targets_by_zone": _TARGETS},
                    {
                        "functional_zone_id": 2,
                        "targets_by_zone": _TARGETS,
                        "generation_parameters": {"rectangle_finder_step": 3},
                    },
                ],
                seed=7,
            ),
            ctx=ctx,
        )
    )

    zones = calls[0]["body"].zones
    assert [zone.generation_parameters for zone in zones] == [{"seed": 7}, {"rectangle_finder_step": 3, "seed": 7}]
    applied = result["applied_parameters"]["zones"]
    assert [zone["functional_zone_id"] for zone in applied] == [1, 2]
    assert applied[1]["generation_parameters"]["rectangle_finder_step"] == 3
    assert ctx.progress == [(1, 2, "zone 1/2"), (2, 2, "zone 2/2")]


def test_capacity_estimate_forwards_progress_only_with_context(monkeypatch):
    progress = []

    async def _estimate(**kwargs):
        progress.append(kwargs["progress"])
        return {}

    monkeypatch.setattr(tools, "require_verified_token", _token)
    monkeypatch.setattr(orchestration, "estimate_capacity_by_blocks", _estimate)

    asyncio.run(tools.estimate_max_residents_by_blocks(**_scenario_args(functional_zone_ids=[1])))
    result = asyncio.run(
        tools.estimate_max_residents_by_blocks(**_scenario_args(functional_zone_ids=[1]), ctx=_FakeContext())
    )

    assert progress[0] is None and callable(progress[1])
    assert result["duration_s"] >= 0


def test_get_generation_result_round_trip(monkeypatch):
    _fake_scenario_generation(monkeypatch)
    generated = asyncio.run(tools.generate_by_scenario(**_scenario_args(targets_by_zone=_TARGETS, seed=5)))

    full = asyncio.run(tools.get_generation_result(result_id=generated["result_id"]))
    meta = asyncio.run(tools.get_generation_result(result_id=generated["result_id"], include_geometry=False))

    assert full["features"] == RESULT_FC["features"]
    assert full["seed"] == 5 and full["summary"] == generated["summary"]
    assert "features" not in meta and meta["generation_id"] == generated["generation_id"]


@pytest.mark.parametrize("result_id", ["../../etc/passwd", "0" * 32])
def test_get_generation_result_rejects_bad_or_unknown_id(monkeypatch, result_id):
    monkeypatch.setattr(tools, "require_verified_token", _token)

    with pytest.raises(McpError) as exc_info:
        asyncio.run(tools.get_generation_result(result_id=result_id))

    assert exc_info.value.error.code == -32602
