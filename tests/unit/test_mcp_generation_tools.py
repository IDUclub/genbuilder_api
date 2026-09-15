"""MCP generation tools: no silent default targets, and every result carries a
``summary``."""
import asyncio

import pytest
from mcp import McpError

from app.logic import generation_orchestration as orchestration
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

RESULT_FC = {
    "type": "FeatureCollection",
    "features": [
        {"type": "Feature", "properties": {"zone": "residential", "residents_number": 800}, "geometry": None},
        {"type": "Feature", "properties": {"is_excluded": True, "residents_number": 50}, "geometry": None},
    ],
}


async def _token():
    return "token"


def _scenario_args(**overrides):
    kwargs = dict(scenario_id=1, year=2024, source="OSM", functional_zone_types=["residential"])
    kwargs.update(overrides)
    return kwargs


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
    assert result["features"] == RESULT_FC["features"]


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
