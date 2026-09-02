"""Building-generation orchestration, shared by the REST routers, MCP tools and
the A2A agent executor.

Each public function here corresponds 1:1 to a ``/generate/*`` REST endpoint
in :mod:`app.routers.generation_routers`, with the FastAPI-specific parameter
parsing (``Query``/``Body``/``Depends``) stripped out so the same logic can be
called from a non-HTTP entry point (an MCP tool, an A2A task) without going
through the ASGI stack.
"""
from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from app.dependencies import (
    build_facade_jobs_client,
    builder,
    facade_jobs_configured,
    urban_db_api,
    zones_service,
)
from app.exceptions.http_exception_wrapper import http_exception
from app.infrastructure.facade_jobs_client import FacadeJobsError
from app.logic.polygon_converter import (
    _explode_to_polygons,
    _scale_numeric_targets,
    _make_block_feature,
    _filter_parts_by_zone_min_area,
)
from app.schema.default_params import DEFAULT_BLOCK_GENERATION_PARAMETERS, DEFAULT_BLOCK_TARGETS_BY_ZONE
from app.schema.dto import BlockFeatureCollection, FunctionalZonesRequest, TerritoryRequest


def _empty_feature_collection() -> dict:
    """Return an empty GeoJSON FeatureCollection."""
    return {"type": "FeatureCollection", "features": []}


def _get_generated_buildings(result: dict | None) -> dict:
    """Extract generated buildings FeatureCollection from builder result."""
    if not isinstance(result, dict):
        return _empty_feature_collection()

    generated = result.get("generated_buildings")
    if isinstance(generated, dict) and generated.get("type") == "FeatureCollection":
        return {
            "type": "FeatureCollection",
            "features": list(generated.get("features") or []),
        }

    if result.get("type") == "FeatureCollection":
        return {
            "type": "FeatureCollection",
            "features": list(result.get("features") or []),
        }

    return _empty_feature_collection()


def _get_selected_features(result: dict | None) -> dict:
    """Extract selected physical objects FeatureCollection from builder result."""
    if not isinstance(result, dict):
        return _empty_feature_collection()

    selected = result.get("selected_features")
    if isinstance(selected, dict) and selected.get("type") == "FeatureCollection":
        return {
            "type": "FeatureCollection",
            "features": list(selected.get("features") or []),
        }

    return _empty_feature_collection()


def _build_excluded_features(selected_fc: dict | None) -> list[dict]:
    """Return excluded features as-is, preserving all prepared properties."""
    features = (selected_fc or {}).get("features") or []
    return list(features)


def merge_generation_result(result: dict | None) -> dict:
    """Merge generated buildings with excluded physical objects."""
    generated_fc = _get_generated_buildings(result)
    selected_fc = _get_selected_features(result)

    return {
        "type": "FeatureCollection",
        "features": [
            *(generated_fc.get("features") or []),
            *_build_excluded_features(selected_fc),
        ],
    }


async def generate_by_scenario(
    *,
    scenario_id: int,
    year: int,
    source: str,
    functional_zone_types: list[str],
    physical_object_id: Optional[list[int]],
    token: str,
    targets_by_zone: Optional[dict[str, dict[str, Any]]],
    generation_parameters: Optional[dict[str, Any]],
) -> dict:
    """Generate buildings for a scenario's territory. Mirrors ``/generate/by_scenario``."""
    result = await builder.run(
        scenario_id=scenario_id,
        year=year,
        source=source,
        token=token,
        functional_zone_types=functional_zone_types,
        targets_by_zone=targets_by_zone,
        generation_parameters_override=generation_parameters,
        physical_object_ids=physical_object_id,
    )
    return merge_generation_result(result)


async def generate_by_territory(payload: TerritoryRequest) -> dict:
    """Generate buildings for an arbitrary set of block polygons. Mirrors ``/generate/by_territory``.

    This is the project-less mode: there is no scenario to pull existing
    buildings from, so the caller may upload them (``existing_buildings``) —
    their footprints are cut out of the blocks and they come back in the
    response marked ``is_excluded``.
    """
    result = await builder.run(
        blocks=payload.blocks,
        targets_by_zone=payload.targets_by_zone,
        generation_parameters_override=payload.generation_parameters,
        existing_buildings=(
            payload.existing_buildings.model_dump()
            if payload.existing_buildings is not None
            else None
        ),
    )
    return merge_generation_result(result)


async def generate_by_blocks(
    *,
    scenario_id: int,
    year: int,
    source: str,
    functional_zone_types: list[str],
    physical_object_id: Optional[list[int]],
    token: str,
    body: FunctionalZonesRequest,
) -> dict:
    """Generate buildings for specific functional zones of a scenario, one block per
    zone (or per polygon part for a MultiPolygon zone). Mirrors ``/generate/by_blocks``."""
    response_json = await urban_db_api.get_scenario_functional_zones(
        scenario_id=scenario_id,
        source=source,
        year=year,
        token=token,
    )
    features = response_json.get("features", [])
    if not features:
        raise http_exception(404, f"No functional zones found for scenario {scenario_id}")

    filtered = []
    for feature in features:
        props = feature.get("properties", {})
        zone_type = (props.get("functional_zone_type") or {}).get("name")
        if zone_type in functional_zone_types:
            filtered.append(feature)

    feature_by_id = {}
    for feature in filtered:
        props = feature.get("properties", {})
        zone_id = props.get("functional_zone_id")
        if zone_id is not None:
            feature_by_id[int(zone_id)] = feature

    requested_ids = [zone.functional_zone_id for zone in body.zones]
    missing_ids = [zone_id for zone_id in requested_ids if zone_id not in feature_by_id]
    if missing_ids:
        raise http_exception(
            422,
            "Functional zones not found for provided ids",
            input_data={"missing_ids": missing_ids},
        )

    combined_features = []
    selected_features_fc = _empty_feature_collection()

    for zone in body.zones:
        feature = feature_by_id[zone.functional_zone_id]
        props = feature.get("properties", {})
        zone_type = (props.get("functional_zone_type") or {}).get("name")

        geometry = feature.get("geometry") or {}
        geom_type = geometry.get("type")

        if geom_type == "Polygon":
            block_feature = {
                "type": "Feature",
                "properties": {**props, "block_id": props.get("functional_zone_id"), "zone": zone_type},
                "geometry": geometry,
            }
            blocks = BlockFeatureCollection.model_validate({"type": "FeatureCollection", "features": [block_feature]})
            result = await builder.run(
                blocks=blocks,
                targets_by_zone=zone.targets_by_zone,
                generation_parameters_override=zone.generation_parameters,
                scenario_id=scenario_id,
                token=token,
                year=year,
                source=source,
                functional_zone_types=functional_zone_types,
                physical_object_ids=physical_object_id,
            )
            combined_features.extend(_get_generated_buildings(result).get("features", []))
            if not selected_features_fc.get("features"):
                selected_features_fc = _get_selected_features(result)
            continue

        if geom_type == "MultiPolygon":
            parts = _explode_to_polygons(geometry, min_area_weight=0.0)

            parts, report = _filter_parts_by_zone_min_area(
                zone_id=zone.functional_zone_id,
                zone_type=zone_type,
                parts=parts,
            )

            if report is not None and report.dropped_count > 0:
                logger.info(
                    "Zone {} ({}): dropped={} kept={}",
                    report.zone_id,
                    report.zone_type,
                    report.dropped_count,
                    report.kept_count,
                )

            if not parts:
                logger.warning("No polygon parts after filtering for zone_id={}", zone.functional_zone_id)
                continue

            for part in parts:
                part_targets = _scale_numeric_targets(zone.targets_by_zone, part.area_weight)
                block_feature = _make_block_feature(
                    base_props=props,
                    zone_type=zone_type,
                    zone_id=zone.functional_zone_id,
                    part_index=part.index,
                    geometry=part.geometry,
                )
                blocks = BlockFeatureCollection.model_validate({"type": "FeatureCollection", "features": [block_feature]})
                result = await builder.run(
                    blocks=blocks,
                    targets_by_zone=part_targets,
                    generation_parameters_override=zone.generation_parameters,
                    scenario_id=scenario_id,
                    token=token,
                    year=year,
                    source=source,
                    functional_zone_types=functional_zone_types,
                    physical_object_ids=physical_object_id,
                )
                combined_features.extend(_get_generated_buildings(result).get("features", []))
            if not selected_features_fc.get("features"):
                selected_features_fc = _get_selected_features(result)
            continue

        raise http_exception(422, f"Unsupported geometry type for zone {zone.functional_zone_id}: {geom_type}")

    combined_features.extend(_build_excluded_features(selected_features_fc))
    return {"type": "FeatureCollection", "features": combined_features}


def _require_facade_jobs() -> None:
    """Fail before running the comparatively expensive building generator."""
    if not facade_jobs_configured():
        raise http_exception(
            503,
            "3D facade generation is unavailable: facade-jobs is not "
            "configured (set FACADE_JOBS_API).",
        )


async def submit_facade_job(
    buildings: dict[str, Any],
    *,
    requested_by: str | None,
) -> dict[str, str]:
    """Submit generated buildings to ``facade-jobs`` without waiting for GLB."""
    _require_facade_jobs()

    try:
        async with build_facade_jobs_client() as client:
            job = await client.submit_job(buildings, requested_by=requested_by)
    except FacadeJobsError as exc:
        if exc.status_code == 0 or exc.status_code == 503:
            status_code = 503
        elif exc.status_code >= 500:
            status_code = 502
        else:
            status_code = exc.status_code
        raise http_exception(
            status_code,
            "Could not queue 3D facade generation.",
            detail=exc.body,
        ) from exc
    except (RuntimeError, ValueError) as exc:
        raise http_exception(503, str(exc)) from exc

    return {
        "job_id": job["job_id"],
        "status_url": job["status_url"],
    }


async def generate_3d_by_scenario(
    *,
    scenario_id: int,
    year: int,
    source: str,
    functional_zone_types: list[str],
    physical_object_id: Optional[list[int]],
    token: str,
    requested_by: str | None,
    targets_by_zone: Optional[dict[str, dict[str, Any]]],
    generation_parameters: Optional[dict[str, Any]],
) -> dict[str, str]:
    _require_facade_jobs()
    buildings = await generate_by_scenario(
        scenario_id=scenario_id,
        year=year,
        source=source,
        functional_zone_types=functional_zone_types,
        physical_object_id=physical_object_id,
        token=token,
        targets_by_zone=targets_by_zone,
        generation_parameters=generation_parameters,
    )
    return await submit_facade_job(buildings, requested_by=requested_by)


async def generate_3d_by_territory(
    payload: TerritoryRequest,
    *,
    requested_by: str | None = None,
) -> dict[str, str]:
    _require_facade_jobs()
    buildings = await generate_by_territory(payload)
    return await submit_facade_job(buildings, requested_by=requested_by)


async def generate_3d_by_blocks(
    *,
    scenario_id: int,
    year: int,
    source: str,
    functional_zone_types: list[str],
    physical_object_id: Optional[list[int]],
    token: str,
    requested_by: str | None,
    body: FunctionalZonesRequest,
) -> dict[str, str]:
    _require_facade_jobs()
    buildings = await generate_by_blocks(
        scenario_id=scenario_id,
        year=year,
        source=source,
        functional_zone_types=functional_zone_types,
        physical_object_id=physical_object_id,
        token=token,
        body=body,
    )
    return await submit_facade_job(buildings, requested_by=requested_by)


async def estimate_max_residents_by_blocks(
    *,
    scenario_id: int,
    year: int,
    source: str,
    functional_zone_types: list[str],
    functional_zone_ids: list[int],
    token: str,
) -> dict[int, int]:
    """Estimate max residents per functional zone. Mirrors ``/generate/max_residents_by_blocks``."""
    blocks_by_zone = await zones_service.prepare_blocks_by_zones(
        scenario_id=scenario_id,
        year=year,
        source=source,
        token=token,
        functional_zone_types=functional_zone_types,
        zone_ids=functional_zone_ids,
    )

    residents_by_block: dict[int, int] = {}

    for zid in functional_zone_ids:
        block_id = int(zid)
        blocks = blocks_by_zone.get(block_id)

        result = await builder.run(
            blocks=blocks,
            targets_by_zone=DEFAULT_BLOCK_TARGETS_BY_ZONE,
            generation_parameters_override=DEFAULT_BLOCK_GENERATION_PARAMETERS,
            scenario_id=scenario_id,
            token=token,
            year=year,
            source=source,
            functional_zone_types=functional_zone_types,
        )

        residents_sum = 0
        for feature_out in _get_generated_buildings(result).get("features", []):
            value = (feature_out.get("properties") or {}).get("residents_number")
            if value is not None:
                residents_sum += int(value)

        residents_by_block[block_id] = residents_sum

    return residents_by_block


__all__ = [
    "merge_generation_result",
    "generate_by_scenario",
    "generate_by_territory",
    "generate_by_blocks",
    "submit_facade_job",
    "generate_3d_by_scenario",
    "generate_3d_by_territory",
    "generate_3d_by_blocks",
    "estimate_max_residents_by_blocks",
]
