"""Building-generation orchestration, shared by the REST routers, MCP tools and
the A2A agent executor.

Each public function here corresponds 1:1 to a ``/generate/*`` REST endpoint
in :mod:`app.routers.generation_routers`, with the FastAPI-specific parameter
parsing (``Query``/``Body``/``Depends``) stripped out so the same logic can be
called from a non-HTTP entry point (an MCP tool, an A2A task) without going
through the ASGI stack.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import geopandas as gpd
from fastapi import HTTPException
from loguru import logger
from shapely.geometry import shape
from shapely.ops import unary_union

from app.dependencies import builder, physical_objects_service, urban_db_api, zones_service
from app.exceptions.http_exception_wrapper import http_exception
from app.logic.polygon_converter import (
    _explode_to_polygons,
    _scale_numeric_targets,
    _make_block_feature,
    _filter_parts_by_zone_min_area,
)
from app.logic.zone_taxonomy import normalize_zone
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


def _zone_type_name(feature: dict) -> Optional[str]:
    return ((feature.get("properties") or {}).get("functional_zone_type") or {}).get("name")


async def _areas_m2(geometries: list[dict]) -> list[float]:
    """Areas (m²) of WGS84 GeoJSON geometries, measured in a local UTM CRS."""
    if not geometries:
        return []

    def _compute() -> list[float]:
        series = gpd.GeoSeries([shape(g) for g in geometries], crs="EPSG:4326")
        projected = series.to_crs(series.estimate_utm_crs())
        return [round(float(area), 1) for area in projected.area]

    return await asyncio.to_thread(_compute)


async def load_existing_buildings(*, scenario_id: int, token: str) -> list[dict]:
    """Load every existing building of a scenario, normalized as excluded features.

    Fails loudly: if UrbanDB can't be reached the caller gets an error instead
    of a layout silently generated on top of buildings it asked to keep.
    """
    try:
        fc = await urban_db_api.get_physical_objects(scenario_id=scenario_id, token=token)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to load physical objects for scenario {}", scenario_id)
        raise http_exception(
            502,
            f"Failed to load existing buildings for scenario {scenario_id}",
            detail=str(exc),
        ) from exc
    return physical_objects_service.select_building_features(fc)


def _buildings_within(buildings: list[dict], geometries: list[dict]) -> list[dict]:
    """Keep buildings whose footprint intersects any of the given geometries."""
    if not buildings or not geometries:
        return []
    area = unary_union([shape(geometry) for geometry in geometries])
    kept = []
    for feature in buildings:
        try:
            if shape(feature["geometry"]).intersects(area):
                kept.append(feature)
        except Exception:  # noqa: BLE001 - one broken footprint must not fail the request
            logger.warning("Skipping building with invalid geometry: {}", feature.get("id"))
    return kept


def _existing_buildings_payload(
    buildings: list[dict],
    physical_object_id: Optional[list[int]],
) -> tuple[Optional[dict], Optional[list[int]]]:
    """Build the ``existing_buildings`` FC for ``builder.run`` and drop the ids it
    already covers from ``physical_object_id``, so no object is excluded (and
    returned) twice."""
    if not buildings:
        return None, physical_object_id
    covered = {(feature.get("properties") or {}).get("physical_object_id") for feature in buildings}
    remaining = [pid for pid in (physical_object_id or []) if pid not in covered] or None
    return {"type": "FeatureCollection", "features": buildings}, remaining


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
    preserve_existing_buildings: bool = False,
) -> dict:
    """Generate buildings for a scenario's territory. Mirrors ``/generate/by_scenario``.

    With ``preserve_existing_buildings`` every existing building of the scenario
    is cut out of the territory and returned marked ``is_excluded``.
    """
    existing_buildings = None
    if preserve_existing_buildings:
        buildings = await load_existing_buildings(scenario_id=scenario_id, token=token)
        existing_buildings, physical_object_id = _existing_buildings_payload(buildings, physical_object_id)

    result = await builder.run(
        scenario_id=scenario_id,
        year=year,
        source=source,
        token=token,
        functional_zone_types=functional_zone_types,
        targets_by_zone=targets_by_zone,
        generation_parameters_override=generation_parameters,
        physical_object_ids=physical_object_id,
        existing_buildings=existing_buildings,
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
    preserve_existing_buildings: bool = False,
) -> dict:
    """Generate buildings for specific functional zones of a scenario, one block per
    zone (or per polygon part for a MultiPolygon zone). Mirrors ``/generate/by_blocks``.

    With ``preserve_existing_buildings`` the scenario's existing buildings that
    intersect the requested zones are cut out and returned marked ``is_excluded``.
    """
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

    existing_buildings = None
    if preserve_existing_buildings:
        buildings = await load_existing_buildings(scenario_id=scenario_id, token=token)
        buildings = _buildings_within(
            buildings,
            [feature_by_id[zone_id].get("geometry") for zone_id in requested_ids],
        )
        existing_buildings, physical_object_id = _existing_buildings_payload(buildings, physical_object_id)

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
                existing_buildings=existing_buildings,
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
                    existing_buildings=existing_buildings,
                )
                combined_features.extend(_get_generated_buildings(result).get("features", []))
            if not selected_features_fc.get("features"):
                selected_features_fc = _get_selected_features(result)
            continue

        raise http_exception(422, f"Unsupported geometry type for zone {zone.functional_zone_id}: {geom_type}")

    combined_features.extend(_build_excluded_features(selected_features_fc))
    return {"type": "FeatureCollection", "features": combined_features}


async def list_functional_zones(
    *,
    scenario_id: int,
    year: int,
    source: str,
    token: str,
    functional_zone_types: Optional[list[str]] = None,
) -> dict:
    """List a scenario's functional zones — id, type and area — so a caller can
    pick ``functional_zone_ids`` for block generation or capacity estimates."""
    response_json = await urban_db_api.get_scenario_functional_zones(
        scenario_id=scenario_id,
        source=source,
        year=year,
        token=token,
    )
    features = [
        feature
        for feature in response_json.get("features") or []
        if (feature.get("properties") or {}).get("functional_zone_id") is not None
        and (feature.get("geometry") or {}).get("type") in {"Polygon", "MultiPolygon"}
        and (not functional_zone_types or _zone_type_name(feature) in functional_zone_types)
    ]
    areas = await _areas_m2([feature["geometry"] for feature in features])

    zones = []
    totals_by_type: dict[str, dict[str, Any]] = {}
    for feature, area in zip(features, areas):
        props = feature.get("properties") or {}
        zone_type = _zone_type_name(feature)
        zones.append(
            {
                "functional_zone_id": int(props["functional_zone_id"]),
                "functional_zone_type": zone_type,
                "generation_zone": normalize_zone(zone_type) or None,
                "name": props.get("name"),
                "geometry_type": feature["geometry"]["type"],
                "area_m2": area,
            }
        )
        totals = totals_by_type.setdefault(zone_type or "unknown", {"count": 0, "area_m2": 0.0})
        totals["count"] += 1
        totals["area_m2"] = round(totals["area_m2"] + area, 1)

    return {
        "scenario_id": scenario_id,
        "year": year,
        "source": source,
        "zones": zones,
        "totals_by_type": totals_by_type,
    }


async def estimate_capacity_by_blocks(
    *,
    scenario_id: int,
    year: int,
    source: str,
    functional_zone_types: list[str],
    functional_zone_ids: list[int],
    token: str,
    preserve_existing_buildings: bool = False,
) -> dict[int, dict[str, Any]]:
    """Estimate per-zone capacity at the service's maximum-density targets.

    Each zone reports its area, max residents and max living area, plus the
    existing buildings standing inside it. With ``preserve_existing_buildings``
    those buildings are cut out first, so the estimate is the *additional*
    capacity of the remaining land.
    """
    blocks_by_zone = await zones_service.prepare_blocks_by_zones(
        scenario_id=scenario_id,
        year=year,
        source=source,
        token=token,
        functional_zone_types=functional_zone_types,
        zone_ids=functional_zone_ids,
    )
    buildings = await load_existing_buildings(scenario_id=scenario_id, token=token)

    zone_ids = [int(zid) for zid in functional_zone_ids]
    geometries = {
        zid: [feature.geometry.model_dump() for feature in blocks_by_zone[zid].features]
        for zid in zone_ids
    }
    areas = await _areas_m2([geometries[zid][0] for zid in zone_ids])

    estimates: dict[int, dict[str, Any]] = {}
    for zid, zone_area in zip(zone_ids, areas):
        blocks = blocks_by_zone[zid]
        in_zone = _buildings_within(buildings, geometries[zid])

        result = await builder.run(
            blocks=blocks,
            targets_by_zone=DEFAULT_BLOCK_TARGETS_BY_ZONE,
            generation_parameters_override=DEFAULT_BLOCK_GENERATION_PARAMETERS,
            scenario_id=scenario_id,
            token=token,
            year=year,
            source=source,
            functional_zone_types=functional_zone_types,
            existing_buildings=(
                {"type": "FeatureCollection", "features": in_zone}
                if preserve_existing_buildings and in_zone
                else None
            ),
        )

        max_residents = 0
        max_living_area = 0.0
        for feature_out in _get_generated_buildings(result).get("features", []):
            props = feature_out.get("properties") or {}
            max_residents += int(props.get("residents_number") or 0)
            max_living_area += float(props.get("living_area") or 0.0)

        estimates[zid] = {
            "functional_zone_id": zid,
            "functional_zone_type": blocks.features[0].properties.zone,
            "zone_area_m2": zone_area,
            "max_residents": max_residents,
            "max_living_area": round(max_living_area, 1),
            "existing_buildings_count": len(in_zone),
            "existing_living_area": round(
                sum(float(f["properties"].get("living_area") or 0.0) for f in in_zone), 1
            ),
            "existing_residents": int(
                round(sum(float(f["properties"].get("residents_number") or 0.0) for f in in_zone))
            ),
        }

    return estimates


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
    "list_functional_zones",
    "load_existing_buildings",
    "estimate_capacity_by_blocks",
    "estimate_max_residents_by_blocks",
]
