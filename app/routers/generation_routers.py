from typing import Annotated, List, Optional
from fastapi import APIRouter, Body, Query, Depends
from loguru import logger

from app.dependencies import builder, urban_db_api, zones_service
from app.exceptions.http_exception_wrapper import http_exception
from app.logic.polygon_converter import _explode_to_polygons, _scale_numeric_targets, _make_block_feature, \
    _filter_parts_by_zone_min_area
from app.schema.default_params import DEFAULT_BLOCK_GENERATION_PARAMETERS, DEFAULT_BLOCK_TARGETS_BY_ZONE
from app.schema.dto import (
    ScenarioBody,
    TerritoryRequest,
    BuildingFeatureCollection,
    BlockFeatureCollection,
    FunctionalZonesRequest,
)
from app.utils import auth

generation_router = APIRouter()


@generation_router.post(
    "/generate/by_scenario",
    summary="Generate buildings for target scenario",
    response_model=BuildingFeatureCollection,
)
async def pipeline_route(
    scenario_id: Annotated[int, Query(..., description="Scenario ID", examples=[198])],
    year: Annotated[int, Query(..., description="Data year", examples=[2024])],
    source: Annotated[str, Query(..., description="Data source", examples=["OSM"])],
    functional_zone_types: Annotated[
        List[str],
        Query(..., description="Target functional zone types", examples=[['residential', 'business', 'industrial']]),
    ],
    physical_object_id: Annotated[
        Optional[List[int]],
        Query(
            description=(
                    "Physical object id(s) to exclude from generation territory."
            ),
            examples=[[2058130, 2058131]],
        ),
    ] = None,
    token: str = Depends(auth.verify_token),
    body: ScenarioBody = Body(default_factory=ScenarioBody),
):
    return await builder.run(
        scenario_id=scenario_id,
        year=year,
        source=source,
        token=token,
        functional_zone_types=functional_zone_types,
        targets_by_zone=body.targets_by_zone,
        generation_parameters_override=body.generation_parameters,
        physical_object_ids=physical_object_id,
    )


@generation_router.post(
    "/generate/by_territory", summary="Generate buildings for target territories",
    response_model=BuildingFeatureCollection)
async def pipeline_route(
        payload: TerritoryRequest = Body(
            ...,
            description="Body for request"
        )
):
    return await builder.run(
        blocks=payload.blocks,
        targets_by_zone=payload.targets_by_zone,
        generation_parameters_override=payload.generation_parameters
    )


@generation_router.post(
    "/generate/by_blocks", summary="Generate buildings for target blocks", response_model=BuildingFeatureCollection
)
async def generate_by_functional_zones(
        scenario_id: Annotated[int, Query(..., description="Scenario ID", examples=[198])],
        year: Annotated[int, Query(..., description="Data year", examples=[2024])],
        source: Annotated[str, Query(..., description="Data source", examples=["OSM"])],
        functional_zone_types: Annotated[
            List[str],
            Query(
                ...,
                description="Target functional zone types",
                examples=["residential", "business", "industrial"],
            ),
        ],
        physical_object_id: Annotated[
                Optional[List[int]],
                Query(
                    description=(
                            "Physical object id(s) to exclude from generation territory."
                    ),
                    examples=[[2058130, 2058131]],
                ),
            ] = None,
        token: str = Depends(auth.verify_token),
        body: FunctionalZonesRequest = Body(
            ..., description="Per-zone targets and generation parameters"
        ),
):
    response_json = await urban_db_api.get_scenario_functional_zones(
        scenario_id=scenario_id,
        source=source,
        year=year,
        token=token,
    )
    features = response_json.get("features", [])
    if not features:
        raise http_exception(
            404, f"No functional zones found for scenario {scenario_id}"
        )

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
            404,
            "Functional zones not found for provided ids",
            input_data={"missing_ids": missing_ids},
        )

    excluded_features_out = []
    if physical_object_id:
        ids_set = {int(x) for x in physical_object_id}

        try:
            physical_objects_fc = await urban_db_api.get_physical_objects(
                scenario_id=scenario_id,
                token=token,
            )
            selected = builder.physical_objects_service.select_features_by_ids(
                physical_objects_fc,
                ids_set,
            )
        except Exception as e:
            logger.warning("Failed to fetch excluded physical objects for response: {}", e)
            selected = []

        for feat in selected:
            props = feat.get("properties") or {}

            po_id = (
                    props.get("physical_object_id")
                    or props.get("excluded_physical_object_id")
                    or props.get("id")
            )

            excluded_features_out.append(
                {
                    "type": "Feature",
                    "geometry": feat.get("geometry"),
                    "properties": {
                        "is_excluded": True,
                        "physical_object_id": po_id,
                    },
                }
            )

    combined_features = []

    for zone in body.zones:
        feature = feature_by_id[zone.functional_zone_id]
        props = feature.get("properties", {})
        zone_type = (props.get("functional_zone_type") or {}).get("name")

        geometry = feature.get("geometry")
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

            blocks = BlockFeatureCollection.model_validate(
                {"type": "FeatureCollection", "features": [block_feature]}
            )

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

            combined_features.extend(result.get("features", []))

    combined_features.extend(excluded_features_out)
    return {"type": "FeatureCollection", "features": combined_features}


@generation_router.post(
    "/generate/max_residents_by_blocks",
    summary="Estimate max residents for target blocks",
)
async def residents_by_functional_zones(
    scenario_id: Annotated[int, Query(..., description="Scenario ID", examples=[198])],
    year: Annotated[int, Query(..., description="Data year", examples=[2024])],
    source: Annotated[str, Query(..., description="Data source", examples=["OSM"])],
    functional_zone_types: Annotated[
        list[str],
        Query(..., description="Target functional zone types", examples=["residential", "business"]),
    ],
    functional_zone_ids: Annotated[
        Optional[List[int]],
        Query(
            description=(
                    "Physical object id(s) to exclude from generation territory."
            ),
            examples=[[2058130, 2058131]],
        )
    ],
    token: str = Depends(auth.verify_token),
):

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
        for feature_out in result.get("features", []):
            value = (feature_out.get("properties") or {}).get("residents_number")
            if value is not None:
                residents_sum += int(value)

        residents_by_block[block_id] = residents_sum

    return residents_by_block
