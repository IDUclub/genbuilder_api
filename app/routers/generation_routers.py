from typing import Annotated, List
from fastapi import APIRouter, Body, Query, Depends
from app.dependencies import builder, urban_db_api
from app.exceptions.http_exception_wrapper import http_exception
from app.schema.dto import (
    ScenarioBody,
    TerritoryRequest,
    BuildingFeatureCollection,
    FunctionalZonesRequest,
    BlockFeatureCollection,
)
from app.utils import auth

generation_router = APIRouter()


@generation_router.post("/generate/by_scenario", summary="Generate buildings for target scenario", response_model=BuildingFeatureCollection)
async def pipeline_route(
    scenario_id: Annotated[int,  Query(..., description="Scenario ID", example=198)],
    year:       Annotated[int,  Query(..., description="Data year", example=2024)],
    source:     Annotated[str,  Query(..., description="Data source", example="OSM")],
    functional_zone_types: Annotated[List[str], Query(
        ..., description="Target functional zone types"
    , example=['residential', 'business', 'industrial'])],
    token: str = Depends(auth.verify_token),
    body: ScenarioBody = Body(
        default_factory=ScenarioBody,
        description="Targets and hyperparameters as JSON (optional)"
    )
    ):
    return await builder.run(
        scenario_id=scenario_id,
        year=year,
        source=source,
        token=token,
        functional_zone_types=functional_zone_types,
        targets_by_zone=body.targets_by_zone,
        generation_parameters_override=body.generation_parameters
    )


@generation_router.post(
    "/generate/by_territory", summary="Generate buildings for target territories", response_model=BuildingFeatureCollection)
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
    scenario_id: Annotated[int, Query(..., description="Scenario ID", example=198)],
    year: Annotated[int, Query(..., description="Data year", example=2024)],
    source: Annotated[str, Query(..., description="Data source", example="OSM")],
    functional_zone_types: Annotated[
        List[str],
        Query(
            ...,
            description="Target functional zone types",
            example=["residential", "business", "industrial"],
        ),
    ],
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

    combined_features = []
    for zone in body.zones:
        feature = feature_by_id[zone.functional_zone_id]
        props = feature.get("properties", {})
        zone_type = (props.get("functional_zone_type") or {}).get("name")
        block_feature = {
            "type": "Feature",
            "properties": {
                **props,
                "block_id": props.get("functional_zone_id"),
                "zone": zone_type,
            },
            "geometry": feature.get("geometry"),
        }
        blocks = BlockFeatureCollection.model_validate(
            {"type": "FeatureCollection", "features": [block_feature]}
        )
        result = await builder.run(
            blocks=blocks,
            targets_by_zone=zone.targets_by_zone,
            generation_parameters_override=zone.generation_parameters,
        )
        combined_features.extend(result.get("features", []))

    return {"type": "FeatureCollection", "features": combined_features}