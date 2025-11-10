from typing import Annotated, List
from fastapi import APIRouter, Body, Query

from app.dependencies import builder
from app.schema.dto import PIPELINE_EXAMPLE, ScenarioBody, TerritoryRequest, BuildingFeatureCollection

generation_router = APIRouter()


@generation_router.post("/generate/by_scenario", summary="Generate buildings for target scenario", response_model=BuildingFeatureCollection)
async def pipeline_route(
    scenario_id: Annotated[int,  Query(..., description="Scenario ID", example=198)],
    year:       Annotated[int,  Query(..., description="Data year", example=2024)],
    source:     Annotated[str,  Query(..., description="Data source", example="OSM")],
    functional_zone_types: Annotated[List[str], Query(
        ..., description="Target functional zone types"
    , example=['residential', 'business', 'industrial'])],
    body: ScenarioBody = Body(
        default_factory=ScenarioBody,
        description="Targets and hyperparameters as JSON (optional)"
    )
    ):
    return await builder.run(
        scenario_id=scenario_id,
        year=year,
        source=source,
        functional_zone_types=functional_zone_types,
        targets_by_zone=body.targets_by_zone,
        infer_params=body.params,
    )


@generation_router.post(
    "/generate/by_territory", summary="Generate buildings for target territories", response_model=BuildingFeatureCollection)
async def pipeline_route(
    payload: TerritoryRequest = Body(
        ...,
        description="Body for request",
        examples={
            "demo": {
                "summary": "Request example",
                "value": PIPELINE_EXAMPLE,
            }
        },
    )
    ):
    return await builder.run(
        blocks=payload.blocks,
        targets_by_zone=payload.targets_by_zone,
        infer_params=payload.params,
    )
