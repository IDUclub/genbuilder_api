# app/routers/generation_routers.py
from fastapi import APIRouter, Body, HTTPException

from app.logic.generation import builder
from app.schema.dto import PIPELINE_EXAMPLE, ScenarioRequest, TerritoryRequest

generation_router = APIRouter()


@generation_router.post(
    f"/generate/by_scenario", summary="Generate buildings for target scenario"
)
async def pipeline_route(
    payload: ScenarioRequest = Body(
        ...,
        description="Body for request",
    )
):
    try:
        return await builder.run(
            scenario_id=payload.scenario_id,
            functional_zone_types=payload.functional_zone_types,
            targets_by_zone=payload.targets_by_zone,
            infer_params=payload.params,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")


@generation_router.post(
    "/generate/by_territory", summary="Generate buildings for target territories"
)
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
    try:
        return await builder.run(
            blocks=payload.blocks,
            targets_by_zone=payload.targets_by_zone,
            infer_params=payload.params,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")
