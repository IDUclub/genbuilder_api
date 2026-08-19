from typing import Annotated, List, Optional
from fastapi import APIRouter, Body, Query, Depends

from app.logic import generation_orchestration as orchestration
from app.schema.building_properties import (
    BUILDING_PROPERTIES_SCHEMA,
    BuildingPropertiesSchema,
)
from app.schema.dto import (
    ScenarioBody,
    TerritoryRequest,
    BuildingFeatureCollection,
    FacadeJobAccepted,
    FunctionalZonesRequest,
)
from app.utils import auth

generation_router = APIRouter()


@generation_router.get(
    "/generate/properties_schema",
    summary="Display labels for generated building properties",
    response_model=BuildingPropertiesSchema,
)
async def properties_schema() -> BuildingPropertiesSchema:
    return BUILDING_PROPERTIES_SCHEMA


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
    return await orchestration.generate_by_scenario(
        scenario_id=scenario_id,
        year=year,
        source=source,
        functional_zone_types=functional_zone_types,
        physical_object_id=physical_object_id,
        token=token,
        targets_by_zone=body.targets_by_zone,
        generation_parameters=body.generation_parameters,
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
    return await orchestration.generate_by_territory(payload)


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
    return await orchestration.generate_by_blocks(
        scenario_id=scenario_id,
        year=year,
        source=source,
        functional_zone_types=functional_zone_types,
        physical_object_id=physical_object_id,
        token=token,
        body=body,
    )


@generation_router.post(
    "/generate/3d/by_scenario",
    summary="Generate buildings and queue a 3D facade job",
    response_model=FacadeJobAccepted,
    status_code=202,
)
async def generate_3d_by_scenario(
    scenario_id: Annotated[int, Query(..., description="Scenario ID", examples=[198])],
    year: Annotated[int, Query(..., description="Data year", examples=[2024])],
    source: Annotated[str, Query(..., description="Data source", examples=["OSM"])],
    functional_zone_types: Annotated[
        List[str],
        Query(
            ...,
            description="Target functional zone types",
            examples=[["residential", "business", "industrial"]],
        ),
    ],
    physical_object_id: Annotated[
        Optional[List[int]],
        Query(
            description="Physical object id(s) to exclude from generation territory.",
            examples=[[2058130, 2058131]],
        ),
    ] = None,
    user: auth.AuthUser = Depends(auth.get_current_user),
    body: ScenarioBody = Body(default_factory=ScenarioBody),
) -> dict[str, str]:
    return await orchestration.generate_3d_by_scenario(
        scenario_id=scenario_id,
        year=year,
        source=source,
        functional_zone_types=functional_zone_types,
        physical_object_id=physical_object_id,
        token=user.token,
        requested_by=user.user_id,
        targets_by_zone=body.targets_by_zone,
        generation_parameters=body.generation_parameters,
    )


@generation_router.post(
    "/generate/3d/by_territory",
    summary="Generate buildings for territories and queue a 3D facade job",
    response_model=FacadeJobAccepted,
    status_code=202,
)
async def generate_3d_by_territory(
    payload: TerritoryRequest = Body(..., description="Body for request"),
) -> dict[str, str]:
    # The original /generate/by_territory endpoint is intentionally anonymous;
    # mirror that contract and let facade-jobs place it in the anonymous quota.
    return await orchestration.generate_3d_by_territory(payload)


@generation_router.post(
    "/generate/3d/by_blocks",
    summary="Generate buildings for blocks and queue a 3D facade job",
    response_model=FacadeJobAccepted,
    status_code=202,
)
async def generate_3d_by_blocks(
    scenario_id: Annotated[int, Query(..., description="Scenario ID", examples=[198])],
    year: Annotated[int, Query(..., description="Data year", examples=[2024])],
    source: Annotated[str, Query(..., description="Data source", examples=["OSM"])],
    functional_zone_types: Annotated[
        List[str],
        Query(
            ...,
            description="Target functional zone types",
            examples=[["residential", "business", "industrial"]],
        ),
    ],
    physical_object_id: Annotated[
        Optional[List[int]],
        Query(
            description="Physical object id(s) to exclude from generation territory.",
            examples=[[2058130, 2058131]],
        ),
    ] = None,
    user: auth.AuthUser = Depends(auth.get_current_user),
    body: FunctionalZonesRequest = Body(
        ..., description="Per-zone targets and generation parameters"
    ),
) -> dict[str, str]:
    return await orchestration.generate_3d_by_blocks(
        scenario_id=scenario_id,
        year=year,
        source=source,
        functional_zone_types=functional_zone_types,
        physical_object_id=physical_object_id,
        token=user.token,
        requested_by=user.user_id,
        body=body,
    )


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
    return await orchestration.estimate_max_residents_by_blocks(
        scenario_id=scenario_id,
        year=year,
        source=source,
        functional_zone_types=functional_zone_types,
        functional_zone_ids=functional_zone_ids,
        token=token,
    )
