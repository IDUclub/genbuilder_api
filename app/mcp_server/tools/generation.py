"""MCP tools for building generation.

Each tool is a thin wrapper around :mod:`app.logic.generation_orchestration`
— the same functions the ``/generate/*`` REST routes call — so behaviour
stays identical between the REST API and MCP. Tools that operate on a
UrbanDB scenario (anything keyed by ``scenario_id``) require the caller's
Keycloak bearer token, taken from the Authorization header automatically
(see ``app.mcp_server.auth``); ``generate_by_territory`` needs no token
since it operates on inline geometry only.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional

from fastmcp import FastMCP
from pydantic import ValidationError

from mcp import ErrorData, McpError

from app.logic import generation_orchestration as orchestration
from app.mcp_server.auth import require_verified_token
from app.mcp_server.exceptions import map_errors
from app.schema.dto import FunctionalZonesRequest, ScenarioBody, TerritoryRequest

generation_mcp = FastMCP("GenBuilder Generation")


def _validation_error(exc: ValidationError) -> McpError:
    return McpError(ErrorData(code=-32602, message=f"Invalid params: {exc}"))


def _validate(model: type, **fields: Any):
    """Validate through a DTO, omitting unset (None) fields so the model's own
    defaults apply — mirrors how FastAPI builds the same model from a JSON
    body where the key was simply absent. Passing ``None`` explicitly would
    instead *override* a non-None default (e.g. ``TerritoryRequest.targets_by_zone``)."""
    try:
        return model.model_validate({k: v for k, v in fields.items() if v is not None})
    except ValidationError as exc:
        raise _validation_error(exc) from exc


@generation_mcp.tool(
    name="generate_by_scenario",
    title="Generate buildings for a scenario's territory",
    description="""Generate buildings across the whole territory of a UrbanDB scenario.

USE WHEN: the user wants building generation for an entire project/scenario
(all its functional zones), not a specific subset of blocks.

AUTH (automatic, do NOT ask the user): the caller's Keycloak bearer token is
taken from the Authorization header and forwarded to UrbanDB.

PARAMETERS
- scenario_id (int, required): the project/scenario to generate for.
- year (int, required): data year of the scenario's functional zones.
- source (string, required): zone data source, e.g. "OSM", "PZZ", "User".
- functional_zone_types (list[string], required): zone types to generate
  into, e.g. ["residential", "business", "industrial"].
- physical_object_id (list[int], optional): physical object ids to exclude
  from the generation territory (e.g. buildings already present).
- targets_by_zone (object, optional): per-zone generation targets
  (residents / coverage_area / floors_avg / density_scenario /
  default_floor_group). Omit to use the service defaults.
- generation_parameters (object, optional): low-level generation parameter
  overrides (e.g. {"rectangle_finder_step": 5}).

RETURNS: a GeoJSON FeatureCollection of generated + excluded building
features (each feature carries floors_count / living_area /
functional_area / building_area / zone / service in its properties).

ERRORS: -32002 AUTH_TOKEN_EXPIRED if the token is rejected upstream.""",
    tags={"generation", "scenario"},
)
@map_errors
async def generate_by_scenario(
    scenario_id: Annotated[int, "The project/scenario id to generate for."],
    year: Annotated[int, "Data year of the scenario's functional zones."],
    source: Annotated[str, "Zone data source, e.g. 'OSM', 'PZZ', 'User'."],
    functional_zone_types: Annotated[list[str], "Target functional zone types."],
    physical_object_id: Annotated[
        Optional[list[int]], "Physical object id(s) to exclude from the territory."
    ] = None,
    targets_by_zone: Annotated[
        Optional[dict[str, dict[str, Any]]], "Per-zone generation targets."
    ] = None,
    generation_parameters: Annotated[
        Optional[dict[str, Any]], "Generation parameter overrides."
    ] = None,
) -> dict[str, Any]:
    token = await require_verified_token()
    body = _validate(
        ScenarioBody,
        targets_by_zone=targets_by_zone,
        generation_parameters=generation_parameters,
    )
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


@generation_mcp.tool(
    name="generate_by_territory",
    title="Generate buildings for arbitrary block polygons",
    description="""Generate buildings for a caller-supplied set of block polygons — no
UrbanDB scenario needed.

USE WHEN: the user has their own GeoJSON blocks (each with a `zone`
property) and wants generation without referencing a scenario_id.

AUTH: none required — this tool operates only on inline geometry.

PARAMETERS
- blocks (GeoJSON FeatureCollection, required): Polygon features, each with
  `properties.zone` set (e.g. "residential", "business").
- targets_by_zone (object, optional): per-zone generation targets. Omit to
  use the service defaults.
- generation_parameters (object, optional): low-level generation parameter
  overrides.

RETURNS: a GeoJSON FeatureCollection of generated + excluded building
features.

ERRORS: -32602 Invalid params if a block's geometry isn't a Polygon or is
missing the `zone` property.""",
    tags={"generation", "territory"},
)
@map_errors
async def generate_by_territory(
    blocks: Annotated[
        dict[str, Any],
        "GeoJSON FeatureCollection of Polygon blocks; each feature needs properties.zone.",
    ],
    targets_by_zone: Annotated[
        Optional[dict[str, dict[str, Any]]], "Per-zone generation targets."
    ] = None,
    generation_parameters: Annotated[
        Optional[dict[str, Any]], "Generation parameter overrides."
    ] = None,
) -> dict[str, Any]:
    payload = _validate(
        TerritoryRequest,
        blocks=blocks,
        targets_by_zone=targets_by_zone,
        generation_parameters=generation_parameters,
    )
    return await orchestration.generate_by_territory(payload)


@generation_mcp.tool(
    name="generate_by_blocks",
    title="Generate buildings for specific functional zones of a scenario",
    description="""Generate buildings for a chosen subset of a scenario's functional
zones, with per-zone targets — one generation run per zone (or per polygon
part, for a MultiPolygon zone).

USE WHEN: the user wants generation for specific functional zone ids within
a scenario, each with its own targets, rather than the whole territory.

AUTH (automatic): the caller's Keycloak bearer token, forwarded to UrbanDB.

PARAMETERS
- scenario_id, year, source, functional_zone_types: same as generate_by_scenario.
- physical_object_id (list[int], optional): ids to exclude from the territory.
- zones (list, required): [{ functional_zone_id (int), targets_by_zone (object),
  generation_parameters (object, optional) }, ...] — one entry per zone to generate.

RETURNS: a GeoJSON FeatureCollection combining generated buildings from all
requested zones.

ERRORS:
- -32602 if a requested functional_zone_id doesn't exist for this scenario/year/source.
- -32602 if a zone's geometry type is unsupported (only Polygon/MultiPolygon).""",
    tags={"generation", "scenario", "zones"},
)
@map_errors
async def generate_by_blocks(
    scenario_id: Annotated[int, "The project/scenario id."],
    year: Annotated[int, "Data year of the scenario's functional zones."],
    source: Annotated[str, "Zone data source, e.g. 'OSM', 'PZZ', 'User'."],
    functional_zone_types: Annotated[list[str], "Target functional zone types."],
    zones: Annotated[
        list[dict[str, Any]],
        "Per-zone generation configs: "
        "[{functional_zone_id, targets_by_zone, generation_parameters?}, ...].",
    ],
    physical_object_id: Annotated[
        Optional[list[int]], "Physical object id(s) to exclude from the territory."
    ] = None,
) -> dict[str, Any]:
    body = _validate(FunctionalZonesRequest, zones=zones)

    token = await require_verified_token()
    return await orchestration.generate_by_blocks(
        scenario_id=scenario_id,
        year=year,
        source=source,
        functional_zone_types=functional_zone_types,
        physical_object_id=physical_object_id,
        token=token,
        body=body,
    )


@generation_mcp.tool(
    name="estimate_max_residents_by_blocks",
    title="Estimate maximum residents for functional zones",
    description="""Run generation with the service's default (maximum-density) targets
for each given functional zone and return the resulting resident count per zone.

USE WHEN: the user wants a capacity estimate ("how many people could this
zone hold at maximum density") rather than a generated building layout.

AUTH (automatic): the caller's Keycloak bearer token, forwarded to UrbanDB.

PARAMETERS
- scenario_id, year, source, functional_zone_types: same as generate_by_scenario.
- functional_zone_ids (list[int], required): the functional zones to estimate.

RETURNS: an object mapping each functional_zone_id (as a string key) to its
estimated resident count.""",
    tags={"generation", "scenario", "estimate"},
    annotations={"readOnlyHint": True},
)
@map_errors
async def estimate_max_residents_by_blocks(
    scenario_id: Annotated[int, "The project/scenario id."],
    year: Annotated[int, "Data year of the scenario's functional zones."],
    source: Annotated[str, "Zone data source, e.g. 'OSM', 'PZZ', 'User'."],
    functional_zone_types: Annotated[list[str], "Target functional zone types."],
    functional_zone_ids: Annotated[list[int], "Functional zone ids to estimate."],
) -> dict[int, int]:
    token = await require_verified_token()
    return await orchestration.estimate_max_residents_by_blocks(
        scenario_id=scenario_id,
        year=year,
        source=source,
        functional_zone_types=functional_zone_types,
        functional_zone_ids=functional_zone_ids,
        token=token,
    )
