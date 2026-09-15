"""MCP tools for building generation.

Each tool is a thin wrapper around :mod:`app.logic.generation_orchestration`
— the same functions the ``/generate/*`` REST routes call — so behaviour
stays identical between the REST API and MCP. Tools that operate on a
UrbanDB scenario (anything keyed by ``scenario_id``) require the caller's
Keycloak bearer token, taken from the Authorization header automatically
(see ``app.mcp_server.auth``); ``generate_by_territory`` needs no token
since it operates on inline geometry only.

Unlike the REST routes, generation tools never fall back to default targets
silently: an agent must either pass ``targets_by_zone`` or opt in with
``use_defaults: true``, and every result carries a ``summary`` saying which
targets were used and how far the layout falls short of them.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional

from fastmcp import FastMCP
from pydantic import ValidationError

from mcp import ErrorData, McpError

from app.logic import generation_orchestration as orchestration
from app.logic.generation_summary import TargetsSource, build_generation_summary, combine_targets
from app.mcp_server.auth import require_verified_token
from app.mcp_server.exceptions import map_errors
from app.schema.dto import FunctionalZonesRequest, ScenarioBody, TerritoryRequest

generation_mcp = FastMCP("GenBuilder Generation")

_SUMMARY_DOC = """`summary` — totals so the result need not be walked feature by feature:
  buildings, living_area_total, residents_total, buildings_by_zone and
  residents_by_zone (newly generated buildings only); excluded_buildings,
  existing_living_area, existing_residents (buildings kept as is);
  targets — per zone target_residents / achieved_residents /
  residents_deficit and target_functional_area / achieved_functional_area /
  functional_area_deficit; targets_source ("request" or "service_defaults");
  existing_buildings_preserved (scenario modes)."""


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


def _targets_source(targets_by_zone: Optional[dict[str, Any]], use_defaults: bool) -> TargetsSource:
    """Refuse to generate against targets nobody chose.

    The service defaults are a fixed demo profile (e.g. 1100 residents per
    residential zone); applying them silently yields a plausible-looking layout
    that answers a question the user never asked.
    """
    if targets_by_zone is not None:
        return "request"
    if use_defaults:
        return "service_defaults"
    raise McpError(
        ErrorData(
            code=-32602,
            message=(
                "Invalid params: targets_by_zone is required. Pass per-zone targets "
                '(e.g. {"residents": {"residential": 3000}, "coverage_area": {"business": 20000}}) '
                "or set use_defaults=true to explicitly generate with the service default targets."
            ),
        )
    )


def _with_summary(fc: dict[str, Any], **summary_kwargs: Any) -> dict[str, Any]:
    return {**fc, "summary": build_generation_summary(fc.get("features") or [], **summary_kwargs)}


@generation_mcp.tool(
    name="list_functional_zones",
    title="List a scenario's functional zones",
    description="""List the functional zones of a UrbanDB scenario with their ids, types and
areas.

USE WHEN: before generate_by_blocks / estimate_max_residents_by_blocks, to find
which functional_zone_ids exist and how large they are, or to answer "which
zones does this scenario have".

AUTH (automatic, do NOT ask the user): the caller's Keycloak bearer token.

PARAMETERS
- scenario_id (int, required): the project/scenario.
- year (int, required): data year of the functional zones.
- source (string, required): zone data source, e.g. "OSM", "PZZ", "User".
- functional_zone_types (list[string], optional): keep only these zone types.

RETURNS: { scenario_id, year, source,
  zones: [{ functional_zone_id, functional_zone_type, generation_zone (the
  canonical zone generation uses, e.g. residential_midrise -> residential),
  name, geometry_type, area_m2 }],
  totals_by_type: { <functional_zone_type>: { count, area_m2 } } }.
An empty `zones` list means the scenario has no zones for that year/source —
try another year/source rather than generating.""",
    tags={"zones", "scenario"},
    annotations={"readOnlyHint": True},
)
@map_errors
async def list_functional_zones(
    scenario_id: Annotated[int, "The project/scenario id."],
    year: Annotated[int, "Data year of the scenario's functional zones."],
    source: Annotated[str, "Zone data source, e.g. 'OSM', 'PZZ', 'User'."],
    functional_zone_types: Annotated[
        Optional[list[str]], "Only list zones of these functional zone types."
    ] = None,
) -> dict[str, Any]:
    token = await require_verified_token()
    return await orchestration.list_functional_zones(
        scenario_id=scenario_id,
        year=year,
        source=source,
        token=token,
        functional_zone_types=functional_zone_types,
    )


@generation_mcp.tool(
    name="generate_by_scenario",
    title="Generate buildings for a scenario's territory",
    description=f"""Generate buildings across the whole territory of a UrbanDB scenario.

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
- targets_by_zone (object, required unless use_defaults=true): per-zone
  generation targets (residents / coverage_area / floors_avg /
  density_scenario / default_floor_group), keyed by canonical zone
  (residential, business, industrial, transport, special, unknown).
- use_defaults (bool, default false): set true ONLY when the user explicitly
  accepts the service default targets; then omit targets_by_zone.
- preserve_existing_buildings (bool, default false): keep every existing
  building of the scenario — their footprints are cut out of the territory
  and they come back marked `is_excluded: true`. Set true whenever the
  territory is not a greenfield site.
- physical_object_id (list[int], optional): extra physical object ids to
  exclude from the generation territory.
- generation_parameters (object, optional): low-level generation parameter
  overrides (e.g. {{"rectangle_finder_step": 5}}).

RETURNS: a GeoJSON FeatureCollection of generated + excluded building
features (each feature carries floors_count / living_area /
functional_area / building_area / zone / service in its properties), plus
{_SUMMARY_DOC}

ERRORS:
- -32602 if neither targets_by_zone nor use_defaults=true is given.
- -32002 AUTH_TOKEN_EXPIRED if the token is rejected upstream.
- -32603 if existing buildings / physical objects can't be loaded — generation
  is refused rather than run on top of them.""",
    tags={"generation", "scenario"},
)
@map_errors
async def generate_by_scenario(
    scenario_id: Annotated[int, "The project/scenario id to generate for."],
    year: Annotated[int, "Data year of the scenario's functional zones."],
    source: Annotated[str, "Zone data source, e.g. 'OSM', 'PZZ', 'User'."],
    functional_zone_types: Annotated[list[str], "Target functional zone types."],
    targets_by_zone: Annotated[
        Optional[dict[str, dict[str, Any]]],
        "Per-zone generation targets. Required unless use_defaults is true.",
    ] = None,
    use_defaults: Annotated[
        bool, "Explicitly generate with the service default targets (only when targets_by_zone is omitted)."
    ] = False,
    preserve_existing_buildings: Annotated[
        bool, "Keep the scenario's existing buildings and generate only around them."
    ] = False,
    physical_object_id: Annotated[
        Optional[list[int]], "Physical object id(s) to exclude from the territory."
    ] = None,
    generation_parameters: Annotated[
        Optional[dict[str, Any]], "Generation parameter overrides."
    ] = None,
) -> dict[str, Any]:
    targets_source = _targets_source(targets_by_zone, use_defaults)
    body = _validate(
        ScenarioBody,
        targets_by_zone=targets_by_zone,
        generation_parameters=generation_parameters,
    )
    token = await require_verified_token()
    fc = await orchestration.generate_by_scenario(
        scenario_id=scenario_id,
        year=year,
        source=source,
        functional_zone_types=functional_zone_types,
        physical_object_id=physical_object_id,
        token=token,
        targets_by_zone=body.targets_by_zone,
        generation_parameters=body.generation_parameters,
        preserve_existing_buildings=preserve_existing_buildings,
    )
    return _with_summary(
        fc,
        targets_by_zone=body.targets_by_zone,
        targets_source=targets_source,
        existing_buildings_preserved=preserve_existing_buildings,
    )


@generation_mcp.tool(
    name="generate_by_territory",
    title="Generate buildings for arbitrary block polygons",
    description=f"""Generate buildings for a caller-supplied set of block polygons — no
UrbanDB scenario needed.

USE WHEN: the user has their own GeoJSON blocks (each with a `zone`
property) and wants generation without referencing a scenario_id.

AUTH: none required — this tool operates only on inline geometry.

PARAMETERS
- blocks (GeoJSON FeatureCollection, required): Polygon/MultiPolygon features,
  each with `properties.zone` set (e.g. "residential", "business").
- targets_by_zone (object, required unless use_defaults=true): per-zone
  generation targets.
- use_defaults (bool, default false): set true ONLY when the user explicitly
  accepts the service default targets; then omit targets_by_zone.
- existing_buildings (GeoJSON FeatureCollection, optional): footprints of
  buildings that already stand on the territory. They are cut out of the
  blocks before generation, so nothing is generated on top of them, and they
  come back in the response marked `is_excluded: true`. Properties are
  optional — the geometry is what matters.
- generation_parameters (object, optional): low-level generation parameter
  overrides.

RETURNS: a GeoJSON FeatureCollection of generated + excluded building
features, plus
{_SUMMARY_DOC}

ERRORS: -32602 Invalid params if neither targets_by_zone nor use_defaults=true
is given, or a block's geometry is missing, isn't a Polygon/MultiPolygon, or
the `zone` property is missing.""",
    tags={"generation", "territory"},
)
@map_errors
async def generate_by_territory(
    blocks: Annotated[
        dict[str, Any],
        "GeoJSON FeatureCollection of Polygon/MultiPolygon blocks; each feature needs properties.zone.",
    ],
    targets_by_zone: Annotated[
        Optional[dict[str, dict[str, Any]]],
        "Per-zone generation targets. Required unless use_defaults is true.",
    ] = None,
    use_defaults: Annotated[
        bool, "Explicitly generate with the service default targets (only when targets_by_zone is omitted)."
    ] = False,
    existing_buildings: Annotated[
        Optional[dict[str, Any]],
        "GeoJSON FeatureCollection of existing building footprints to exclude from generation.",
    ] = None,
    generation_parameters: Annotated[
        Optional[dict[str, Any]], "Generation parameter overrides."
    ] = None,
) -> dict[str, Any]:
    targets_source = _targets_source(targets_by_zone, use_defaults)
    payload = _validate(
        TerritoryRequest,
        blocks=blocks,
        existing_buildings=existing_buildings,
        targets_by_zone=targets_by_zone,
        generation_parameters=generation_parameters,
    )
    fc = await orchestration.generate_by_territory(payload)
    return _with_summary(
        fc,
        targets_by_zone=payload.targets_by_zone,
        targets_source=targets_source,
    )


@generation_mcp.tool(
    name="generate_by_blocks",
    title="Generate buildings for specific functional zones of a scenario",
    description=f"""Generate buildings for a chosen subset of a scenario's functional
zones, with per-zone targets — one generation run per zone (or per polygon
part, for a MultiPolygon zone).

USE WHEN: the user wants generation for specific functional zone ids within
a scenario, each with its own targets, rather than the whole territory. Call
list_functional_zones first to get valid ids.

AUTH (automatic): the caller's Keycloak bearer token, forwarded to UrbanDB.

PARAMETERS
- scenario_id, year, source, functional_zone_types: same as generate_by_scenario.
- zones (list, required): [{{ functional_zone_id (int), targets_by_zone (object,
  required), generation_parameters (object, optional) }}, ...] — one entry per
  zone to generate.
- preserve_existing_buildings (bool, default false): keep the scenario's
  existing buildings inside the requested zones — they are cut out and come
  back marked `is_excluded: true`.
- physical_object_id (list[int], optional): extra ids to exclude from the territory.

RETURNS: a GeoJSON FeatureCollection combining generated buildings from all
requested zones, plus
{_SUMMARY_DOC}
Targets in the summary are the sum of the per-zone targets.

ERRORS:
- -32602 if a requested functional_zone_id doesn't exist for this scenario/year/source.
- -32602 if a zone's geometry type is unsupported (only Polygon/MultiPolygon).
- -32603 if existing buildings can't be loaded while preserve_existing_buildings=true.""",
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
    preserve_existing_buildings: Annotated[
        bool, "Keep existing buildings inside the requested zones and generate only around them."
    ] = False,
    physical_object_id: Annotated[
        Optional[list[int]], "Physical object id(s) to exclude from the territory."
    ] = None,
) -> dict[str, Any]:
    body = _validate(FunctionalZonesRequest, zones=zones)

    token = await require_verified_token()
    fc = await orchestration.generate_by_blocks(
        scenario_id=scenario_id,
        year=year,
        source=source,
        functional_zone_types=functional_zone_types,
        physical_object_id=physical_object_id,
        token=token,
        body=body,
        preserve_existing_buildings=preserve_existing_buildings,
    )
    return _with_summary(
        fc,
        targets_by_zone=combine_targets(zone.targets_by_zone for zone in body.zones),
        targets_source="request",
        existing_buildings_preserved=preserve_existing_buildings,
    )


_CAPACITY_TOTAL_KEYS = (
    "zone_area_m2",
    "max_residents",
    "max_living_area",
    "existing_buildings_count",
    "existing_living_area",
    "existing_residents",
)


@generation_mcp.tool(
    name="estimate_max_residents_by_blocks",
    title="Estimate capacity of functional zones",
    description="""Run generation with the service's maximum-density targets for each
given functional zone and report its capacity.

USE WHEN: the user wants a capacity estimate ("how many people could this
zone hold at maximum density") rather than a generated building layout, or
before generate_by_blocks to choose realistic targets.

AUTH (automatic): the caller's Keycloak bearer token, forwarded to UrbanDB.

PARAMETERS
- scenario_id, year, source, functional_zone_types: same as generate_by_scenario.
- functional_zone_ids (list[int], required): the functional zones to estimate
  (see list_functional_zones).
- preserve_existing_buildings (bool, default false): cut the zones' existing
  buildings out first, so the estimate is the ADDITIONAL capacity of the
  remaining land. Existing buildings inside each zone are reported either way.

RETURNS: {
  zones: [{ functional_zone_id, functional_zone_type, zone_area_m2,
  max_residents, max_living_area, existing_buildings_count,
  existing_living_area, existing_residents }],
  totals: { the same numeric fields summed over all zones },
  existing_buildings_preserved: bool }.

ERRORS:
- -32602 if a functional_zone_id doesn't exist for this scenario/year/source.
- -32603 if existing buildings can't be loaded from UrbanDB.""",
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
    preserve_existing_buildings: Annotated[
        bool, "Estimate only the additional capacity around existing buildings."
    ] = False,
) -> dict[str, Any]:
    token = await require_verified_token()
    estimates = await orchestration.estimate_capacity_by_blocks(
        scenario_id=scenario_id,
        year=year,
        source=source,
        functional_zone_types=functional_zone_types,
        functional_zone_ids=functional_zone_ids,
        token=token,
        preserve_existing_buildings=preserve_existing_buildings,
    )
    zones = list(estimates.values())
    totals = {key: sum(zone[key] for zone in zones) for key in _CAPACITY_TOTAL_KEYS}
    for key in ("zone_area_m2", "max_living_area", "existing_living_area"):
        totals[key] = round(totals[key], 1)
    return {
        "zones": zones,
        "totals": totals,
        "existing_buildings_preserved": preserve_existing_buildings,
    }
