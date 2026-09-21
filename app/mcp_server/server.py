"""MCP server, mounted in-process on the main FastAPI app (see ``app/main.py``).

Unlike IDUclub/PzzCompareAPI's MCP server — a separate process that talks to
its API over HTTP because that project already runs Celery/Redis for
background tasks — GenBuilder has no task queue: generation runs in-request
inside this same process. So the MCP tools call the existing service layer
(``app.logic.generation_orchestration``) directly instead of round-tripping
through HTTP to itself.
"""
from __future__ import annotations

import inspect

from fastmcp import FastMCP

from app.mcp_server.tools.generation import generation_mcp

_INSTRUCTIONS = """GenBuilder — generate building layouts for urban blocks and UrbanDB
scenarios.

START WITH list_functional_zones (scenario_id, year, source) to see which
functional zones exist, their ids, types and areas. An empty list means no
zones for that year/source — do not generate against it.

THREE WAYS TO GENERATE:
1. generate_by_scenario: generate across a whole scenario's territory. Needs
   scenario_id, year, source, functional_zone_types. Requires the caller's
   Keycloak bearer token (taken automatically from the Authorization header).
2. generate_by_blocks: generate for specific functional zone ids within a
   scenario, each with its own targets. Same auth as above.
3. generate_by_territory: generate for caller-supplied GeoJSON block
   polygons, no scenario needed. No auth required.

TARGETS: generate_by_scenario / generate_by_territory require targets_by_zone.
Pass use_defaults=true only if the user explicitly accepts the service
default targets; the result's summary.targets_source says which were used.

EXISTING BUILDINGS: in scenario modes pass preserve_existing_buildings=true
unless the site is greenfield — existing buildings are then kept and nothing
is generated on top of them. If they can't be loaded the tool fails instead
of silently ignoring them.

RESULTS: every generation result carries `summary` (buildings, residents,
living area, per-zone breakdown, excluded buildings, and target vs achieved
with the deficit per zone). Ground your answer on it instead of counting
features.

RESULT STORAGE: layers are large, so generation tools store the full
FeatureCollection and return `result_id` + `layer` (a /files link for maps or
other services) instead of geometry. Pass include_geometry=true only when you
really need the features inline, or read them later with
get_generation_result(result_id). Stored results expire after a while. If
`storage_warning` is present, storage failed and the features are inlined.

REPRODUCIBILITY: each result echoes `seed`, `applied_parameters` (effective
generation parameters and targets), `generation_id` and `duration_s`. To
reproduce a layout, call the same tool with the same inputs and that seed.

Use estimate_max_residents_by_blocks to get a capacity estimate (zone area,
max residents, max living area, existing buildings per zone) without
producing a full building layout.

LONG CHAINS: the bearer token is read from the Authorization header on every
call and lives ~5 minutes, so refresh it between steps of a long chain rather
than holding one token for the whole plan. generate_by_blocks and
estimate_max_residents_by_blocks send progress notifications per zone when
the request carries a progressToken; cancelling the request stops them at
the next zone boundary.

If a tool returns AUTH_TOKEN_EXPIRED, ask for a fresh bearer token and
retry — do not reuse the rejected one."""

main_mcp = FastMCP("GenBuilder MCP", instructions=_INSTRUCTIONS)
main_mcp.mount(generation_mcp)

_http_app_kwargs: dict = {"path": "/"}
if "host_origin_protection" in inspect.signature(main_mcp.http_app).parameters:
    _http_app_kwargs["host_origin_protection"] = False

mcp_app = main_mcp.http_app(**_http_app_kwargs)

__all__ = ["main_mcp", "mcp_app"]
