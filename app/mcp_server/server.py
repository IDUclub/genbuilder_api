"""MCP server, mounted in-process on the main FastAPI app (see ``app/main.py``).

Unlike IDUclub/PzzCompareAPI's MCP server — a separate process that talks to
its API over HTTP because that project already runs Celery/Redis for
background tasks — GenBuilder has no task queue: generation runs in-request
inside this same process. So the MCP tools call the existing service layer
(``app.logic.generation_orchestration``) directly instead of round-tripping
through HTTP to itself.
"""
from __future__ import annotations

from fastmcp import FastMCP

from app.mcp_server.tools.generation import generation_mcp

_INSTRUCTIONS = """GenBuilder — generate building layouts for urban blocks and UrbanDB
scenarios.

THREE WAYS TO GENERATE:
1. generate_by_scenario: generate across a whole scenario's territory. Needs
   scenario_id, year, source, functional_zone_types. Requires the caller's
   Keycloak bearer token (taken automatically from the Authorization header).
2. generate_by_blocks: generate for specific functional zone ids within a
   scenario, each with its own targets. Same auth as above.
3. generate_by_territory: generate for caller-supplied GeoJSON block
   polygons, no scenario needed. No auth required.

Use estimate_max_residents_by_blocks to get a capacity estimate (max
residents per zone) without producing a full building layout.

If a tool returns AUTH_TOKEN_EXPIRED, ask for a fresh bearer token and
retry — do not reuse the rejected one."""

main_mcp = FastMCP("GenBuilder MCP", instructions=_INSTRUCTIONS)
main_mcp.mount(generation_mcp)

# Mounted at /mcp on the main app (``app.mount("/mcp", mcp_app)``), so the
# sub-app's own path must be "/". Host/origin checks are left to the parent
# app's existing CORS + Keycloak auth layers rather than duplicated here.
mcp_app = main_mcp.http_app(path="/", host_origin_protection=False)

__all__ = ["main_mcp", "mcp_app"]
