"""A2A discovery + JSON-RPC routes, mounted in-process on the main FastAPI app.

No example project was given for this half of the request, so the shape
here is GenBuilder-specific: a single AgentSkill wrapping the existing
conversational generation flow (see ``app.a2a_server.agent``). The agent
card is served at the A2A-standard well-known path; JSON-RPC (including
streaming via ``message/stream``) is served at ``/a2a``.
"""
from __future__ import annotations

import os

from fastapi import FastAPI

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

from app.a2a_server.agent import GenBuilderAgentExecutor

_RPC_PATH = "/a2a"


def _public_base_url() -> str:
    return (os.getenv("A2A_PUBLIC_URL") or "http://localhost:8000").rstrip("/")


def _build_agent_card() -> AgentCard:
    skill = AgentSkill(
        id="generate_buildings_chat",
        name="Generate buildings from a natural-language request",
        description=(
            "Takes a free-text building-generation request for a UrbanDB scenario "
            "territory or caller-supplied block polygons, and returns generated "
            "buildings as a GeoJSON FeatureCollection artifact. Asks a clarifying "
            "question (task state input-required) if mandatory generation targets "
            "(e.g. residents, coverage area) are missing from the request."
        ),
        input_modes=["text/plain", "application/json"],
        output_modes=["application/json", "text/plain"],
        tags=["generation", "urban-planning", "geojson"],
        examples=[
            "Generate residential buildings for 5000 residents in scenario 198, 2024, OSM",
            "Generate mixed residential/business buildings for these block polygons",
        ],
    )
    return AgentCard(
        name="GenBuilder Agent",
        description="Conversational building-generation agent for urban blocks and UrbanDB scenarios.",
        version="0.1.0",
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["application/json", "text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                url=f"{_public_base_url()}{_RPC_PATH}",
                protocol_version="1.0",
            )
        ],
        skills=[skill],
    )


def register_a2a_routes(app: FastAPI) -> None:
    """Mount A2A discovery (``/.well-known/agent-card.json``) and JSON-RPC
    (``/a2a``, streaming-capable) routes on the main FastAPI app."""
    agent_card = _build_agent_card()
    request_handler = DefaultRequestHandler(
        agent_executor=GenBuilderAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(agent_card),
        # v0.3 compat: most A2A clients in the wild don't send the
        # A2A-Version header yet and default to the older method-name/shape
        # conventions ("message/send" etc.) — accept both on one endpoint.
        jsonrpc_routes=create_jsonrpc_routes(request_handler, rpc_url=_RPC_PATH, enable_v0_3_compat=True),
    )


__all__ = ["register_a2a_routes"]
