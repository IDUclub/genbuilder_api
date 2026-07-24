import os
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastmcp.utilities.lifespan import combine_lifespans
from loguru import logger
from starlette.responses import RedirectResponse

from app.a2a_server import register_a2a_routes
from app.dependencies import (
    build_keycloak_token_config,
    config,
    keycloak_service_configured,
    set_service_token_client,
    setup_logger,
)
from app.mcp_server import mcp_app
from app.routers.generation_routers import generation_router
from app.routers.generation_chat_routers import generation_chat_router
from app.routers.logs_routers import logs_router
from app.observability import OpenTelemetryAgent, PrometheusConfig
from app.observability.metrics import setup_metrics
from app.common.middlewares import ExceptionHandlerMiddleware, ObservabilityMiddleware

setup_logger(config)

metrics = setup_metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    prometheus_port = int(os.getenv("PROMETHEUS_PORT", "9464"))
    otel_agent = OpenTelemetryAgent(
        prometheus_config=PrometheusConfig(host="0.0.0.0", port=prometheus_port)
    )

    async with AsyncExitStack() as stack:
        # Shared Keycloak service-token client for outbound M2M auth (ChatStorage).
        # Created once per process; background refresh keeps the token fresh.
        if keycloak_service_configured():
            from idu_service_auth import KeycloakTokenClient

            token_client = await stack.enter_async_context(
                KeycloakTokenClient(build_keycloak_token_config())
            )
            set_service_token_client(token_client)
            logger.info("Keycloak service token client initialized.")
        else:
            logger.warning(
                "Keycloak service credentials are not fully configured "
                "(KEYCLOAK_URL/REALM/CLIENT_ID/CLIENT_SECRET); ChatStorage "
                "persistence will be disabled."
            )
        try:
            yield
        finally:
            set_service_token_client(None)
            otel_agent.shutdown()


app = FastAPI(
    title="GenBuilder API",
    version="0.1.1",
    lifespan=combine_lifespans(lifespan, mcp_app.lifespan),
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ExceptionHandlerMiddleware, metrics=metrics)
app.add_middleware(ObservabilityMiddleware, metrics=metrics)


@app.get("/", include_in_schema=False)
async def read_root():
    return RedirectResponse("/docs")


app.include_router(logs_router)
app.include_router(generation_router)
app.include_router(generation_chat_router)

# MCP tools (see app/mcp_server) — streamable-HTTP transport at /mcp.
app.mount("/mcp", mcp_app)

# A2A discovery (/.well-known/agent-card.json) + JSON-RPC (/a2a).
register_a2a_routes(app)
