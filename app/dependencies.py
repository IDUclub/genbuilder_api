from __future__ import annotations

import os
from typing import TYPE_CHECKING

from iduconfig import Config
from loguru import logger

from app.logic.functional_zones_service import FunctionalZonesService
from app.logic.logger_setup import setup_logger

from app.api.urbandb_api_gateway import UrbanDBAPI

from app.logic.generation_params import GenParams, ParamsProvider
from app.logic.building_capacity_optimizer import CapacityOptimizer
from app.logic.maximum_inscribed_rectangle import MIR
from app.logic.physical_objects_service import PhysicalObjectsService
from app.logic.segments.capacity_calculator import SegmentCapacityCalculator
from app.logic.segments.scenario_search import BlockScenarioSearch
from app.logic.segments.plots_allocator import BlockPlotsAllocator
from app.logic.segments.context import BlockSegmentsContextBuilder
from app.logic.segments.solver import BlockSolver
from app.logic.segments.segments import SegmentsAllocator
from app.logic.plots.plots import PlotsGenerator
from app.logic.plots.plot_slicer import PlotSegmentSlicer
from app.logic.plots.plot_merge import PlotMerger
from app.logic.plots.plot_tuner import PlotTuner
from app.logic.buildings import BuildingsGenerator
from app.logic.block_generator import BlockGenerator
from app.logic.service_generation import ServiceGenerator
from app.logic.building_params import (
    BuildingGenParams,
    BuildingParamsProvider,
    PARAMS_BY_TYPE
)
from app.logic.generation import Genbuilder
from app.infrastructure.ollama_chat_client import OllamaChatClient
from app.infrastructure.chat_storage_client import ChatStorageClient

if TYPE_CHECKING:
    from idu_service_auth import KeycloakTokenClient, KeycloakTokenConfig

config = Config()
setup_logger(config, log_level="INFO")

urban_db_api = UrbanDBAPI(config)

base_params = GenParams()
params_provider = ParamsProvider(base_params)

buildings_params = BuildingGenParams(params_by_type=PARAMS_BY_TYPE)
buildings_params_provider = BuildingParamsProvider(base=buildings_params)

building_capacity_optimizer = CapacityOptimizer(buildings_params_provider)
max_rectangle_finder = MIR()

segment_capacity_calculator = SegmentCapacityCalculator()
scenario_search = BlockScenarioSearch()
plots_allocator = BlockPlotsAllocator()
segment_context = BlockSegmentsContextBuilder()
block_solver = BlockSolver(building_capacity_optimizer, segment_capacity_calculator, 
                           scenario_search, plots_allocator, segment_context)
segments_allocator = SegmentsAllocator(building_capacity_optimizer, buildings_params_provider, block_solver)

plot_slicer = PlotSegmentSlicer()
plor_merger = PlotMerger()
plot_tuner = PlotTuner(params_provider, buildings_params_provider)
plots_generator = PlotsGenerator(params_provider, buildings_params_provider, building_capacity_optimizer, plot_slicer, plor_merger, plot_tuner)

buildings_generator = BuildingsGenerator()
block_generator = BlockGenerator(building_capacity_optimizer, max_rectangle_finder, 
                    segments_allocator, plots_generator, buildings_generator, params_provider)
service_generator = ServiceGenerator(params_provider)
physical_objects_service = PhysicalObjectsService()
builder = Genbuilder(
    config, urban_db_api,
    params_provider, block_generator, service_generator, buildings_params_provider, physical_objects_service
)
zones_service = FunctionalZonesService(urban_db_api)

CHAT_LA_PER_PERSON: float = float(base_params.la_per_person)


def _optional_env(key: str) -> str | None:
    """Read an optional env var. Config.get raises on absence; these keys are
    optional (conversational generation is a bolt-on), so read via os.getenv —
    Config() has already loaded the .env into the environment."""
    value = os.getenv(key)
    return value or None


def chat_llm_configured() -> bool:
    """True when an Ollama backend + chat model are configured."""
    return bool(_optional_env("Ollama_API")) and bool(_optional_env("Chat_Model"))


def build_ollama_chat_client(temperature: float | None = None) -> OllamaChatClient:
    """Construct a per-request Ollama chat client (caller owns its lifetime)."""
    return OllamaChatClient(
        _optional_env("Ollama_API") or "",
        default_model=_optional_env("Chat_Model") or "",
        temperature=0.3 if temperature is None else temperature,
    )


# --- Keycloak service token (outbound M2M auth) --------------------------------
#
# ChatStorage now accepts a service-account (client-credentials) token instead of
# the user's token; the end-user is identified via an ``X-User-Id`` header. The
# service token is obtained with the ``idu-service-auth`` library and shared for
# the whole process (created in the FastAPI lifespan, see ``app/main.py``).

_service_token_client: "KeycloakTokenClient | None" = None


def keycloak_service_configured() -> bool:
    """True when the Keycloak service-account credentials are configured."""
    return all(
        _optional_env(key)
        for key in ("KEYCLOAK_URL", "KEYCLOAK_REALM", "KEYCLOAK_CLIENT_ID", "KEYCLOAK_CLIENT_SECRET")
    )


def build_keycloak_token_config() -> "KeycloakTokenConfig":
    """Build the ``idu-service-auth`` config from environment variables."""
    from idu_service_auth import KeycloakTokenConfig

    return KeycloakTokenConfig(
        auth_server_url=config.get("KEYCLOAK_URL"),
        realm=config.get("KEYCLOAK_REALM"),
        client_id=config.get("KEYCLOAK_CLIENT_ID"),
        client_secret=config.get("KEYCLOAK_CLIENT_SECRET"),
        scope=_optional_env("KEYCLOAK_SCOPE"),
        background_refresh=True,
    )


def set_service_token_client(client: "KeycloakTokenClient | None") -> None:
    """Register the process-wide service token client (called from lifespan)."""
    global _service_token_client
    _service_token_client = client


def get_service_token_client() -> "KeycloakTokenClient | None":
    """Return the process-wide service token client, or None if not initialized."""
    return _service_token_client


def build_chat_storage_client() -> ChatStorageClient | None:
    """Construct a per-request ChatStorage client, or None when unavailable.

    Persistence requires both a ChatStorage URL and an initialized Keycloak
    service token client (ChatStorage is called with the service token). If the
    service token client isn't available, persistence is disabled rather than
    falling back to a user token.
    """
    base_url = _optional_env("ChatStorage_API")
    if not base_url:
        return None
    token_client = get_service_token_client()
    if token_client is None:
        logger.warning(
            "ChatStorage_API is set but the Keycloak service token client is not "
            "initialized; chat history persistence is disabled."
        )
        return None
    return ChatStorageClient(base_url, token_client)
