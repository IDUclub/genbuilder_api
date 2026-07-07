import os

from iduconfig import Config

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


def build_chat_storage_client() -> ChatStorageClient | None:
    """Construct a per-request ChatStorage client, or None when not configured."""
    base_url = _optional_env("ChatStorage_API")
    return ChatStorageClient(base_url) if base_url else None
