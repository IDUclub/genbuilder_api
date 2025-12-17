from iduconfig import Config

from app.logic.logger_setup import setup_logger

from app.api.urbandb_api_gateway import UrbanDBAPI

from app.logic.generation_params import GenParams, ParamsProvider
from app.logic.building_capacity_optimizer import CapacityOptimizer
from app.logic.maximum_inscribed_rectangle import MIR
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

builder = Genbuilder(
    config, urban_db_api, 
    params_provider, block_generator, service_generator, buildings_params_provider
)
