from iduconfig import Config

from app.logic.logger_setup import setup_logger

from app.api.urbandb_api_gateway import UrbanDBAPI

from app.logic.generation_params import GenParams, ParamsProvider
from app.logic.building_capacity_optimizer import CapacityOptimizer
from app.logic.maximum_inscribed_rectangle import MIR
from app.logic.segments import SegmentsAllocator
from app.logic.plots.plots import PlotsGenerator
from app.logic.plots.plot_slicer import PlotSegmentSlicer
from app.logic.plots.plot_merge import PlotMerger
from app.logic.plots.plot_tuner import PlotTuner
from app.logic.buildings import ResidentialBuildingsGenerator
from app.logic.residential_generator import ResidentialGenBuilder
from app.logic.residential_service_generation import ResidentialServiceGenerator
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
segments_allocator = SegmentsAllocator(building_capacity_optimizer, buildings_params_provider)
plot_slicer = PlotSegmentSlicer()
plor_merger = PlotMerger()
plot_tuner = PlotTuner(params_provider, buildings_params_provider)
plots_generator = PlotsGenerator(params_provider, buildings_params_provider, building_capacity_optimizer, plot_slicer, plor_merger, plot_tuner)
residential_buildings_generator = ResidentialBuildingsGenerator()
residential_generator = ResidentialGenBuilder(building_capacity_optimizer, max_rectangle_finder, 
                    segments_allocator, plots_generator, residential_buildings_generator, params_provider)
residential_service_generator = ResidentialServiceGenerator(params_provider)

builder = Genbuilder(
    config, urban_db_api, 
    params_provider, residential_generator, residential_service_generator, buildings_params_provider
)
