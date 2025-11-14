from iduconfig import Config

from app.logic.logger_setup import setup_logger

from app.api.urbandb_api_gateway import UrbanDBAPI
from app.api.genbuilder_gateway import GenbuilderInferenceAPI

from app.logic.centroids_normalization import Snapper
from app.logic.postprocessing.generation_params import GenParams, ParamsProvider
from app.logic.postprocessing.grid_operations import GridOperations
from app.logic.postprocessing.shapes_library import ShapesLibrary
from app.logic.postprocessing.site_planner import SitePlanner
from app.logic.postprocessing.buildings_postprocessing import BuildingsPostProcessor
from app.logic.postprocessing.attributes_calculation import BuildingAttributes
from app.logic.postprocessing.isolines import DensityIsolines
from app.logic.postprocessing.built_grid import GridGenerator
from app.logic.postprocessing.buildings_generation import BuildingGenerator
from app.logic.postprocessing.services_generation import ServiceGenerator
from app.logic.generation import Genbuilder

config = Config()
setup_logger(config, log_level="INFO")

urban_db_api = UrbanDBAPI(config)
genbuilder_inference_api = GenbuilderInferenceAPI(config)

base_params = GenParams()
params_provider = ParamsProvider(base_params)

snapper = Snapper()
attributes_calculator = BuildingAttributes()

grid_operations = GridOperations(params_provider)
shapes_library = ShapesLibrary(params_provider)
buildings_postprocessor = BuildingsPostProcessor(grid_operations, params_provider)
planner = SitePlanner(grid_operations, shapes_library, params_provider)
buildings_generator = BuildingGenerator(grid_operations, buildings_postprocessor, params_provider)
service_generator = ServiceGenerator(shapes_library, planner, params_provider)
grid_generator = GridGenerator(params_provider)
density_isolines = DensityIsolines()

builder = Genbuilder(
    config, urban_db_api, genbuilder_inference_api,
    snapper, density_isolines, grid_generator,
    buildings_generator, service_generator, attributes_calculator,
    params_provider
)
