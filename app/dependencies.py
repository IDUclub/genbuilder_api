from iduconfig import Config

from app.logic.logger_setup import setup_logger

from app.api.urbandb_api_gateway import UrbanDBAPI
from app.api.genbuilder_gateway import GenbuilderInferenceAPI

from app.logic.centroids_normalization import Snapper
from app.logic.postprocessing import (BuildingAttributes,
                                      BuildingGenerator, DensityIsolines,
                                      GridGenerator, GenParams)
from app.logic.generation import Genbuilder

config = Config()
setup_logger(config, log_level="INFO")

urban_db_api = UrbanDBAPI(config)
genbuilder_inference_api = GenbuilderInferenceAPI(config)

generation_parameters = GenParams()

snapper = Snapper()
attributes_calculator = BuildingAttributes()
buildings_generator = BuildingGenerator(generation_parameters)
grid_generator = GridGenerator()
density_isolines = DensityIsolines()

builder = Genbuilder(config, urban_db_api, genbuilder_inference_api, 
                snapper, density_isolines, grid_generator, 
                buildings_generator, attributes_calculator)
