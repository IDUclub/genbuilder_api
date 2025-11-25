from __future__ import annotations

import asyncio

import pandas as pd
from loguru import logger

from genbuilder_api.app.logic.building_generation.building_capacity_optimizer import CapacityOptimizer
from app.logic.building_generation.maximum_inscribed_rectangle import MIR
from app.logic.building_generation.segments import SegmentsAllocator
from app.logic.building_generation.plots import PlotsGenerator
from app.logic.building_generation.buildings import BuildingsGenerator
from app.logic.postprocessing.generation_params import GenParams, ParamsProvider

class ResidentialGenBuilder:

    def __init__(
        self,
        building_capacity_optimizer: CapacityOptimizer,
        max_rectangle_finder: MIR,
        segments_allocator: SegmentsAllocator,
        plots_generator: PlotsGenerator,
        buildings_generator: BuildingsGenerator, 
        params_provider: ParamsProvider
    ) -> None:
        self._params = params_provider
        self.building_capacity_optimizer = building_capacity_optimizer
        self.max_rectangle_finder = max_rectangle_finder
        self.segments_allocator = segments_allocator
        self.plots_generator = plots_generator
        self.buildings_generator = buildings_generator

    @property
    def generation_parameters(self) -> GenParams:
        return self._params.current()
    
    async def run(self, residential_la_target, 
            density_scenario, default_floors_group, residential_blocks):
        
        residential_blocks["la_target"] = residential_la_target * residential_blocks.geometry.area / residential_blocks.geometry.area.sum()
        residential_blocks["floors_group"] = residential_blocks.get("floors_group", default_floors_group)
        residential_blocks = self.building_capacity_optimizer.compute_blocks_for_gdf(residential_blocks, density_scenario)
        logger.info("Initial parameters calculated")

        segments = self.max_rectangle_finder.pack_inscribed_rectangles_for_gdf(
            residential_blocks,
            step=self.generation_parameters.rectangle_finder_step,
            min_side=self.generation_parameters.minimal_rectangle_side,
            n_jobs=self.generation_parameters.jobs_number,
        )
        logger.info("Maximum inscribed rectangles found")

        residential_blocks, segments = self.segments_allocator.update_blocks_with_segments(
            residential_blocks,
            segments,
            far=density_scenario,
        )
        logger.info("Block stats updated")

        plots = self.plots_generator.generate_plots(segments)
        logger.info("Plots created")
        buildings_gdf = self.buildings_generator.generate_buildings_from_plots(plots)
        logger.info("Residential building generated")
        return buildings_gdf