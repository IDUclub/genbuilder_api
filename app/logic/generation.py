from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, List

import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely.geometry import mapping, shape
from iduconfig import Config

from app.schema.dto import BlockFeatureCollection
from app.exceptions.http_exception_wrapper import http_exception

from app.dependencies import GenbuilderInferenceAPI
from app.dependencies import UrbanDBAPI
from app.logic.centroids_normalization import Snapper
from app.logic.postprocessing.buildings_generation import BuildingGenerator
from app.logic.postprocessing.services_generation import ServiceGenerator
from app.logic.postprocessing.attributes_calculation import BuildingAttributes
from app.logic.postprocessing.isolines import DensityIsolines
from app.logic.postprocessing.built_grid import GridGenerator
from app.logic.postprocessing.generation_params import GenParams, ParamsProvider


class Genbuilder:
    """
    Asynchronous orchestrator for the full Genbuilder pipeline.

    Runs centroid inference and snapping, grid and isoline construction,
    residential building generation, attribute assignment and service generation.
    Use `run` to execute the full pipeline and get generated buildings.
    """
    def __init__(self, config: Config, urban_api: UrbanDBAPI, genbuilder_inference_api: GenbuilderInferenceAPI, 
                snapper: Snapper, density_isolines: DensityIsolines, grid_generator: GridGenerator, 
                buildings_generator: BuildingGenerator, service_generator: ServiceGenerator, attributes_calculator: BuildingAttributes, params_provider: ParamsProvider):
        self.config = config
        self.urban_api = urban_api
        self.genbuilder_inference_api = genbuilder_inference_api
        self.snapper = snapper
        self.density_isolines = density_isolines
        self.grid_generator = grid_generator
        self.buildings_generator = buildings_generator
        self.service_generator = service_generator
        self.attributes_calculator = attributes_calculator
        self.generation_parameters = params_provider

    async def infer_centroids_for_gdf(
        self,
        gdf: gpd.GeoDataFrame,
        infer_params: Dict[str, Any],
        la_target: dict[str, float],
        floors_avg: dict[str, float],
    ) -> gpd.GeoDataFrame:

        tasks = []
        for _, row in gdf.iterrows():
            zone_label = str(row.get("zone", "")).strip()
            la_value = la_target.get(zone_label)
            floors_value = floors_avg.get(zone_label)

            feature = {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": {k: v for k, v in row.items() if k != "geometry"},
            }

            tasks.append(
                self.genbuilder_inference_api.generate_centroids(
                    feature=feature,
                    zone_label=zone_label,
                    infer_params=infer_params,
                    la_target=la_value,
                    floors_avg=floors_value,
                )
            )

        results = await asyncio.gather(*tasks)

        rows: list[dict] = []
        for result in results:
            for feature in result.get("features", []):
                props = dict(feature.get("properties") or {})
                props["geometry"] = shape(feature["geometry"])
                rows.append(props)
        if not rows:
            raise http_exception(
                404,
                f"No centroids generated with current parameters"
            )
        else:
            gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)
            gdf.geometry = gdf.centroid

        return gdf

    async def run(
        self,
        targets_by_zone: Dict[str, Dict[str, float]],
        infer_params: Dict[str, Any],
        blocks: Optional[BlockFeatureCollection] = None,
        scenario_id: Optional[int] = None,
        year: Optional[int] = None,
        source: Optional[str] = None,
        functional_zone_types: Optional[List[str]] = None,
        generation_parameters_override: dict | None = None,
    ):
        base_parameters = self.generation_parameters.current()
        new_parameters = (
            base_parameters.patched(generation_parameters_override)
            if generation_parameters_override
            else base_parameters
        )
        if blocks is not None:
            dumped = blocks.model_dump()
            gdf_blocks = gpd.GeoDataFrame.from_features(dumped["features"])
        else:
            gdf_blocks = await self.urban_api.get_territories_for_buildings(scenario_id, year, source)
            if functional_zone_types:
                gdf_blocks = gdf_blocks[gdf_blocks["zone"].isin(functional_zone_types)]
        utm = gdf_blocks.estimate_utm_crs()
        gdf_blocks = gdf_blocks.to_crs(utm)
        with self.generation_parameters.override(new_parameters):
            centroids = await self.infer_centroids_for_gdf(
                gdf_blocks,
                infer_params,
                targets_by_zone["la_target"],
                targets_by_zone["floors_avg"],
            )
            logger.info(f"Centroids generated, {len(centroids)} total")
            snapper_result = await asyncio.to_thread(self.snapper.run, centroids, gdf_blocks)
            logger.info("Centroids snapped")
            centroids = snapper_result["centroids"]
            midline = snapper_result["midline"]
            empty_grid = await asyncio.to_thread(
                self.grid_generator.make_grid_for_blocks,
                blocks_gdf=gdf_blocks,
                midlines=gpd.GeoSeries([midline], crs=gdf_blocks.crs),
                block_id_col="block_id",
            )
            isolines = await asyncio.to_thread(
                self.density_isolines.build, gdf_blocks, centroids, zone_id_col="zone"
            )
            logger.info("Isolines generated")
            grid = await asyncio.to_thread(self.grid_generator.fit_transform, empty_grid, isolines)
            logger.info("Grid created")
            territory_id = await self.urban_api.get_territory_by_scenario(scenario_id)
            service_normatives = await self.urban_api.get_normatives_for_territory(territory_id)
            building_stage = await self.buildings_generator.fit_transform(
                grid, gdf_blocks, zone_name_aliases=["zone"]
            )

            cells_with_buildings = building_stage["cells"]         
            living_buildings = building_stage["buildings"] 
            logger.info("Residential buildings generated")
            attributes_result = await asyncio.to_thread(
                self.attributes_calculator.fit_transform,
                living_buildings,
                gdf_blocks,
                targets_by_zone,
            )
            living_buildings = attributes_result["buildings"]
            living_area_per_zone = attributes_result["allocated_by_zone"]
            logger.info("Residential building attributes generated")
            service_buildings = await self.service_generator.fit_transform(
                cells_with_buildings, living_area_per_zone, service_normatives
            )
            logger.info("Service buildings generated")
            buildings_all = pd.concat([living_buildings, service_buildings], ignore_index=True)
            buildings_all = buildings_all[['floors_count', 'living_area', 'service', 'capacity', 'geometry']]
            buildings_all['living_area'] = buildings_all['living_area'].fillna(0).round(0)
            buildings_all["service"] = [
                [{service: capacity}] if mask else []
                for mask, service, capacity in zip(buildings_all["service"].notna(), 
                                                   buildings_all["service"], buildings_all["capacity"])
            ]
            buildings_all = buildings_all.drop(columns=["capacity"]).to_crs(4326)
            return json.loads(buildings_all.to_json())