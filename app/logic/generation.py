from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, List

import geopandas as gpd
from loguru import logger
from shapely.geometry import mapping, shape
from iduconfig import Config

from app.schema.dto import BlockFeatureCollection
from app.exceptions.http_exception_wrapper import http_exception

from app.dependencies import GenbuilderInferenceAPI
from app.dependencies import UrbanDBAPI
from app.logic.centroids_normalization import Snapper
from app.logic.postprocessing.buildings_generation import BuildingGenerator
from app.logic.postprocessing.attributes_calculation import BuildingAttributes
from app.logic.postprocessing.isolines import DensityIsolines
from app.logic.postprocessing.built_grid import GridGenerator
from app.logic.postprocessing.generation_params import GenParams, ParamsProvider


class Genbuilder:
    """
    Asynchronous orchestrator for the full **urban block generation pipeline** —
    from centroid inference to final building generation and attribute assignment.

    **Pipeline overview:**
        1. **Centroid inference**
           Uses the `genbuilder_inference` module to generate candidate building centroids
           for each functional zone, based on area and height targets (`la_target`, `floors_avg`).

        2. **Centroid snapping**
           Aligns centroids to the block’s “second ring” midline using `Snapper`.

        3. **Grid generation**
           Constructs a regular cell grid for each block via `GridGenerator.make_grid_for_blocks`.

        4. **Density isolines**
           Builds smooth isolines of centroid density inside each block using `DensityIsolines.build`.

        5. **Grid tagging**
           Tags grid cells by isoline level and adjacency rules (`GridGenerator.fit_transform`).

        6. **Building generation**
           Converts filled cells into building polygons using `BuildingsGenerator.fit_transform`.

        7. **Attribute calculation**
           Calculates floors, living area, and distribution coefficients via
           `BuildingAttributes.fit_transform`.

    **Main entry points:**
        - `infer_centroids_for_gdf(gdf, infer_params, la_target, floors_avg)` —
          Infers centroids asynchronously for each functional zone within the provided GeoDataFrame.

        - `run(targets_by_zone, infer_params, blocks=None, scenario_id=None, functional_zone_types=None)` —
          Executes the complete end-to-end generative pipeline, returning a serialized
          GeoJSON of generated buildings.

    **Input expectations:**
        - All geometries must be in a **projected CRS (meters)**, e.g. `EPSG:32636`.
        - The `targets_by_zone` dictionary must include two keys:
            * `'la_target'`: total living area per zone (`{zone: float}`)
            * `'floors_avg'`: average floor count per zone (`{zone: float}`)
        - The `blocks` argument must be a `BlockFeatureCollection`
          or another GeoJSON-compatible FeatureCollection.
        - Alternatively, `scenario_id` may be provided to automatically load blocks
          from the UrbanDB API (`urban_db_api.get_territories_for_buildings`).
    """
    def __init__(self, config: Config, urban_api: UrbanDBAPI, genbuilder_inference_api: GenbuilderInferenceAPI, 
                snapper: Snapper, density_isolines: DensityIsolines, grid_generator: GridGenerator, 
                buildings_generator: BuildingGenerator, attributes_calculator: BuildingAttributes, params_provider: ParamsProvider):
        self.config = config
        self.urban_api = urban_api
        self.genbuilder_inference_api = genbuilder_inference_api
        self.snapper = snapper
        self.density_isolines = density_isolines
        self.grid_generator = grid_generator
        self.buildings_generator = buildings_generator
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
        generation_parameters_override: dict | None = None
    ):
        base_parameters = self.generation_parameters.current()
        if generation_parameters_override:
            new_parameters = base_parameters.patched(generation_parameters_override)
        else:
            base_parameters
        if blocks:
            gdf_blocks = blocks.model_dump()
            gdf_blocks = gpd.GeoDataFrame.from_features(gdf_blocks["features"])
            if gdf_blocks.crs is None:
                gdf_blocks.set_crs(32636, inplace=True)
        if scenario_id:
            gdf_blocks = await self.urban_api.get_territories_for_buildings(scenario_id, year, source)
            if functional_zone_types:
                gdf_blocks = gdf_blocks[gdf_blocks["zone"].isin(functional_zone_types)]
            if len(gdf_blocks) == 0:
                raise http_exception(404, f"No blocks for zone(s): {', '.join(functional_zone_types)}")
            gdf_blocks.to_crs(32636, inplace=True)

        with self.generation_parameters.override(new_parameters):
            if blocks:
                gdf_blocks = blocks.model_dump()
                gdf_blocks = gpd.GeoDataFrame.from_features(gdf_blocks["features"])
                if gdf_blocks.crs is None:
                    gdf_blocks.set_crs(32636, inplace=True)
            if scenario_id:
                gdf_blocks = await self.urban_api.get_territories_for_buildings(scenario_id, year, source)
                if functional_zone_types:
                    gdf_blocks = gdf_blocks[gdf_blocks["zone"].isin(functional_zone_types)]
                gdf_blocks.to_crs(32636, inplace=True)

            if gdf_blocks.zone.isna().any():
                raise ValueError("Input blocks got empty zone values")

            centroids = await self.infer_centroids_for_gdf( 
                gdf_blocks,
                infer_params,
                targets_by_zone["la_target"],
                targets_by_zone["floors_avg"])
            logger.info(f"Centroids generated, {len(centroids)} total")
            snapper_result = await asyncio.to_thread(self.snapper.run, centroids, gdf_blocks)
            logger.info("Centroids snapped")
            centroids = snapper_result["centroids"]
            midline = snapper_result["midline"]
            empty_grid = await asyncio.to_thread(self.grid_generator.make_grid_for_blocks,
                blocks_gdf=gdf_blocks,
                midlines=gpd.GeoSeries([midline], crs=gdf_blocks.crs),
                block_id_col="block_id")
            isolines = await asyncio.to_thread(self.density_isolines.build, gdf_blocks, centroids, zone_id_col="zone")
            logger.info("isolines generated")
            grid = await asyncio.to_thread(self.grid_generator.fit_transform, empty_grid, isolines) 
            logger.info("Grid created")
            buildings = await  self.buildings_generator.fit_transform(grid, gdf_blocks, zone_name_aliases=["zone"])
            logger.info("Buildings generated")
            attributes_result = await asyncio.to_thread(self.attributes_calculator.fit_transform, buildings, gdf_blocks, targets_by_zone
            )
            buildings = attributes_result["buildings"]
            logger.info("Buildings attributes generated")
            buildings = buildings[["living_area", "floors_count", "service", "capacity", "geometry"]].to_crs(4326)
            return json.loads(buildings.to_json())