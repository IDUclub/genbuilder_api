from __future__ import annotations
import geopandas as gpd
from shapely.geometry import mapping, shape
import asyncio
from typing import Any, Dict
from loguru import logger
import json

from app.api.genbuilder_gateway import genbuilder_inference
from app.api.urbandb_api_gateway import urban_db_api
from app.logic.centroids_normalization import snapper

from app.logic.postprocessing import (
    density_isolines,
    grid_generator,
    buildings_generator,
    attributes_calculator)

class Genbuilder():
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
    async def infer_centroids_for_gdf(
        self,
        gdf: gpd.GeoDataFrame,
        infer_params: Dict[str, Any],
        la_target: dict[str, float],
        floors_avg: dict[str, float],
    ) -> gpd.GeoDataFrame:
        
        try:
            zones = (
                gdf["zone"].astype(str).str.strip().replace({"None": ""}).tolist()
                if "zone" in gdf.columns else []
            )
        except Exception:
            zones = []

        if not zones:
            raise ValueError("В GeoDataFrame отсутствует столбец 'zone' или он пуст.")

        unique_zones = {z for z in zones if z}
        missing_la = [z for z in unique_zones if la_target.get(z) is None]
        missing_floors = [z for z in unique_zones if floors_avg.get(z) is None]

        if missing_la or missing_floors:
            parts = []
            if missing_la:
                parts.append(f"la_target отсутствует для зон: {sorted(missing_la)}")
            if missing_floors:
                parts.append(f"floors_avg отсутствует для зон: {sorted(missing_floors)}")
            raise ValueError(
                "Не заданы таргеты для некоторых зон. " +
                " | ".join(parts) +
                ". Добавьте значения в targets_by_zone прежде чем вызывать инференс."
            )
        
        def _to_float(x: Any, *, label: str, zone: str) -> float:
            try:
                return float(x)
            except Exception as e:
                raise TypeError(f"{label} для зоны '{zone}' должен быть числом, получено {x!r}") from e
        tasks = []
        for _, row in gdf.iterrows():
            zone_label = str(row.get("zone", "")).strip()
            if not zone_label:
                raise ValueError("У квартала отсутствует корректная метка 'zone'.")

            la_value = _to_float(la_target.get(zone_label), label="la_target", zone=zone_label)
            floors_value = _to_float(floors_avg.get(zone_label), label="floors_avg", zone=zone_label)

            feature = {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": {k: v for k, v in row.items() if k != "geometry"},
            }

            tasks.append(
                genbuilder_inference.generate_centroids(
                    feature=feature,
                    zone_label=zone_label,
                    infer_params=infer_params,
                    la_target=la_value,       
                    floors_avg=floors_value, 
                )
            )

        results = await asyncio.gather(*tasks)

        rows: list[dict] = []
        for r in results:
            for f in r.get("features", []):
                props = dict(f.get("properties") or {})
                props["geometry"] = shape(f["geometry"])
                rows.append(props)
        if not rows:
            raise ValueError("No centroids generated")
        else:
            gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)
            gdf.geometry = gdf.centroid

        return gdf

    async def run(self, targets_by_zone, infer_params, blocks = None, scenario_id = None, functional_zone_types = None):
        if blocks:
            gdf_blocks = blocks.model_dump()
            gdf_blocks = gpd.GeoDataFrame.from_features(gdf_blocks["features"])
            if gdf_blocks.crs is None:
                gdf_blocks.set_crs(32636, inplace=True)
        if scenario_id:
            gdf_blocks = await urban_db_api.get_territories_for_buildings(scenario_id)
            if functional_zone_types:
                gdf_blocks = gdf_blocks[gdf_blocks['zone'].isin(functional_zone_types)]
            gdf_blocks.to_crs(32636, inplace=True)
        logger.info(f"{gdf_blocks}")
        centroids = await self.infer_centroids_for_gdf(gdf_blocks, infer_params, targets_by_zone['la_target'], targets_by_zone['floors_avg'])
        logger.info(f"Centroids generated, {len(centroids)} total")
        snapper_result = snapper.run(centroids, gdf_blocks)
        logger.info("Centroids snapped")
        centroids = snapper_result['centroids']
        midline = snapper_result['midline']
        empty_grid = grid_generator.make_grid_for_blocks(
            blocks_gdf=gdf_blocks,
            cell_size_m=10, 
            midlines=gpd.GeoSeries([midline], crs=gdf_blocks.crs),
            block_id_col="block_id",
            offset_m=10.0
        )
        isolines = density_isolines.build(gdf_blocks, centroids, zone_id_col='zone')
        logger.info("isolines generated")
        logger.info(f"Isolines {isolines}")
        grid = grid_generator.fit_transform(empty_grid, isolines)
        logger.info("Grid created")
        buildings = buildings_generator.fit_transform(grid, gdf_blocks, zone_name_aliases=['zone'])
        logger.info("Buildings generated")
        logger.info(f"{buildings[buildings.is_living == False]}")
        buildings = attributes_calculator.fit_transform(buildings, gdf_blocks, targets_by_zone)["buildings"]
        logger.info("Buildings attributes generated")
        buildings = buildings[['service', 'living_area', 'floors_count', 'geometry']]
        return json.loads(buildings.to_json())

builder = Genbuilder()