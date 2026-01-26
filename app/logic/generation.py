from __future__ import annotations

import json
import asyncio
from typing import Any, Dict, Optional, List

import geopandas as gpd
import pandas as pd
from loguru import logger
from iduconfig import Config

from app.schema.dto import BlockFeatureCollection
from app.dependencies import UrbanDBAPI
from app.logic.generation_params import ParamsProvider

from app.logic.building_params import (
    BuildingGenParams,
    BuildingParamsProvider,
)
from app.logic.block_generator import BlockGenerator
from app.logic.service_generation import (
    ServiceGenerator,
)
from app.logic.restrictions import check_buildings_setbacks

class Genbuilder:
    """
    Async orchestrator for the full Genbuilder pipeline that loads or receives
    urban blocks, runs residential / non-residential / mixed generation flows
    with optional parameter overrides and service normatives, and returns a
    unified GeoJSON FeatureCollection of buildings with key attributes.
    """

    def __init__(
        self,
        config: Config,
        urban_api: UrbanDBAPI,
        params_provider: ParamsProvider,
        residential_buildings_generator: BlockGenerator,
        residential_service_generator: ServiceGenerator,
        buildings_params_provider: BuildingParamsProvider,
    ):
        self.config = config
        self.urban_api = urban_api
        self.generation_parameters = params_provider
        self.residential_buildings_generator = residential_buildings_generator
        self.residential_service_generator = residential_service_generator
        self.buildings_generation_parameters = buildings_params_provider

    async def run(
        self,
        targets_by_zone: Dict[str, Dict[str, float]],
        token: str = None,
        blocks: Optional[BlockFeatureCollection] = None,
        scenario_id: Optional[int] = None,
        year: Optional[int] = None,
        source: Optional[str] = None,
        functional_zone_types: Optional[List[str]] = None,
        generation_parameters_override: dict | None = None,
        buildings_parameters_override: dict | None = None,
    ):
        base_parameters = self.generation_parameters.current()
        new_parameters = (
            base_parameters.patched(generation_parameters_override)
            if generation_parameters_override
            else base_parameters
        )

        base_building_parameters: BuildingGenParams = (
            self.buildings_generation_parameters.current()
        )
        new_building_parameters = (
            base_building_parameters.patched(buildings_parameters_override)
            if buildings_parameters_override
            else base_building_parameters
        )
        if blocks is not None:
            dumped = blocks.model_dump()
            gdf_blocks = await asyncio.to_thread(
                gpd.GeoDataFrame.from_features, dumped["features"]
            )
            logger.info(
                f"Genbuilder.run: using blocks from request, count={len(gdf_blocks)}"
            )
        else:
            gdf_blocks = await self.urban_api.get_territories_for_buildings(
                scenario_id, year, source, token
            )
            logger.info(
                f"Genbuilder.run: loaded blocks from UrbanDB, count={len(gdf_blocks)}"
            )
            if functional_zone_types:
                before = len(gdf_blocks)
                gdf_blocks = gdf_blocks[gdf_blocks["zone"].isin(functional_zone_types)]
                logger.info(
                    f"Genbuilder.run: filtered by functional_zone_types={functional_zone_types}, "
                    f"before={before}, after={len(gdf_blocks)}"
                )

        if gdf_blocks.crs is None:
            gdf_blocks.set_crs("EPSG:4326", inplace=True)

        if gdf_blocks.empty:
            logger.warning("Genbuilder.run: no blocks to process, returning empty FC")
            empty = gpd.GeoDataFrame(
                columns=[
                    "floors_count",
                    "living_area",
                    "functional_area",
                    "building_area",
                    "service",
                    "zone",
                    "geometry",
                ],
                geometry="geometry",
                crs="EPSG:4326",
            )
            empty_json = await asyncio.to_thread(empty.to_json)
            return json.loads(empty_json)

        utm = await asyncio.to_thread(gdf_blocks.estimate_utm_crs)
        gdf_blocks = await asyncio.to_thread(gdf_blocks.to_crs, utm)

        res_blocks = gdf_blocks[gdf_blocks["zone"] == "residential"].copy()

        mixed_blocks = gdf_blocks[
            gdf_blocks["zone"].isin(["business", "unknown"])
        ].copy()

        nonres_blocks = gdf_blocks[
            gdf_blocks["zone"].isin(["industrial", "transport", "special"])
        ].copy()

        ignored_blocks = gdf_blocks[
            ~gdf_blocks["zone"].isin(
                [
                    "residential",
                    "business",
                    "unknown",
                    "industrial",
                    "transport",
                    "special",
                ]
            )
        ].copy()

        logger.info(
            f"Genbuilder.run: blocks split by zone: "
            f"residential={len(res_blocks)}, mixed={len(mixed_blocks)}, "
            f"non_residential={len(nonres_blocks)}, ignored={len(ignored_blocks)}"
        )
        la_by_zone: Dict[str, float] = {}
        residents_by_zone: Dict[str, float] = (
                targets_by_zone.get("residents", {}) or {}
        )
        if residents_by_zone:
            la_per_person = float(new_parameters.la_per_person)
            for zone, residents in residents_by_zone.items():
                residents_val = float(residents or 0.0)
                if residents_val > 0:
                    la_by_zone[zone] = residents_val * la_per_person
            logger.info(
                "Genbuilder.run: converted residents to la_target using "
                f"la_per_person={la_per_person}, residents_by_zone={residents_by_zone}"
            )
        coverage_by_zone: Dict[str, float] = (
            targets_by_zone.get("coverage_area", {}) or {}
        )
        floors_avg_by_zone: Dict[str, float] = (
            targets_by_zone.get("floors_avg", {}) or {}
        )
        density_by_zone: Dict[str, str] = (
            targets_by_zone.get("density_scenario", {}) or {}
        )

        default_fg_by_zone: Dict[str, str] = (
            targets_by_zone.get("default_floor_group") or {}
        )
        res_la_target = float(la_by_zone.get("residential", 0.0))
        res_density_scenario = str(density_by_zone.get("residential", "min"))
        res_default_fg = str(default_fg_by_zone.get("residential", "medium"))

        logger.info(
            f"Genbuilder.run: residential group -> blocks={len(res_blocks)}, "
            f"la_target={res_la_target}, "
            f"density_scenario={res_density_scenario}, "
            f"default_floor_group='{res_default_fg}'"
        )
        nonres_coverage_by_zone: Dict[str, float] = {
            z: float(coverage_by_zone.get(z, 0.0))
            for z in ("industrial", "transport", "special")
        }
        total_nonres_cov_target = float(sum(nonres_coverage_by_zone.values()))

        floors_avg_nonres: Dict[str, float] = {
            z: float(floors_avg_by_zone[z])
            for z in nonres_coverage_by_zone
            if z in floors_avg_by_zone
        }
        logger.info(
            f"Genbuilder.run: non-residential group -> blocks={len(nonres_blocks)}, "
            f"coverage_by_zone={nonres_coverage_by_zone}, "
            f"total_coverage_target={total_nonres_cov_target}, "
            f"floors_avg_nonres={floors_avg_nonres}"
        )
        mixed_la_target = float(
            la_by_zone.get("business", 0.0) + la_by_zone.get("unknown", 0.0)
        )
        mixed_cov_target = float(
            coverage_by_zone.get("business", 0.0)
            + coverage_by_zone.get("unknown", 0.0)
        )

        mixed_density_scenario = str(
            density_by_zone.get("business")
            or density_by_zone.get("unknown")
            or "min"
        )
        mixed_default_fg = str(
            default_fg_by_zone.get("business")
            or default_fg_by_zone.get("unknown")
            or "high"
        )

        floors_avg_mixed: Dict[str, float] = {
            z: float(floors_avg_by_zone[z])
            for z in ("business", "unknown")
            if z in floors_avg_by_zone
        }

        logger.info(
            f"Genbuilder.run: mixed group (business+unknown) -> blocks={len(mixed_blocks)}, "
            f"la_target={mixed_la_target}, coverage_target={mixed_cov_target}, "
            f"density_scenario={mixed_density_scenario}, "
            f"default_floor_group='{mixed_default_fg}', "
            f"floors_avg_mixed={floors_avg_mixed}"
        )

        service_normatives = None
        if scenario_id is not None:
            territory_id = await self.urban_api.get_territory_by_scenario(scenario_id, token)
            service_normatives = await self.urban_api.get_normatives_for_territory(
                territory_id, token
            )
            logger.info(
                f"Genbuilder.run: loaded service normatives for territory_id={territory_id}"
            )
        else:
            logger.warning(
                "Genbuilder.run: scenario_id is None, service normatives are not loaded"
            )

        res_blocks_out = res_plots = res_buildings = None
        nonres_blocks_out = nonres_plots = nonres_buildings = None
        mixed_blocks_out = mixed_plots = mixed_buildings = None
        residential_services = None

        with self.generation_parameters.override(new_parameters):
            with self.buildings_generation_parameters.override(
                new_building_parameters
            ):
                if len(res_blocks) > 0 and res_la_target > 0:
                    logger.info(
                        "Genbuilder.run: starting residential generation pipeline"
                    )
                    res_blocks_out, res_plots, res_buildings = (
                        await self.residential_buildings_generator.run(
                            mode="residential",
                            blocks=res_blocks,
                            la_target=res_la_target,
                            density_scenario=res_density_scenario,
                            default_floor_group=res_default_fg,
                        )
                    )
                    logger.info(
                        f"Genbuilder.run: residential generation finished, "
                        f"blocks={len(res_blocks_out)}, plots={len(res_plots)}, "
                        f"buildings={len(res_buildings)}"
                    )

                    if (
                        service_normatives is not None
                        and res_blocks_out is not None
                        and res_plots is not None
                        and res_buildings is not None
                        and len(res_buildings) > 0
                    ):
                        residential_services = await self.residential_service_generator.generate_services(
                            res_blocks_out,
                            res_plots,
                            res_buildings,
                            service_normatives,
                            utm,
                        )
                        logger.info(
                            f"Genbuilder.run: residential services generated, "
                            f"count={len(residential_services)}"
                        )
                    else:
                        logger.info(
                            "Genbuilder.run: residential services not generated "
                            "(missing normatives or buildings)"
                        )
                else:
                    logger.info(
                        "Genbuilder.run: residential generation skipped "
                        "(no blocks or la_target <= 0)"
                    )
                if len(nonres_blocks) > 0 and total_nonres_cov_target > 0:
                    logger.info(
                        "Genbuilder.run: starting non-residential generation pipeline"
                    )
                    nonres_blocks_out, nonres_plots, nonres_buildings = (
                        await self.residential_buildings_generator.run(
                            mode="non_residential",
                            blocks=nonres_blocks,
                            coverage_target_by_zone=nonres_coverage_by_zone,
                            floors_avg_by_zone=floors_avg_nonres,
                        )
                    )
                    logger.info(
                        f"Genbuilder.run: non-residential generation finished, "
                        f"blocks={len(nonres_blocks_out)}, plots={len(nonres_plots)}, "
                        f"buildings={len(nonres_buildings)}"
                    )
                else:
                    logger.info(
                        "Genbuilder.run: non-residential generation skipped "
                        "(no blocks or coverage_target <= 0)"
                    )
                if (
                    len(mixed_blocks) > 0
                    and (mixed_la_target > 0 or mixed_cov_target > 0)
                ):
                    logger.info(
                        "Genbuilder.run: starting mixed generation pipeline "
                        "(business + unknown)"
                    )
                    mixed_blocks_out, mixed_plots, mixed_buildings = (
                        await self.residential_buildings_generator.run(
                            mode="mixed",
                            blocks=mixed_blocks,
                            la_target=mixed_la_target,
                            coverage_target=mixed_cov_target,
                            density_scenario=mixed_density_scenario,
                            default_floor_group=mixed_default_fg,
                            floors_avg_by_zone=floors_avg_mixed,
                        )
                    )
                    logger.info(
                        f"Genbuilder.run: mixed generation finished, "
                        f"blocks={len(mixed_blocks_out)}, plots={len(mixed_plots)}, "
                        f"buildings={len(mixed_buildings)}"
                    )
                else:
                    logger.info(
                        "Genbuilder.run: mixed generation skipped "
                        "(no blocks or both targets are 0)"
                    )

        frames: List[gpd.GeoDataFrame] = []

        if nonres_buildings is not None and len(nonres_buildings) > 0:
            frames.append(nonres_buildings)
        if mixed_buildings is not None and len(mixed_buildings) > 0:
            frames.append(mixed_buildings)
        if res_buildings is not None and len(res_buildings) > 0:
            frames.append(res_buildings)
        if residential_services is not None and len(residential_services) > 0:
            frames.append(residential_services)

        if not frames:
            logger.warning(
                "Genbuilder.run: no buildings generated in any zone group, "
                "returning empty FeatureCollection"
            )
            empty = gpd.GeoDataFrame(
                columns=[
                    "floors_count",
                    "living_area",
                    "functional_area",
                    "building_area",
                    "service",
                    "zone",
                    "geometry",
                ],
                geometry="geometry",
                crs="EPSG:4326",
            )
            empty_json = await asyncio.to_thread(empty.to_json)
            return json.loads(empty_json)

        concat_df = await asyncio.to_thread(pd.concat, frames, ignore_index=True)
        buildings_all = gpd.GeoDataFrame(
            concat_df,
            geometry="geometry",
            crs=utm,
        )

        if "living_area" not in buildings_all.columns:
            buildings_all["living_area"] = 0.0
        buildings_all["living_area"] = buildings_all["living_area"].fillna(0).round(0)

        if "functional_area" not in buildings_all.columns:
            buildings_all["functional_area"] = 0.0
        buildings_all["functional_area"] = buildings_all["functional_area"].fillna(0.0)

        if "floors_count" not in buildings_all.columns:
            buildings_all["floors_count"] = 0.0
        buildings_all["floors_count"] = buildings_all["floors_count"].fillna(0.0)

        footprint_area = buildings_all.geometry.area
        floors = buildings_all["floors_count"]
        buildings_all["building_area"] = footprint_area * floors

        if "service" in buildings_all.columns and "capacity" in buildings_all.columns:
            buildings_all["service"] = [
                [{service: capacity}]
                if (pd.notna(service) and pd.notna(capacity))
                else []
                for service, capacity in zip(
                    buildings_all["service"], buildings_all["capacity"]
                )
            ]
        else:
            if "service" not in buildings_all.columns:
                buildings_all["service"] = [[] for _ in range(len(buildings_all))]
        buildings_all = check_buildings_setbacks(buildings_all)
        buildings_all['building_area'] = buildings_all['building_area'].round(0)
        buildings_all = buildings_all[
            [
                "floors_count",
                "living_area",
                "building_area",
                "service",
                "broke_restriction_zone",
                "building_type",
                "geometry",
            ]
        ]
        buildings_all = await asyncio.to_thread(
            gpd.sjoin,
            buildings_all,
            gdf_blocks[["zone", "geometry"]],
            how="left",
            predicate="intersects",
        )
        buildings_all = buildings_all.drop(columns=["index_right"])
        if "zone" not in buildings_all.columns:
            buildings_all["zone"] = None

        res_mask = buildings_all["zone"] == "residential"
        nonres_mask = buildings_all["zone"].isin(
            ["industrial", "transport", "special"]
        )
        mixed_mask = buildings_all["zone"].isin(["business", "unknown"])
        buildings_all.loc[nonres_mask, "living_area"] = 0.0
        logger.info(
            f"Genbuilder.run: final zones distribution in buildings: "
            f"residential={res_mask.sum()}, mixed={mixed_mask.sum()}, "
            f"non_residential={nonres_mask.sum()}"
        )
        buildings_all = await asyncio.to_thread(buildings_all.to_crs, 4326)

        logger.info(
            f"Genbuilder.run: final buildings count={len(buildings_all)}, "
            f"crs={buildings_all.crs}"
        )

        return json.loads(buildings_all.to_json())
