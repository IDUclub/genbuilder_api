from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import asyncio
import geopandas as gpd
import pandas as pd
from loguru import logger

from app.logic.building_capacity_optimizer import CapacityOptimizer
from app.logic.maximum_inscribed_rectangle import MIR
from app.logic.segments.segments import SegmentsAllocator
from app.logic.plots.plots import PlotsGenerator
from app.logic.buildings import BuildingsGenerator
from app.logic.generation_params import GenParams, ParamsProvider


class BlockGenerator:
    """
    Orchestrates the full block → segments → plots → buildings pipeline for
    residential, non-residential, and mixed zones, distributing targets by area,
    computing capacity, packing rectangles, and generating building footprints.
    """

    def __init__(
        self,
        building_capacity_optimizer: CapacityOptimizer,
        max_rectangle_finder: MIR,
        segments_allocator: SegmentsAllocator,
        plots_generator: PlotsGenerator,
        buildings_generator: BuildingsGenerator,
        params_provider: ParamsProvider,
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

    @staticmethod
    def _distribute_target_by_area(
        gdf: gpd.GeoDataFrame,
        total_target: float,
        target_col: str,
    ) -> gpd.GeoDataFrame:
        gdf = gdf.copy()
        gdf[target_col] = 0.0

        if total_target <= 0 or gdf.empty:
            return gdf

        areas = gdf.geometry.area
        total_area = float(areas.sum())
        if total_area <= 0:
            return gdf

        gdf[target_col] = total_target * (areas / total_area)
        return gdf

    @staticmethod
    def _distribute_target_by_area_per_zone(
        gdf: gpd.GeoDataFrame,
        target_by_zone: Dict[str, float],
        target_col: str,
    ) -> gpd.GeoDataFrame:
        gdf = gdf.copy()
        gdf[target_col] = 0.0

        if not target_by_zone or gdf.empty:
            return gdf

        for zone, total_target in target_by_zone.items():
            if total_target <= 0:
                continue
            mask = gdf["zone"] == zone
            if not mask.any():
                continue

            areas = gdf.loc[mask].geometry.area
            total_area = float(areas.sum())
            if total_area <= 0:
                continue

            gdf.loc[mask, target_col] = total_target * (areas / total_area)

        return gdf

    async def run(
        self,
        mode: str,
        *,
        blocks: gpd.GeoDataFrame,
        la_target: Optional[float] = None,
        density_scenario: Optional[str] = None,
        default_floor_group: Optional[str] = None,
        coverage_target: Optional[float] = None,
        coverage_target_by_zone: Optional[Dict[str, float]] = None,
        floors_avg_by_zone: Optional[Dict[str, float]] = None,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:

        mode = str(mode).lower()
        if mode not in {"residential", "non_residential", "mixed"}:
            raise ValueError(f"Unknown generation mode: {mode!r}")

        logger.debug(
            f"BlockGenerator.run: mode='{mode}', blocks={len(blocks)}, "
            f"la_target={la_target}, coverage_target={coverage_target}, "
            f"coverage_target_by_zone={coverage_target_by_zone}, "
            f"density_scenario={density_scenario}, "
            f"default_floor_group={default_floor_group}, "
            f"floors_avg_by_zone={floors_avg_by_zone}"
        )

        blocks_gdf = blocks.copy()

        if floors_avg_by_zone:
            blocks_gdf["floors_avg"] = blocks_gdf["zone"].map(
                lambda z: float(floors_avg_by_zone.get(z, 0.0))
            )

        if mode == "residential":
            la_total = float(la_target or 0.0)
            blocks_gdf = self._distribute_target_by_area(
                blocks_gdf, la_total, target_col="la_target"
            )
            default_fg = default_floor_group or "medium"
            if "floors_group" not in blocks_gdf.columns:
                blocks_gdf["floors_group"] = default_fg
            else:
                blocks_gdf["floors_group"] = blocks_gdf["floors_group"].fillna(
                    default_fg
                )

            far_scenario = density_scenario or "min"
            logger.debug(
                f"BlockGenerator.run[residential]: blocks={len(blocks_gdf)}, "
                f"la_total={la_total}, far='{far_scenario}', "
                f"default_floor_group='{default_fg}'"
            )

            target_col = "la_target"

        elif mode == "non_residential":
            cov_by_zone = coverage_target_by_zone or {}
            blocks_gdf = self._distribute_target_by_area_per_zone(
                blocks_gdf, cov_by_zone, target_col="functional_target"
            )
            far_scenario = "mean"
            logger.debug(
                f"BlockGenerator.run[non_residential]: blocks={len(blocks_gdf)}, "
                f"coverage_target_by_zone={cov_by_zone}, far='{far_scenario}'"
            )

            target_col = "functional_target"

        else: 
            la_total = float(la_target or 0.0)
            cov_total = float(coverage_target or 0.0)

            blocks_gdf = self._distribute_target_by_area(
                blocks_gdf, la_total, target_col="la_target"
            )
            blocks_gdf = self._distribute_target_by_area(
                blocks_gdf, cov_total, target_col="functional_target"
            )

            default_fg = default_floor_group or "high"
            if "floors_group" not in blocks_gdf.columns:
                blocks_gdf["floors_group"] = default_fg
            else:
                blocks_gdf["floors_group"] = blocks_gdf["floors_group"].fillna(
                    default_fg
                )

            far_scenario = density_scenario or "min"
            logger.debug(
                f"BlockGenerator.run[mixed]: blocks={len(blocks_gdf)}, "
                f"la_total={la_total}, coverage_total={cov_total}, "
                f"far='{far_scenario}', default_floor_group='{default_fg}'"
            )
            target_col = "la_target"

        if not (
            (
                mode == "residential"
                and (blocks_gdf.get("la_target", pd.Series(0)).sum() > 0)
            )
            or (
                mode == "non_residential"
                and (blocks_gdf.get("functional_target", pd.Series(0)).sum() > 0)
            )
            or (
                mode == "mixed"
                and (
                    blocks_gdf.get("la_target", pd.Series(0)).sum() > 0
                    or blocks_gdf.get("functional_target", pd.Series(0)).sum() > 0
                )
            )
        ):
            logger.warning(
                f"BlockGenerator.run: mode='{mode}' -> no positive targets "
                f"after distribution, returning empty outputs"
            )
            empty = gpd.GeoDataFrame(
                columns=list(blocks_gdf.columns),
                geometry="geometry",
                crs=blocks_gdf.crs,
            )
            return empty, empty, empty

        blocks_with_capacity = await asyncio.to_thread(
            self.building_capacity_optimizer.compute_blocks_for_gdf,
            blocks_gdf,
            far=far_scenario,
            target_col=target_col,
            mode=mode,
        )

        logger.debug(
            f"BlockGenerator.run[{mode}]: capacity computed for "
            f"{len(blocks_with_capacity)} blocks"
        )
        segments = await asyncio.to_thread(
            self.max_rectangle_finder.pack_inscribed_rectangles_for_gdf,
            blocks_with_capacity,
            step=self.generation_parameters.rectangle_finder_step,
            min_side=self.generation_parameters.minimal_rectangle_side,
            n_jobs=self.generation_parameters.jobs_number,
        )

        logger.debug(
            f"BlockGenerator.run[{mode}]: segments generated, count={len(segments)}"
        )
        blocks_final, segments_final = await asyncio.to_thread(
            self.segments_allocator.update_blocks_with_segments,
            blocks_with_capacity,
            segments,
            far=far_scenario,
            mode=mode,
            target_col=target_col,
        )

        logger.debug(
            f"BlockGenerator.run[{mode}]: blocks and segments updated "
            f"(blocks={len(blocks_final)}, segments={len(segments_final)})"
        )
        plots = await asyncio.to_thread(
            self.plots_generator.generate_plots,
            segments_final,
            mode=mode,
            target_col=target_col,
        )

        logger.debug(
            f"BlockGenerator.run[{mode}]: plots generated, count={len(plots)}"
        )
        buildings_gdf = await asyncio.to_thread(
            self.buildings_generator.generate_buildings_from_plots,
            plots,
            mode=mode,
        )

        logger.debug(
            f"BlockGenerator.run[{mode}]: buildings generated, count={len(buildings_gdf)}"
        )

        return blocks_final, plots, buildings_gdf
