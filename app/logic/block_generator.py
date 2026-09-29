from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import asyncio
import math

import geopandas as gpd
import numpy as np
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

    # Rounds of moving buildings out of blocks where they do not fit.
    _MAX_PLACEMENT_ROUNDS = 10

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

    def _usable_per_block(
        self, gdf: gpd.GeoDataFrame, far: str, mode: str
    ) -> pd.Series:
        """Living area of one base-size building in each block (0 if unknown type)."""
        return pd.Series(
            [
                self.building_capacity_optimizer.usable_per_building_for_row(
                    row, far, mode
                )
                for _, row in gdf.iterrows()
            ],
            index=gdf.index,
            dtype=float,
        )

    @staticmethod
    def _largest_remainder(
        need: pd.Series, total: int, areas: pd.Series
    ) -> pd.Series:
        """Round fractional building counts to integers summing up to ``total``.

        Ties in the remainder go to the larger block.
        """
        counts = np.floor(need + 1e-9).astype(int)
        left = total - int(counts.sum())
        if left > 0:
            order = (
                pd.DataFrame({"rem": need - counts, "area": areas.reindex(need.index)})
                .sort_values(["rem", "area"], ascending=False)
                .index[:left]
            )
            counts.loc[order] += 1
        return counts

    @classmethod
    def _split_buildings_by_area(
        cls, gdf: gpd.GeoDataFrame, total_target: float, usable: pd.Series
    ) -> pd.Series:
        """Whole buildings per block for the living-area target of a zone group.

        Rounding a fractional per-block share up to one building in every block
        overshoots the target by ~one building per block. Instead the number of
        buildings is rounded once for the whole group and handed out to blocks by
        area share with the largest-remainder method, so the group exceeds the
        target by less than one building. Blocks left without a building get 0.
        """
        counts = pd.Series(0, index=gdf.index, dtype=int)
        if total_target <= 0 or gdf.empty:
            return counts
        areas = gdf.geometry.area.where(usable > 0, 0.0)
        total_area = float(areas.sum())
        if total_area <= 0:
            return counts
        # Blocks of a different building type may hold more or less living
        # area per building.
        need = (total_target * areas / total_area / usable.where(usable > 0)).fillna(0.0)
        total = int(math.ceil(float(need.sum()) - 1e-9))
        return cls._largest_remainder(need, total, areas)

    @staticmethod
    def _targets_from_counts(counts: pd.Series, usable: pd.Series) -> pd.Series:
        # Slightly below n × usable so the per-block ceil() downstream yields n.
        return (counts * usable - 1e-6).clip(lower=0.0)

    def _place_buildings(
        self,
        blocks_gdf: gpd.GeoDataFrame,
        segments: gpd.GeoDataFrame,
        counts: pd.Series,
        usable: pd.Series,
        total_target: float,
        *,
        far: str,
        mode: str,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Fit the allocated buildings into blocks, moving what does not fit.

        A block may be too small or oddly shaped for its share. Its shortfall is
        handed to blocks that still have room (again by area), so the zone group
        keeps the requested number of buildings. Blocks without a building skip
        the segment search — with a zero target it would fill them to capacity —
        and are returned unchanged with zero counts.
        """
        counts = counts.astype(int).copy()
        areas = blocks_gdf.geometry.area
        full: set = set()
        # Living area left over once every block is full; the plot tuner covers
        # it with larger / taller buildings, as it did before the whole-building
        # split.
        overflow = pd.Series(0.0, index=counts.index)

        for _ in range(self._MAX_PLACEMENT_ROUNDS):
            active = counts > 0
            blocks = blocks_gdf.loc[active].copy()
            blocks["la_target"] = (
                self._targets_from_counts(counts[active], usable[active])
                + overflow[active]
            )
            with_capacity = self.building_capacity_optimizer.compute_blocks_for_gdf(
                blocks, far=far, target_col="la_target", mode=mode
            )
            blocks_final, segments_final = (
                self.segments_allocator.update_blocks_with_segments(
                    with_capacity,
                    segments[segments["src_index"].isin(blocks.index)],
                    far=far,
                    mode=mode,
                    target_col="la_target",
                )
            )
            placed = (
                pd.to_numeric(blocks_final["buildings_count"], errors="coerce")
                .reindex(counts.index)
                .fillna(0)
                .astype(int)
            )
            short = (counts - placed).clip(lower=0)
            deficit = int(short.sum())
            if deficit == 0 or overflow.sum() > 0:
                break

            # Cap the blocks that ran out of room and move the rest elsewhere.
            full |= set(short.index[short > 0])
            counts = counts.where(short == 0, placed)
            free = ~counts.index.isin(list(full)) & (usable > 0).to_numpy()
            if not free.any():
                filled = areas.where(counts > 0, 0.0)
                left = total_target - float((counts * usable).sum())
                if left <= 0 or filled.sum() <= 0:
                    continue  # one more pass so capped blocks get matching targets
                overflow = left * filled / float(filled.sum())
                logger.info(
                    f"BlockGenerator[{mode}]: no room for {deficit} more buildings, "
                    f"{left:.0f} m² left to larger buildings in "
                    f"{int((counts > 0).sum())} blocks"
                )
                continue
            free_areas = areas[free]
            need = deficit * free_areas / float(free_areas.sum())
            counts.loc[free] += self._largest_remainder(need, deficit, free_areas)
            logger.debug(
                f"BlockGenerator[{mode}]: {deficit} buildings did not fit, "
                f"moved to {int(free.sum())} blocks with room"
            )

        idle = blocks_gdf.loc[counts <= 0].assign(
            la_target=0.0, building_capacity=0, buildings_count=0
        )
        if not idle.empty:
            # Keep empty blocks in the output: services may still be placed there.
            blocks_final = pd.concat([blocks_final, idle]).sort_index()
        logger.debug(
            f"BlockGenerator[{mode}]: {int(placed.sum())} buildings placed in "
            f"{int((placed > 0).sum())}/{len(blocks_gdf)} blocks"
        )
        return blocks_final, segments_final

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
        # Whole buildings per block for the living-area modes.
        building_counts: Optional[pd.Series] = None

        if floors_avg_by_zone:
            blocks_gdf["floors_avg"] = blocks_gdf["zone"].map(
                lambda z: float(floors_avg_by_zone.get(z, 0.0))
            )

        if mode == "residential":
            la_total = float(la_target or 0.0)
            default_fg = default_floor_group or "medium"
            if "floors_group" not in blocks_gdf.columns:
                blocks_gdf["floors_group"] = default_fg
            else:
                blocks_gdf["floors_group"] = blocks_gdf["floors_group"].fillna(
                    default_fg
                )

            far_scenario = density_scenario or "min"
            usable = self._usable_per_block(blocks_gdf, far_scenario, mode)
            building_counts = self._split_buildings_by_area(
                blocks_gdf, la_total, usable
            )
            blocks_gdf["la_target"] = self._targets_from_counts(building_counts, usable)
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

            default_fg = default_floor_group or "high"
            if "floors_group" not in blocks_gdf.columns:
                blocks_gdf["floors_group"] = default_fg
            else:
                blocks_gdf["floors_group"] = blocks_gdf["floors_group"].fillna(
                    default_fg
                )

            far_scenario = density_scenario or "min"
            usable = self._usable_per_block(blocks_gdf, far_scenario, mode)
            building_counts = self._split_buildings_by_area(
                blocks_gdf, la_total, usable
            )
            blocks_gdf["la_target"] = self._targets_from_counts(building_counts, usable)
            blocks_gdf = self._distribute_target_by_area(
                blocks_gdf, cov_total, target_col="functional_target"
            )
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
        if building_counts is not None and building_counts.sum() > 0:
            blocks_final, segments_final = await asyncio.to_thread(
                self._place_buildings,
                blocks_gdf,
                segments,
                building_counts,
                usable,
                la_total,
                far=far_scenario,
                mode=mode,
            )
        else:
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
