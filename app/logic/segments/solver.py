from __future__ import annotations

from typing import Any, Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger

from app.logic.building_capacity_optimizer import CapacityOptimizer
from app.logic.building_params import BuildingParams
from app.logic.segments.context import BlockSegmentsContextBuilder
from app.logic.segments.capacity_calculator import (
    SegmentCapacityCalculator,
    SegmentCapacityResult,
)
from app.logic.segments.scenario_search import BlockScenarioSearch
from app.logic.segments.plots_allocator import (
    BlockPlotsAllocator,
    BlockPlotsAllocationResult,
)


class BlockSolver:
    """
    High-level per-block solver that combines segment context, capacity
    calculation, scenario search, and plot allocation to produce final
    capacity metrics and updated rectangular segments for a single block.
    """

    def __init__(
        self,
        capacity_optimizer: CapacityOptimizer,
        segment_capacity_calculator: SegmentCapacityCalculator,
        scenario_search: BlockScenarioSearch,
        plots_allocator: BlockPlotsAllocator,
        segment_context_builder: BlockSegmentsContextBuilder
    ):
        self.capacity_optimizer = capacity_optimizer
        self.segment_capacity_calculator = segment_capacity_calculator
        self.scenario_search = scenario_search 
        self.plots_allocator = plots_allocator
        self.segment_context_builder = segment_context_builder

    def solve_block_with_segments(
        self,
        row: pd.Series,
        rects_block: gpd.GeoDataFrame,
        far: str,
        *,
        building_params: BuildingParams,
        la_ratio: float | None = None,
        mode: str = "residential",
        target_col: str = "la_target",
    ) -> Tuple[Dict[str, Any], gpd.GeoDataFrame]:
        block_id = row.name
        target_raw = row.get(target_col, 0.0)
        if rects_block.empty:
            logger.debug(
                f"[BlockSolver.solve_block_with_segments] block_id={block_id}, "
                f"mode={mode}: rects_block is EMPTY, return zeros/NaN"
            )
            rects_block = rects_block.copy()
            rects_block["plots_capacity"] = 0
            rects_block["plot_front"] = np.nan
            rects_block["plot_depth"] = np.nan

            return {
                "building_need": row.get("building_need", np.nan),
                "building_capacity": 0,
                "buildings_count": 0,
                "building_length": row.get("building_length", np.nan),
                "building_width": row.get("building_width", np.nan),
                "floors_count": row.get("floors_count", np.nan),
                "plot_front": row.get("plot_front", np.nan),
                "plot_side_used": row.get("plot_front", np.nan),
                "plot_depth": row.get("plot_depth", np.nan),
                "plot_area": row.get("plot_area", np.nan),
                "living_per_building": row.get("living_per_building", np.nan),
                "total_usable_area": row.get("total_usable_area", np.nan),
                "la_diff": row.get("la_diff", np.nan),
                "la_ratio": row.get("la_ratio", np.nan),
                "far_initial": row.get("far_initial", np.nan),
                "far_final": row.get("far_final", np.nan),
                "far_diff": row.get("far_diff", np.nan),
            }, rects_block
        try:
            target_val = float(target_raw)
        except (TypeError, ValueError):
            target_val = 0.0

        logger.debug(
            f"[BlockSolver.solve_block_with_segments] block_id={block_id}, mode={mode}: "
            f"target_val={target_val} (from column '{target_col}')"
        )

        if target_val <= 0:
            logger.debug(
                f"[BlockSolver.solve_block_with_segments] block_id={block_id}, "
                f"mode={mode}: target_val <= 0 (target_val={target_val}), "
                f"return zeros/NaN"
            )
            rects_block = rects_block.copy()
            rects_block["plots_capacity"] = 0
            rects_block["plot_front"] = np.nan
            rects_block["plot_depth"] = np.nan

            return {
                "building_need": 0,
                "building_capacity": 0,
                "buildings_count": 0,
                "building_length": row.get("building_length", np.nan),
                "building_width": row.get("building_width", np.nan),
                "floors_count": row.get("floors_count", np.nan),
                "plot_front": row.get("plot_front", np.nan),
                "plot_side_used": row.get("plot_front", np.nan),
                "plot_depth": np.nan,
                "plot_area": 0.0,
                "living_per_building": 0.0,
                "total_usable_area": 0.0,
                "la_diff": 0.0,
                "la_ratio": np.nan,
                "far_initial": row.get("far_initial", np.nan),
                "far_final": np.nan,
                "far_diff": np.nan,
            }, rects_block
        area_min, area_max, area_base = self.capacity_optimizer.get_plot_area_params(
            far,
            building_params,
        )
        plot_area_base = float(area_base)

        (
            base_L_idx,
            base_W_idx,
            base_H_idx,
            base_F_idx,
        ) = self.capacity_optimizer.pick_indices(
            far,
            building_params,
        )

        base_L = float(building_params.building_length_range[base_L_idx])
        base_W = float(building_params.building_width_range[base_W_idx])
        base_H = float(building_params.building_height[base_H_idx])
        base_F = float(building_params.plot_side[base_F_idx])

        logger.debug(
            f"[BlockSolver.solve_block_with_segments] block_id={block_id}, mode={mode}: "
            f"area_min={area_min}, area_max={area_max}, area_base={area_base}, "
            f"base_L={base_L}, base_W={base_W}, base_H={base_H}, base_F={base_F}"
        )
        eff_la_ratio = building_params.la_coef if la_ratio is None else la_ratio

        ctx = self.segment_context_builder.build(
            row=row,
            rects_block=rects_block.copy(),
            mode=mode,
            target_col=target_col,
            la_ratio=float(eff_la_ratio),
            plot_area_base=plot_area_base,
            base_dims=(base_L, base_W, base_H),
        )

        logger.debug(
            f"[BlockSolver.solve_block_with_segments] block_id={block_id}, mode={mode}: "
            f"ctx.target_val={ctx.target_val}, ctx.la_ratio={ctx.la_ratio}, "
            f"ctx.plot_area_base={ctx.plot_area_base}, ctx.far_initial={ctx.far_initial}"
        )
        capacity_result: SegmentCapacityResult = (
            self.segment_capacity_calculator.compute(
                ctx=ctx,
                building_params=building_params,
            )
        )
        max_cap = capacity_result.max_capacity

        logger.debug(
            f"[BlockSolver.solve_block_with_segments] block_id={block_id}, mode={mode}: "
            f"capacity_by_front_idx={capacity_result.capacity_by_front_idx}, "
            f"max_capacity={max_cap}"
        )

        if max_cap <= 0:
            logger.debug(
                f"[BlockSolver.solve_block_with_segments] block_id={block_id}, mode={mode}: "
                f"max_capacity <= 0, return zeros/NaN for plots"
            )
            rects_block = ctx.rects_block.copy()
            rects_block["plots_capacity"] = 0
            rects_block["plot_front"] = np.nan
            rects_block["plot_depth"] = np.nan

            far_initial = ctx.far_initial

            return {
                "building_need": row.get("building_need", np.nan),
                "building_capacity": 0,
                "buildings_count": 0,
                "building_length": row.get("building_length", np.nan),
                "building_width": row.get("building_width", np.nan),
                "floors_count": row.get("floors_count", np.nan),
                "plot_front": np.nan,
                "plot_side_used": np.nan,
                "plot_depth": np.nan,
                "plot_area": plot_area_base,
                "living_per_building": row.get("living_per_building", np.nan),
                "total_usable_area": 0.0,
                "la_diff": -float(target_val),
                "la_ratio": 0.0,
                "far_initial": float(far_initial) if far_initial == far_initial else np.nan,
                "far_final": np.nan,
                "far_diff": np.nan,
            }, rects_block
        scenario_result = self.scenario_search.search(
            ctx=ctx,
            building_params=building_params,
            capacity_result=capacity_result,
            base_indices=(base_L_idx, base_W_idx, base_H_idx, base_F_idx),
        )

        chosen = scenario_result.best_success or scenario_result.fallback

        if chosen is None:
            logger.debug(
                f"[BlockSolver.solve_block_with_segments] block_id={block_id}, "
                f"mode={mode}: chosen is None (no best_success and no fallback) "
                f"-> return zeros/NaN for plots"
            )
            rects_block = ctx.rects_block.copy()
            rects_block["plots_capacity"] = 0
            rects_block["plot_front"] = np.nan
            rects_block["plot_depth"] = np.nan

            far_initial = ctx.far_initial

            return {
                "building_need": row.get("building_need", np.nan),
                "building_capacity": 0,
                "buildings_count": 0,
                "building_length": row.get("building_length", np.nan),
                "building_width": row.get("building_width", np.nan),
                "floors_count": row.get("floors_count", np.nan),
                "plot_front": np.nan,
                "plot_side_used": np.nan,
                "plot_depth": np.nan,
                "plot_area": plot_area_base,
                "living_per_building": row.get("living_per_building", np.nan),
                "total_usable_area": 0.0,
                "la_diff": -float(target_val),
                "la_ratio": 0.0,
                "far_initial": float(far_initial) if far_initial == far_initial else np.nan,
                "far_final": np.nan,
                "far_diff": np.nan,
            }, rects_block
        allocation: BlockPlotsAllocationResult = self.plots_allocator.allocate(
            ctx=ctx,
            chosen=chosen,
        )

        return allocation.block_result, allocation.rects_block
