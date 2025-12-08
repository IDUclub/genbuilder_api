from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import geopandas as gpd
import numpy as np
from loguru import logger

from app.logic.segments.context import BlockSegmentsContext
from app.logic.segments.scenario_search import ChosenScenario


@dataclass
class BlockPlotsAllocationResult:
    block_result: Dict[str, Any]
    rects_block: gpd.GeoDataFrame


class BlockPlotsAllocator:
    """
    Allocates building plots across a block’s segments for a chosen (L, W, H, F)
    scenario, computing per-segment plots_capacity/front/depth and block-level
    capacity metrics returned as an updated segments GeoDataFrame plus summary.
    """

    def allocate(
        self,
        ctx: BlockSegmentsContext,
        chosen: ChosenScenario,
    ) -> BlockPlotsAllocationResult:
        rects_block = ctx.rects_block.copy()

        L = float(chosen.L)
        W = float(chosen.W)
        H = float(chosen.H)
        F = float(chosen.F)

        building_need_val = int(chosen.building_need)
        building_capacity = int(chosen.building_capacity)
        buildings_count = int(chosen.buildings_count)
        usable_one = float(chosen.living_per_building)
        total_usable_area = float(chosen.total_usable_area)
        far_final = float(chosen.far_final)

        target_val = float(ctx.target_val) if ctx.target_val == ctx.target_val else 0.0
        far_initial = float(ctx.far_initial) if ctx.far_initial == ctx.far_initial else np.nan
        plot_area = float(ctx.plot_area_base)

        la_diff = total_usable_area - target_val
        la_ratio_block = total_usable_area / target_val if target_val > 0 else np.nan

        plot_front = F
        if plot_front > 0:
            plot_depth = plot_area / plot_front
        else:
            plot_depth = np.nan

        logger.debug(
            "[BlockPlotsAllocator.allocate] block_id={}: CHOSEN -> "
            "L={}, W={}, H={}, F={}, building_need={}, building_capacity={}, "
            "buildings_count={}, living_per_building={}, total_usable_area={}, "
            "plot_front={}, plot_depth={}, far_initial={}, far_final={}, la_diff={}".format(
                ctx.block_id,
                L,
                W,
                H,
                F,
                building_need_val,
                building_capacity,
                buildings_count,
                usable_one,
                total_usable_area,
                plot_front,
                plot_depth,
                far_initial,
                far_final,
                la_diff,
            )
        )

        rects_block["plots_capacity"] = 0
        rects_block["plot_front"] = float(plot_front)
        rects_block["plot_depth"] = np.nan

        plots_capacity_total = 0
        plot_depth_norm = plot_depth

        for i, idx_seg in enumerate(rects_block.index):
            front_len = float(ctx.seg_fronts[i])
            depth_len = float(ctx.seg_depths[i])

            cap_seg = int(front_len // plot_front) if plot_front > 0 else 0
            if cap_seg <= 0:
                continue

            depth_seg = (
                min(plot_depth_norm, depth_len)
                if not np.isnan(plot_depth_norm)
                else depth_len
            )

            rects_block.at[idx_seg, "plots_capacity"] = cap_seg
            rects_block.at[idx_seg, "plot_front"] = plot_front
            rects_block.at[idx_seg, "plot_depth"] = depth_seg

            plots_capacity_total += cap_seg

        logger.debug(
            "[BlockPlotsAllocator.allocate] block_id={}: "
            "plots_capacity_total={}, initial_building_capacity={}".format(
                ctx.block_id,
                plots_capacity_total,
                building_capacity,
            )
        )

        if plots_capacity_total != building_capacity:
            logger.debug(
                "[BlockPlotsAllocator.allocate] block_id={}: "
                "plots_capacity_total ({}) != building_capacity ({}), overriding "
                "building_capacity".format(
                    ctx.block_id,
                    plots_capacity_total,
                    building_capacity,
                )
            )
            building_capacity = plots_capacity_total

        block_result: Dict[str, Any] = {
            "building_need": int(building_need_val),
            "building_capacity": int(building_capacity),
            "buildings_count": int(buildings_count),
            "building_length": float(L),
            "building_width": float(W),
            "floors_count": float(H),
            "plot_front": float(plot_front),
            "plot_side_used": float(plot_front),
            "plot_depth": float(plot_depth) if not np.isnan(plot_depth) else np.nan,
            "plot_area": float(plot_area),
            "living_per_building": float(usable_one),
            "total_usable_area": float(total_usable_area),
            "la_diff": float(la_diff),
            "la_ratio": float(la_ratio_block),
            "far_initial": float(far_initial) if not np.isnan(far_initial) else np.nan,
            "far_final": float(far_final),
            "far_diff": float(far_final - far_initial)
            if not np.isnan(far_initial)
            else np.nan,
        }

        logger.debug(
            "[BlockPlotsAllocator.allocate] block_id={}: DONE -> "
            "building_capacity={}, buildings_count={}, plot_front={}, plot_depth={}".format(
                ctx.block_id,
                block_result["building_capacity"],
                block_result["buildings_count"],
                block_result["plot_front"],
                block_result["plot_depth"],
            )
        )

        return BlockPlotsAllocationResult(
            block_result=block_result,
            rects_block=rects_block,
        )
