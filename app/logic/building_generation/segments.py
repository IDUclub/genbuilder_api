from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, Tuple

from loguru import logger
from tqdm.auto import tqdm
import geopandas as gpd
import numpy as np
import pandas as pd

from app.logic.building_generation.building_capacity_optimizer import CapacityOptimizer
from app.logic.building_generation.building_params import (
    BuildingGenParams,
    BuildingParamsProvider,
    BuildingType,
    BuildingParams,
)

class SegmentsAllocator:
    """
    Allocates building capacity across rectangular segments inside blocks.

    Uses block targets and building presets to:
    - choose a consistent (L, W, H, plot_front) scenario per block;
    - compute how many plots/buildings fit into each segment;
    - update blocks with final capacity/FAR and segments with plot_front/depth.
    """
    def __init__(
        self,
        capacity_optimizer: CapacityOptimizer,
        building_params_provider: BuildingParamsProvider,
    ):
        self.capacity_optimizer = capacity_optimizer
        self._building_params = building_params_provider

    @property
    def building_generation_parameters(self) -> BuildingGenParams:
        return self._building_params.current()

    def solve_block_with_segments(
        self,
        row: pd.Series,
        rects_block: gpd.GeoDataFrame,
        far: str,
        *,
        building_params: BuildingParams,
        la_ratio: float | None = None,
    ) -> Tuple[Dict[str, Any], gpd.GeoDataFrame]:

        block_id = row.name
        floors_group = row.get("floors_group")
        la_target_raw = row.get("la_target")
        logger.debug(
            f"[solve_block] start: block_id={block_id}, far={far}, "
            f"floors_group={floors_group}, la_target={la_target_raw}, "
            f"rects_count={len(rects_block)}, "
            f"building_params.la_coef={building_params.la_coef}, "
            f"plot_side_range=({min(building_params.plot_side)}, {max(building_params.plot_side)})"
        )

        if la_ratio is None:
            la_ratio = building_params.la_coef
        logger.debug(f"[solve_block] block_id={block_id}: using la_ratio={la_ratio}")

        if rects_block.empty:
            logger.debug(f"[solve_block] block_id={block_id}: rects_block is EMPTY, return zeros/NaN")
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
                "total_living_area": row.get("total_living_area", np.nan),
                "la_diff": row.get("la_diff", np.nan),
                "la_ratio": row.get("la_ratio", np.nan),
                "far_initial": row.get("far_initial", np.nan),
                "far_final": row.get("far_final", np.nan),
                "far_diff": row.get("far_diff", np.nan),
            }, rects_block

        target_la = float(row["la_target"])
        logger.debug(f"[solve_block] block_id={block_id}: target_la={target_la}")
        if target_la <= 0:
            logger.debug(
                f"[solve_block] block_id={block_id}: target_la <= 0 "
                f"(target_la={target_la}), return zeros/NaN"
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
                "total_living_area": 0.0,
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

        base_L_idx, base_W_idx, base_H_idx, base_F_idx = self.capacity_optimizer.pick_indices(
            far,
            building_params,
        )

        base_L = float(building_params.building_length_range[base_L_idx])
        base_W = float(building_params.building_width_range[base_W_idx])
        base_H = float(building_params.building_height[base_H_idx])
        base_F = float(building_params.plot_side[base_F_idx])

        far_initial = float(
            row.get("far_initial", (base_L * base_W * base_H) / plot_area_base)
        )

        logger.debug(
            f"[solve_block] block_id={block_id}: "
            f"area_min={area_min}, area_max={area_max}, area_base={area_base}, "
            f"base_L={base_L}, base_W={base_W}, base_H={base_H}, base_F={base_F}, "
            f"far_initial={far_initial}"
        )


        rects_block = rects_block.copy()
        seg_fronts: list[float] = []
        seg_depths: list[float] = []

        for _, seg in rects_block.iterrows():
            w = float(seg["width"])
            h = float(seg["height"])
            seg_front = max(w, h)
            seg_depth = min(w, h)
            seg_fronts.append(seg_front)
            seg_depths.append(seg_depth)

        logger.debug(
            f"[solve_block] block_id={block_id}: seg_fronts={seg_fronts}, seg_depths={seg_depths}"
        )


        capacity_by_front_idx: list[int] = []
        for F in building_params.plot_side:
            F_val = float(F)
            if F_val <= 0:
                capacity_by_front_idx.append(0)
                continue
            cap_total = 0
            for front_len in seg_fronts:
                cap_total += int(front_len // F_val)
            capacity_by_front_idx.append(cap_total)

        max_cap = max(capacity_by_front_idx) if capacity_by_front_idx else 0
        logger.debug(
            f"[solve_block] block_id={block_id}: "
            f"capacity_by_front_idx={capacity_by_front_idx}, max_capacity={max_cap}"
        )

        if max_cap <= 0:
            logger.debug(
                f"[solve_block] block_id={block_id}: max_capacity <= 0, "
                f"return zeros/NaN for plots"
            )
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
                "plot_front": np.nan,
                "plot_side_used": np.nan,
                "plot_depth": np.nan,
                "plot_area": plot_area_base,
                "living_per_building": row.get("living_per_building", np.nan),
                "total_living_area": 0.0,
                "la_diff": -target_la,
                "la_ratio": 0.0,
                "far_initial": far_initial,
                "far_final": np.nan,
                "far_diff": np.nan,
            }, rects_block


        best_success: Dict[str, Any] | None = None
        best_success_score: Tuple[float, float, float, float, float] | None = None

        fallback: Dict[str, Any] | None = None
        fallback_total_la: float = -1.0

        def consider_candidate(
            L_idx: int,
            W_idx: int,
            H_idx: int,
            F_idx: int,
            only_long_front: bool,
        ) -> None:
            nonlocal best_success, best_success_score, fallback, fallback_total_la

            L = float(building_params.building_length_range[L_idx])
            W = float(building_params.building_width_range[W_idx])
            H = float(building_params.building_height[H_idx])
            F = float(building_params.plot_side[F_idx])

            living_per_building = L * W * H * la_ratio
            if living_per_building <= 0:
                return

            if F > 0:
                plot_depth = plot_area_base / F
            else:
                plot_depth = float("inf")

            is_long_side_along_front = F >= plot_depth

            building_need = int(math.ceil(target_la / living_per_building)) if target_la > 0 else 0

            capacity_total = int(capacity_by_front_idx[F_idx])


            if building_need > 0:
                buildings_count_fallback = min(capacity_total, building_need)
            else:
                buildings_count_fallback = capacity_total
            total_la_fallback = buildings_count_fallback * living_per_building

            if total_la_fallback > fallback_total_la:
                far_final_fb = (L * W * H) / plot_area_base if plot_area_base > 0 else float("nan")
                fallback_total_la = float(total_la_fallback)
                fallback = {
                    "L": L,
                    "W": W,
                    "H": H,
                    "F": F,
                    "building_need": building_need,
                    "building_capacity": capacity_total,
                    "buildings_count": buildings_count_fallback,
                    "living_per_building": living_per_building,
                    "total_living_area": total_la_fallback,
                    "far_final": far_final_fb,
                }
                logger.debug(
                    f"[solve_block] block_id={block_id}: fallback updated -> "
                    f"L={L}, W={W}, H={H}, F={F}, "
                    f"capacity_total={capacity_total}, total_la_fallback={total_la_fallback}"
                )


            if building_need <= 0 or capacity_total < building_need:
                return

            if only_long_front and not is_long_side_along_front:
                return

            buildings_count = building_need
            total_la = buildings_count * living_per_building
            la_diff_abs = abs(total_la - target_la)

            far_final = (L * W * H) / plot_area_base if plot_area_base > 0 else float("nan")
            far_diff_abs = abs(far_final - far_initial)

            front_reduction = base_F - F
            floor_delta = H - base_H
            area_diff_abs = abs((L * W) - (base_L * base_W))

            score = (
                la_diff_abs,
                far_diff_abs,
                front_reduction,
                floor_delta,
                area_diff_abs,
            )

            if (best_success_score is None) or (score < best_success_score):
                best_success_score = score
                best_success = {
                    "L": L,
                    "W": W,
                    "H": H,
                    "F": F,
                    "building_need": building_need,
                    "building_capacity": capacity_total,
                    "buildings_count": buildings_count,
                    "living_per_building": living_per_building,
                    "total_living_area": total_la,
                    "far_final": far_final,
                }
                logger.debug(
                    f"[solve_block] block_id={block_id}: best_success updated -> "
                    f"L={L}, W={W}, H={H}, F={F}, "
                    f"building_need={building_need}, capacity_total={capacity_total}, "
                    f"total_la={total_la}, score={score}"
                )

        logger.debug(
            f"[solve_block] block_id={block_id}: "
            f"searching candidates: len(L_range)={len(building_params.building_length_range)}, "
            f"len(W_range)={len(building_params.building_width_range)}, "
            f"len(H_range)={len(building_params.building_height)}, "
            f"len(F_range)={len(building_params.plot_side)}"
        )


        for L_idx in range(base_L_idx, len(building_params.building_length_range)):
            for W_idx in range(base_W_idx, len(building_params.building_width_range)):
                consider_candidate(L_idx, W_idx, base_H_idx, base_F_idx, only_long_front=True)


        if best_success is None:
            logger.debug(
                f"[solve_block] block_id={block_id}: no best_success after L/W scan, "
                f"trying H-range with fixed F"
            )
            for H_idx in range(base_H_idx, len(building_params.building_height)):
                for L_idx in range(base_L_idx, len(building_params.building_length_range)):
                    for W_idx in range(base_W_idx, len(building_params.building_width_range)):
                        consider_candidate(L_idx, W_idx, H_idx, base_F_idx, only_long_front=True)


        if best_success is None:
            logger.debug(
                f"[solve_block] block_id={block_id}: still no best_success, "
                f"trying smaller F with only_long_front=True"
            )
            for F_idx in range(base_F_idx, 0, -1):
                for H_idx in range(base_H_idx, len(building_params.building_height)):
                    for L_idx in range(base_L_idx, len(building_params.building_length_range)):
                        for W_idx in range(base_W_idx, len(building_params.building_width_range)):
                            consider_candidate(L_idx, W_idx, H_idx, F_idx, only_long_front=True)


        if best_success is None:
            logger.debug(
                f"[solve_block] block_id={block_id}: no best_success yet, "
                f"allowing short-front (only_long_front=False)"
            )
            for F_idx in range(base_F_idx, 0, -1):
                for H_idx in range(base_H_idx, len(building_params.building_height)):
                    for L_idx in range(base_L_idx, len(building_params.building_length_range)):
                        for W_idx in range(base_W_idx, len(building_params.building_width_range)):
                            consider_candidate(L_idx, W_idx, H_idx, F_idx, only_long_front=False)

        chosen = best_success or fallback
        if chosen is None:
            logger.debug(
                f"[solve_block] block_id={block_id}: chosen is None "
                f"(no best_success and no fallback) -> return zeros/NaN for plots"
            )
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
                "plot_front": np.nan,
                "plot_side_used": np.nan,
                "plot_depth": np.nan,
                "plot_area": plot_area_base,
                "living_per_building": row.get("living_per_building", np.nan),
                "total_living_area": 0.0,
                "la_diff": -target_la,
                "la_ratio": 0.0,
                "far_initial": far_initial,
                "far_final": np.nan,
                "far_diff": np.nan,
            }, rects_block


        L = chosen["L"]
        W = chosen["W"]
        H = chosen["H"]
        F = chosen["F"]
        building_need = chosen["building_need"]
        building_capacity = chosen["building_capacity"]
        buildings_count = chosen["buildings_count"]
        living_per_building = chosen["living_per_building"]
        total_living_area = chosen["total_living_area"]
        far_final = chosen["far_final"]

        la_diff = total_living_area - target_la
        la_ratio_block = total_living_area / target_la if target_la > 0 else np.nan

        plot_front = F
        plot_area = plot_area_base
        plot_depth = plot_area / plot_front if plot_front > 0 else np.nan

        logger.debug(
            f"[solve_block] block_id={block_id}: CHOSEN -> "
            f"L={L}, W={W}, H={H}, F={F}, "
            f"building_need={building_need}, building_capacity={building_capacity}, "
            f"buildings_count={buildings_count}, "
            f"living_per_building={living_per_building}, total_living_area={total_living_area}, "
            f"plot_front={plot_front}, plot_depth={plot_depth}, "
            f"far_initial={far_initial}, far_final={far_final}, la_diff={la_diff}"
        )


        plots_capacity_total = 0
        plot_depth_norm = plot_depth

        rects_block["plots_capacity"] = 0
        rects_block["plot_front"] = plot_front
        rects_block["plot_depth"] = np.nan

        for i, idx_seg in enumerate(rects_block.index):
            front_len = seg_fronts[i]
            depth_len = seg_depths[i]

            cap_seg = int(front_len // plot_front) if plot_front > 0 else 0
            if cap_seg <= 0:
                continue

            depth_seg = min(plot_depth_norm, depth_len) if not np.isnan(plot_depth_norm) else depth_len

            rects_block.at[idx_seg, "plots_capacity"] = cap_seg
            rects_block.at[idx_seg, "plot_front"] = plot_front
            rects_block.at[idx_seg, "plot_depth"] = depth_seg

            plots_capacity_total += cap_seg

        logger.debug(
            f"[solve_block] block_id={block_id}: "
            f"plots_capacity_total={plots_capacity_total}, "
            f"initial_building_capacity={building_capacity}"
        )

        if plots_capacity_total != building_capacity:
            logger.debug(
                f"[solve_block] block_id={block_id}: "
                f"plots_capacity_total ({plots_capacity_total}) != building_capacity ({building_capacity}), "
                f"overriding building_capacity"
            )
            building_capacity = plots_capacity_total

        block_result = {
            "building_need": int(building_need),
            "building_capacity": int(building_capacity),
            "buildings_count": int(buildings_count),
            "building_length": float(L),
            "building_width": float(W),
            "floors_count": float(H),
            "plot_front": float(plot_front),
            "plot_side_used": float(plot_front),
            "plot_depth": float(plot_depth) if not np.isnan(plot_depth) else np.nan,
            "plot_area": float(plot_area),
            "living_per_building": float(living_per_building),
            "total_living_area": float(total_living_area),
            "la_diff": float(la_diff),
            "la_ratio": float(la_ratio_block),
            "far_initial": float(far_initial),
            "far_final": float(far_final),
            "far_diff": float(far_final - far_initial),
        }

        logger.debug(
            f"[solve_block] block_id={block_id}: DONE -> "
            f"building_capacity={block_result['building_capacity']}, "
            f"buildings_count={block_result['buildings_count']}, "
            f"plot_front={block_result['plot_front']}, "
            f"plot_depth={block_result['plot_depth']}"
        )

        return block_result, rects_block

    def update_blocks_with_segments(
        self,
        blocks_gdf: gpd.GeoDataFrame,
        rects_gdf: gpd.GeoDataFrame,
        far: str,
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

        blocks = blocks_gdf.copy()
        rects = rects_gdf.copy()

        logger.debug(
            f"[update_blocks] start: blocks_count={len(blocks)}, "
            f"rects_count={len(rects)}, far={far}"
        )

        if rects.empty:
            logger.debug("[update_blocks] rects is EMPTY, nothing to update")
            return blocks, rects

        for col, default in [
            ("plots_capacity", 0),
            ("plot_front", np.nan),
            ("plot_depth", np.nan),
        ]:
            if col not in rects.columns:
                rects[col] = default

        try:
            unique_src = rects["src_index"].unique()
        except Exception:
            unique_src = "N/A"

        logger.debug(f"[update_blocks] rects.src_index.unique={unique_src}")

        for idx, row in blocks.iterrows():
            fg = row.get("floors_group")
            la_target = row.get("la_target")
            logger.debug(
                f"[update_blocks] block_idx={idx}: floors_group={fg}, la_target={la_target}"
            )

            block_rects = rects[rects["src_index"] == idx]
            logger.debug(
                f"[update_blocks] block_idx={idx}: matched_rects_count={len(block_rects)}"
            )

            if block_rects.empty:
                logger.debug(
                    f"[update_blocks] block_idx={idx}: NO rects for this block, "
                    f"set building_capacity=0, buildings_count=0"
                )
                blocks.at[idx, "building_capacity"] = 0
                blocks.at[idx, "buildings_count"] = 0
                continue

            try:
                building_type = BuildingType(fg)
                logger.debug(
                    f"[update_blocks] block_idx={idx}: mapped floors_group={fg} "
                    f"-> building_type={building_type}"
                )
            except Exception as e:
                logger.error(
                    f"[update_blocks] block_idx={idx}: cannot map floors_group={fg} "
                    f"to BuildingType: {e!r}, set building_capacity=0, buildings_count=0"
                )
                blocks.at[idx, "building_capacity"] = 0
                blocks.at[idx, "buildings_count"] = 0
                continue

            try:
                building_params = self.building_generation_parameters.params_by_type[building_type]
            except KeyError as e:
                logger.error(
                    f"[update_blocks] block_idx={idx}: building_type={building_type} "
                    f"not in params_by_type keys="
                    f"{list(self.building_generation_parameters.params_by_type.keys())} "
                    f"-> set building_capacity=0, buildings_count=0"
                )
                blocks.at[idx, "building_capacity"] = 0
                blocks.at[idx, "buildings_count"] = 0
                continue

            logger.debug(
                f"[update_blocks] block_idx={idx}: calling solve_block_with_segments "
                f"with {len(block_rects)} rects, building_type={building_type}"
            )

            block_res, block_rects_res = self.solve_block_with_segments(
                row=row,
                rects_block=block_rects,
                far=far,
                building_params=building_params,
                la_ratio=building_params.la_coef,
            )

            logger.debug(
                f"[update_blocks] block_idx={idx}: solve_block_with_segments returned -> "
                f"building_capacity={block_res.get('building_capacity')}, "
                f"buildings_count={block_res.get('buildings_count')}, "
                f"plot_front={block_res.get('plot_front')}, "
                f"plot_depth={block_res.get('plot_depth')}"
            )

            for key, value in block_res.items():
                blocks.at[idx, key] = value

            rects.loc[block_rects_res.index, ["plots_capacity", "plot_front", "plot_depth"]] = \
                block_rects_res[["plots_capacity", "plot_front", "plot_depth"]].values

            rects.loc[block_rects_res.index, "src_index"] = idx
            rects.loc[block_rects_res.index, "la_target"] = row.get("la_target", np.nan)

            rects.loc[block_rects_res.index, "far_initial"] = block_res.get(
                "far_initial",
                row.get("far_initial", np.nan),
            )
            rects.loc[block_rects_res.index, "building_length"] = block_res.get(
                "building_length",
                row.get("building_length", np.nan),
            )
            rects.loc[block_rects_res.index, "building_width"] = block_res.get(
                "building_width",
                row.get("building_width", np.nan),
            )
            rects.loc[block_rects_res.index, "floors_count"] = block_res.get(
                "floors_count",
                row.get("floors_count", np.nan),
            )
            rects.loc[block_rects_res.index, "floors_group"] = row.get("floors_group", np.nan)

            logger.debug(
                f"[update_blocks] block_idx={idx}: rects updated for indices "
                f"{list(block_rects_res.index)}"
            )

        logger.debug("[update_blocks] DONE")
        return blocks, rects
