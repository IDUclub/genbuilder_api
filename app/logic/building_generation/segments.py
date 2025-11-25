from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from tqdm.auto import tqdm
import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import Polygon


class SegmentsAllocator:

    def __init__(self, params_provider: ParamsProvider, capacity_optimizer: CapacityOptimizer):
        self._params = params_provider
        self.capacity_optimizer = capacity_optimizer

    @property
    def generation_parameters(self) -> GenParams:
        return self._params.current()
    
    def solve_block_with_segments(self,
    row: pd.Series,
    rects_block: gpd.GeoDataFrame,
    far: str,
    *,
    la_ratio: float = self.generation_parameters.la_coef,
) -> Tuple[Dict[str, Any], gpd.GeoDataFrame]:
        """
        ШАГИ 2–3 с приоритетом:
        1) сначала ищем комбинации (L,W,H,F) и building_capacity, где участок
            расположен ДЛИННОЙ стороной по границе квартала (plot_depth <= plot_front);
        2) только если ни одна не даёт building_capacity >= building_need,
            допускаем варианты, где по границе лежит короткая сторона (plot_depth > plot_front).
        """

        if rects_block.empty or row.get("floors_group") != "private":
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
        if target_la <= 0:
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


        area_min, area_max, area_base = self.capacity_opimizer.get_plot_area_params(far)
        plot_area_base = float(area_base)

        base_L_idx, base_W_idx, base_H_idx, base_F_idx = self.capacity_opimizer.pick_indices(far)
        base_L = float(self.generation_parameters.building_length_range[base_L_idx])
        base_W = float(self.generation_parameters.building_width_range[base_W_idx])
        base_H = float(self.generation_parameters.building_height[base_H_idx])
        base_F = float(self.generation_parameters.plot_side[base_F_idx])

        far_initial = float(
            row.get("far_initial", (base_L * base_W * base_H) / plot_area_base)
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


        capacity_by_front_idx: list[int] = []
        for F in plot_side:
            F_val = float(F)
            if F_val <= 0:
                capacity_by_front_idx.append(0)
                continue
            cap_total = 0
            for front_len in seg_fronts:
                cap_total += int(front_len // F_val)
            capacity_by_front_idx.append(cap_total)

        if max(capacity_by_front_idx) <= 0:
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

            L = float(self.generation_parameters.building_length_range[L_idx])
            W = float(self.generation_parameters.building_width_range[W_idx])
            H = float(self.generation_parameters.building_height[H_idx])
            F = float(self.generation_parameters.plot_side[F_idx])

            living_per_building = L * W * H * la_ratio
            if living_per_building <= 0:
                return


            if F > 0:
                plot_depth = plot_area_base / F
            else:
                plot_depth = float("inf")

            is_long_side_along_front = F >= plot_depth

            # building_need
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



        for L_idx in range(base_L_idx, len(self.generation_parameters.building_length_range)):
            for W_idx in range(base_W_idx, len(self.generation_parameters.building_width_range)):
                consider_candidate(L_idx, W_idx, base_H_idx, base_F_idx, only_long_front=True)


        if best_success is None:
            for H_idx in range(base_H_idx, len(self.generation_parameters.building_height)):
                for L_idx in range(base_L_idx, len(self.generation_parameters.building_length_range)):
                    for W_idx in range(base_W_idx, len(self.generation_parameters.building_width_range)):
                        consider_candidate(L_idx, W_idx, H_idx, base_F_idx, only_long_front=True)


        if best_success is None:
            for F_idx in range(base_F_idx, 0, -1):
                for H_idx in range(base_H_idx, len(self.generation_parameters.building_height)):
                    for L_idx in range(base_L_idx, len(self.generation_parameters.building_length_range)):
                        for W_idx in range(base_W_idx, len(self.generation_parameters.building_width_range)):
                            consider_candidate(L_idx, W_idx, H_idx, F_idx, only_long_front=True)


        if best_success is None:
            for F_idx in range(base_F_idx, 0, -1):
                for H_idx in range(base_H_idx, len(self.generation_parameters.building_height)):
                    for L_idx in range(base_L_idx, len(self.generation_parameters.building_length_range)):
                        for W_idx in range(base_W_idx, len(self.generation_parameters.uilding_width_range)):
                            consider_candidate(L_idx, W_idx, H_idx, F_idx, only_long_front=False)

        chosen = best_success or fallback
        if chosen is None:
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

        if plots_capacity_total != building_capacity:
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

        return block_result, rects_block

    def update_blocks_with_segments(self,
        blocks_gdf: gpd.GeoDataFrame,
        rects_gdf: gpd.GeoDataFrame
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Применяет solve_block_with_segments ко всем кварталам.

        На вход:
        - blocks_gdf: кварталы с результатами compute_private_row
        - rects_gdf: прямоугольные сегменты от pack_inscribed_rectangles_for_gdf
                    (должен быть столбец 'src_index' с индексом квартала)

        На выход:
        - обновлённый blocks_gdf
        - обновлённый rects_gdf (plots_capacity, plot_front, plot_depth + блочные атрибуты)
        """
        blocks = blocks_gdf.copy()
        rects = rects_gdf.copy()

        if rects.empty:
            return blocks, rects


        for col, default in [
            ("plots_capacity", 0),
            ("plot_front", np.nan),
            ("plot_depth", np.nan),
        ]:
            if col not in rects.columns:
                rects[col] = default

        for idx, row in blocks.iterrows():
            if row.get("floors_group") != "private":
                continue

            block_rects = rects[rects["src_index"] == idx]
            if block_rects.empty:
                blocks.at[idx, "building_capacity"] = 0
                blocks.at[idx, "buildings_count"] = 0
                continue

            block_res, block_rects_res = self.solve_block_with_segments(
                row=row,
                rects_block=block_rects,
                far=self.generation_parameters.far,
                la_ratio=self.generation_parameters.la_coef,
            )


            for key, value in block_res.items():
                blocks.at[idx, key] = value


            rects.loc[block_rects_res.index, ["plots_capacity", "plot_front", "plot_depth"]] = \
                block_rects_res[["plots_capacity", "plot_front", "plot_depth"]].values


            rects.loc[block_rects_res.index, "src_index"] = idx
            rects.loc[block_rects_res.index, "la_target"] = row.get("la_target", np.nan)

            rects.loc[block_rects_res.index, "far_initial"] = block_res.get("far_initial", row.get("far_initial", np.nan))
            rects.loc[block_rects_res.index, "building_length"] = block_res.get("building_length", row.get("building_length", np.nan))
            rects.loc[block_rects_res.index, "building_width"] = block_res.get("building_width", row.get("building_width", np.nan))
            rects.loc[block_rects_res.index, "floors_count"] = block_res.get("floors_count", row.get("floors_count", np.nan))

        return blocks, rects
