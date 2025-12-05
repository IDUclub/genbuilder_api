from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.logic.building_params import (
    BuildingGenParams,
    BuildingParamsProvider,
    BuildingParams,
)
from app.logic.building_type_resolver import infer_building_type
from app.common.building_math import (
    usable_per_building,
    far_from_dims,
    building_need,
)


class CapacityOptimizer:
    """
    Block-level capacity planner that uses preset building parameters and a FAR
    scenario to derive representative building/plot dimensions and initial
    capacity metrics for individual blocks or entire GeoDataFrames.
    """

    def __init__(
        self,
        building_params_provider: BuildingParamsProvider,
    ):
        self._building_params = building_params_provider

    @property
    def building_generation_parameters(self) -> BuildingGenParams:
        return self._building_params.current()

    def pick_indices(
        self, far: str, params: BuildingParams
    ) -> tuple[int, int, int, int]:

        if far == "min":
            L_idx = 0
            W_idx = 0
            H_idx = 0
            F_idx = len(params.plot_side) - 1
        elif far == "mean":
            L_idx = len(params.building_length_range) // 2
            W_idx = len(params.building_width_range) // 2
            H_idx = len(params.building_height) // 2
            F_idx = len(params.plot_side) // 2

        elif far == "max":
            L_idx = len(params.building_length_range) - 1
            W_idx = len(params.building_width_range) - 1
            H_idx = len(params.building_height) - 1
            F_idx = 0
        else:
            raise ValueError(f"Unknown FAR scenario: {far!r}")

        return L_idx, W_idx, H_idx, F_idx

    def get_plot_area_params(
        self, far: str, params: BuildingParams
    ) -> tuple[float, float, float]:

        if far == "min":
            area_base = params.plot_area_max
        elif far == "mean":
            area_base = 0.5 * (params.plot_area_min + params.plot_area_max)
        elif far == "max":
            area_base = params.plot_area_min
        else:
            raise ValueError(f"Unknown FAR scenario for plot area: {far!r}")

        return params.plot_area_min, params.plot_area_max, area_base

    def solve_block_initial(
        self,
        target_area: float,
        far: str,
        *,
        building_params: BuildingParams,
        la_ratio: float | None = None,
    ) -> dict:

        if la_ratio is None:
            la_ratio = building_params.la_coef

        area_min, area_max, area_base = self.get_plot_area_params(far, building_params)

        base_L_idx, base_W_idx, base_h_idx, base_front_idx = self.pick_indices(
            far, building_params
        )

        L = float(building_params.building_length_range[base_L_idx])
        W = float(building_params.building_width_range[base_W_idx])
        H = float(building_params.building_height[base_h_idx])
        F = float(building_params.plot_side[base_front_idx])

        plot_area_base = float(area_base)
        plot_depth_base = plot_area_base / F if F > 0 else float("nan")

        usable_one = usable_per_building(L, W, H, la_ratio)
        building_need_val = building_need(target_area, usable_one)
        far_target = far_from_dims(L, W, H, plot_area_base)
        
        return {
            "building_length": L,
            "building_width": W,
            "floors": H,
            "plot_front": F,
            "plot_depth": plot_depth_base,
            "plot_area": plot_area_base,
            "living_per_building": float(usable_one),
            "building_need": int(building_need_val),
            "building_capacity": None,
            "far_target": float(far_target),
        }

    def compute_block(
        self,
        row: pd.Series,
        far: str,
        *,
        target_col: str = "la_target",
        mode: str = "residential",
    ) -> pd.Series:

        try:
            building_type = infer_building_type(row, mode=mode)
        except Exception:
            building_type = None

        if building_type is None:
            return pd.Series(
                {
                    "building_need": np.nan,
                    "building_capacity": np.nan,
                    "buildings_count": 0,
                    "plot_front": np.nan,
                    "plot_depth": np.nan,
                    "plot_area": np.nan,
                    "plot_side_used": np.nan,
                    "building_length": np.nan,
                    "building_width": np.nan,
                    "floors_count": np.nan,
                    "living_per_building": np.nan,
                    "total_usable_area": np.nan,
                    "la_diff": np.nan,
                    "la_ratio": np.nan,
                    "far_initial": np.nan,
                    "far_final": np.nan,
                    "far_diff": np.nan,
                }
            )

        try:
            building_params = self.building_generation_parameters.params_by_type[
                building_type
            ]
        except KeyError:
            return pd.Series(
                {
                    "building_need": np.nan,
                    "building_capacity": np.nan,
                    "buildings_count": 0,
                    "plot_front": np.nan,
                    "plot_depth": np.nan,
                    "plot_area": np.nan,
                    "plot_side_used": np.nan,
                    "building_length": np.nan,
                    "building_width": np.nan,
                    "floors_count": np.nan,
                    "living_per_building": np.nan,
                    "total_usable_area": np.nan,
                    "la_diff": np.nan,
                    "la_ratio": np.nan,
                    "far_initial": np.nan,
                    "far_final": np.nan,
                    "far_diff": np.nan,
                }
            )
        try:
            target_val_raw = row.get(target_col, 0.0)
            target_val = float(target_val_raw)
        except (TypeError, ValueError):
            target_val = 0.0

        res = self.solve_block_initial(
            target_area=target_val,
            far=far,
            building_params=building_params,
            la_ratio=building_params.la_coef,
        )

        building_need = res["building_need"]
        usable_per_building = res["living_per_building"]
        total_usable = building_need * usable_per_building
        la_diff = total_usable - target_val
        la_ratio_block = total_usable / target_val if target_val > 0 else np.nan

        return pd.Series(
            {
                "building_need": building_need,
                "building_capacity": np.nan,
                "buildings_count": building_need,
                "plot_front": res["plot_front"],
                "plot_depth": res["plot_depth"],
                "plot_area": res["plot_area"],
                "plot_side_used": res["plot_front"],
                "building_length": res["building_length"],
                "building_width": res["building_width"],
                "floors_count": res["floors"],
                "living_per_building": usable_per_building,
                "total_usable_area": total_usable,
                "la_diff": la_diff,
                "la_ratio": la_ratio_block,
                "far_initial": res["far_target"],
                "far_final": np.nan,
                "far_diff": np.nan,
            }
        )

    def compute_blocks_for_gdf(
        self,
        blocks_gdf: pd.DataFrame,
        far: str,
        *,
        target_col: str = "la_target",
        mode: str = "residential",
    ) -> pd.DataFrame:

        base_cols = blocks_gdf.apply(
            lambda row: self.compute_block(
                row,
                far=far,
                target_col=target_col,
                mode=mode,
            ),
            axis=1,
        )
        return pd.concat([blocks_gdf, base_cols], axis=1)
