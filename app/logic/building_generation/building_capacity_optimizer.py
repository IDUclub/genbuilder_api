from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.logic.building_generation.building_params import (
    BuildingGenParams,
    BuildingParamsProvider,
    BuildingType,
    BuildingParams,
)

class CapacityOptimizer:
    """
    Helper class for block-level capacity planning in residential generation.

    It uses building parameter presets (via BuildingParamsProvider) and a chosen
    FAR scenario ("min", "mean", "max") to:
    - select representative building and plot dimensions;
    - estimate living area per building;
    - compute the required number of buildings to reach target living area;
    - derive initial FAR and plot geometry attributes for each block.

    The main entry points are:
    - solve_block_initial(...) – compute parameters for a single block;
    - compute_block(...) – wrapper for a pandas row;
    - compute_blocks_for_gdf(...) – apply the logic to an entire GeoDataFrame.
    """
    def __init__(
        self,
        building_params_provider: BuildingParamsProvider,
    ):
        self._building_params = building_params_provider

    @property
    def building_generation_parameters(self) -> BuildingGenParams:
        return self._building_params.current()

    def pick_indices(self, far: str, params: BuildingParams) -> tuple[int, int, int, int]:

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

    def get_plot_area_params(self, far: str, params: BuildingParams) -> tuple[float, float, float]:

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
        target_la: float,
        far: str,
        *,
        building_params: BuildingParams,
        la_ratio: float | None = None,
    ) -> dict:

        if la_ratio is None:
            la_ratio = building_params.la_coef

        area_min, area_max, area_base = self.get_plot_area_params(far, building_params)

        base_L_idx, base_W_idx, base_h_idx, base_front_idx = self.pick_indices(far, building_params)

        L = float(building_params.building_length_range[base_L_idx])
        W = float(building_params.building_width_range[base_W_idx])
        H = float(building_params.building_height[base_h_idx])
        F = float(building_params.plot_side[base_front_idx])

        plot_area_base = float(area_base)
        plot_depth_base = plot_area_base / F if F > 0 else float("nan")

        base_house_area = L * W
        living_per_building = base_house_area * H * la_ratio

        if target_la > 0 and living_per_building > 0:
            building_need = int(math.ceil(target_la / living_per_building))
        else:
            building_need = 0

        far_target = (base_house_area * H) / plot_area_base if plot_area_base > 0 else float("nan")

        return {
            "building_length": L,
            "building_width": W,
            "floors": H,
            "plot_front": F,
            "plot_depth": plot_depth_base,
            "plot_area": plot_area_base,
            "living_per_building": float(living_per_building),
            "building_need": int(building_need),
            "building_capacity": None,
            "far_target": float(far_target),
        }

    def compute_block(self, row: pd.Series, far: str) -> pd.Series:
        try:
            building_type = BuildingType(row["floors_group"])
        except Exception:
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
                    "total_living_area": np.nan,
                    "la_diff": np.nan,
                    "la_ratio": np.nan,
                    "far_initial": np.nan,
                    "far_final": np.nan,
                    "far_diff": np.nan,
                }
            )

        building_params = self.building_generation_parameters.params_by_type[building_type]
        target_la = float(row["la_target"])

        res = self.solve_block_initial(
            target_la=target_la,
            far=far,
            building_params=building_params,           
            la_ratio=building_params.la_coef, 
        )

        building_need = res["building_need"]
        living_per_building = res["living_per_building"]
        total_la = building_need * living_per_building
        la_diff = total_la - target_la
        la_ratio_block = total_la / target_la if target_la > 0 else np.nan

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
                "living_per_building": living_per_building,
                "total_living_area": total_la,
                "la_diff": la_diff,
                "la_ratio": la_ratio_block,
                "far_initial": res["far_target"],
                "far_final": np.nan,
                "far_diff": np.nan,
            }
        )
    
    def compute_blocks_for_gdf(
        self,
        blocks_gdf,
        far: str,
    ):

        base_cols = blocks_gdf.apply(
            lambda row: self.compute_block(row, far=far),
            axis=1,
        )
        return pd.concat([blocks_gdf, base_cols], axis=1)