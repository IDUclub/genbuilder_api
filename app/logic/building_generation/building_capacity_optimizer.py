from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from app.logic.building_generation.building_params import (
    BuildingType,
    get_params,
)
import math

class CapacityOptimizer:

    def __init__(self, params_provider: ParamsProvider):
        self._params = params_provider

    @property
    def generation_parameters(self) -> GenParams:
        return self._params.current()

    def pick_indices(self, far: str) -> tuple[int, int, int, int]:
        """
        Возвращает индексы (L_idx, W_idx, H_idx, F_idx)
        под сценарий FAR: 'min' / 'mean' / 'max'.

        ВАЖНО: индексы всегда неотрицательные, чтобы нормально работать в циклах.
        """
        if far == "min":
            L_idx = 0
            W_idx = 0
            H_idx = 0
            F_idx = len(self.generation_parameters.plot_side) - 1  
        elif far == "mean":
            L_idx = len(self.generation_parameters.building_length_range) // 2
            W_idx = len(self.generation_parameters.building_width_range) // 2
            H_idx = len(self.generation_parameters.building_height) // 2
            F_idx = len(self.generation_parameters.plot_side) // 2
        elif far == "max":
            L_idx = len(self.generation_parameters.uilding_length_range) - 1
            W_idx = len(self.generation_parameters.building_width_range) - 1
            H_idx = len(self.generation_parameters.building_height) - 1
            F_idx = 0
        else:
            raise ValueError(f"Unknown FAR scenario: {far!r}")
        return L_idx, W_idx, H_idx, F_idx

    def get_plot_area_params(self, far: str) -> tuple[float, float, float]:
        """
        Возвращает (area_min, area_max, area_base) для участка под сценарий FAR.

        Логика:
        • FAR='min'  -> базовая площадь ближе к максимуму (разреженная застройка),
        • FAR='mean' -> середина диапазона,
        • FAR='max'  -> базовая площадь ближе к минимуму (плотная застройка).
        """
        if far == "min":
            area_base = self.generation_parameters.plot_area_max
        elif far == "mean":
            area_base = 0.5 * (self.generation_parameters.plot_area_min + self.generation_parameters.plot_area_max)
        elif far == "max":
            area_base = self.generation_parameters.plot_area_min
        else:
            raise ValueError(f"Unknown FAR scenario for plot area: {far!r}")
        return self.generation_parameters.plot_area_min, self.generation_parameters.plot_area_max, area_base


    def solve_private_block_initial(self,
        target_la: float,
        far: str,
        *,
        la_ratio: float = self.generation_parameters.la_coef,
    ) -> dict:
        """
        ШАГ 1: подсчёт ИСХОДНЫХ параметров дома/участка для квартала
        под заданный сценарий FAR ('min' / 'mean' / 'max').

        Здесь мы:
        1) выбираем (building_length, building_width, floors, plot_front)
            по сценарию FAR через pick_indices;
        2) выбираем базовую площадь участка (plot_area_base) через get_plot_area_params;
        3) считаем living_per_building = L * W * H * la_ratio;
        4) считаем building_need = ceil(target_la / living_per_building);
        5) считаем сценарный FAR (far_target) для дома на базовом участке.

        НИЧЕГО не делаем с периметром и сегментами — это будет на следующих шагах.
        """
        area_min, area_max, area_base = self.generation_parameters.get_plot_area_params(far)

        base_L_idx, base_W_idx, base_h_idx, base_front_idx = self.generation_parameters.pick_indices(far)

        L = float(self.generation_parameters.building_length_range[base_L_idx])
        W = float(self.generation_parameters.building_width_range[base_W_idx])
        H = float(self.generation_parameters.building_height[base_h_idx])
        F = float(self.generation_parameters.plot_side[base_front_idx])

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

        building_params = get_params(building_type)
        target_la = float(row["la_target"])

        res = self.solve_private_block_initial(
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
        """
        Оборачивает compute_block: принимает GeoDataFrame кварталов
        и возвращает тот же GeoDataFrame с добавленными расчётными колонками.
        """
        base_cols = blocks_gdf.apply(
            lambda row: self.compute_block(row, far=far),
            axis=1,
        )
        return pd.concat([blocks_gdf, base_cols], axis=1)