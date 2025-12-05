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
    Helper class for block-level capacity planning in generation pipeline.

    It uses building parameter presets (via BuildingParamsProvider) and a chosen
    FAR scenario ("min", "mean", "max") to:

    - select representative building and plot dimensions;
    - estimate usable area per building (living/functional, depending on target);
    - compute the required number of buildings to reach target area;
    - derive initial FAR and plot geometry attributes for each block.

    The main entry points are:
    - solve_block_initial(...) – compute parameters for a single block;
    - compute_block(...) – wrapper for a pandas row with mode/target_col;
    - compute_blocks_for_gdf(...) – apply the logic to an entire GeoDataFrame.

    Supported modes:
    - "residential":
        * target_col="la_target" (living area, m²)
        * building type выбирается по floors_group / floors_avg для жилых типов
    - "non_residential":
        * target_col="functional_target" (non-res area, m²)
        * building type выбирается по zone + floors_avg (IND_*, TR_*, SPEC_*)
    - "mixed":
        * базовый capacity считается по target_col (обычно "la_target"),
          но дальнейшая многокритериальная оптимизация учитывает
          и la_target, и functional_target глубже по пайплайну
        * building type для блоков business/unknown выбирается по
          zone + floors_avg (BIZ_*)
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

    def _infer_building_type(self, row: pd.Series, mode: str) -> BuildingType | None:
        """
        Infer BuildingType from row data:

        - сначала пробуем floors_group как прямое значение Enum;
        - если не получилось, используем zone + floors_avg;
        - residential → IZH/MKD_*;
        - business/unknown → BIZ_*;
        - industrial → IND_*;
        - transport → TR_*;
        - special → SPEC_*.
        """
        zone = str(row.get("zone", "")).strip().lower()
        floors_group = row.get("floors_group")
        floors_avg_raw = row.get("floors_avg")
        if isinstance(floors_group, str) and floors_group:
            fg = floors_group.strip()
            try:
                return BuildingType(fg)
            except ValueError:
                try:
                    return BuildingType[fg]
                except KeyError:
                    pass

        floors_avg: float | None
        try:
            if floors_avg_raw is None:
                floors_avg = None
            else:
                floors_avg_val = float(floors_avg_raw)
                if math.isnan(floors_avg_val):
                    floors_avg = None
                else:
                    floors_avg = floors_avg_val
        except (TypeError, ValueError):
            floors_avg = None
        try:
            geom = row.geometry
            block_area = float(geom.area) if geom is not None else 0.0
        except Exception:
            block_area = 0.0
        if zone == "residential" or mode == "residential":
            if floors_avg is None:
                return BuildingType.MKD_5_8
            if floors_avg < 3:
                return BuildingType.MKD_2_4
            if floors_avg < 9:
                return BuildingType.MKD_5_8
            if floors_avg < 17:
                return BuildingType.MKD_9_16
            return BuildingType.HIGHRISE
        if zone in {"business", "unknown"}:
            if floors_avg is None:
                return BuildingType.BIZ_MID

            if floors_avg <= 4:
                if block_area >= 15000.0:
                    return BuildingType.BIZ_MALL
                else:
                    return BuildingType.BIZ_LOW
            if floors_avg <= 8:
                return BuildingType.BIZ_MID
            return BuildingType.BIZ_TOWER

        if zone == "industrial":
            if floors_avg is None:
                return BuildingType.IND_LIGHT
            if floors_avg < 2:
                return BuildingType.IND_WAREHOUSE
            if floors_avg < 4:
                return BuildingType.IND_LIGHT
            return BuildingType.IND_HEAVY

        if zone == "transport":
            if floors_avg is None:
                return BuildingType.TR_STATION
            if floors_avg <= 2:
                return BuildingType.TR_DEPOT
            if floors_avg <= 4:
                return BuildingType.TR_STATION
            return BuildingType.TR_PARKING

        if zone == "special":
            if block_area >= 20000.0:
                return BuildingType.SPEC_WASTE
            return BuildingType.SPEC_TECH
        return None

    def solve_block_initial(
        self,
        target_area: float,
        far: str,
        *,
        building_params: BuildingParams,
        la_ratio: float | None = None,
    ) -> dict:
        """
        Solve initial capacity for a block given a target "usable" area.

        target_area:
            For residential: living area target (la_target).
            For non-res: functional area target (functional_target).
            Units must be m² in both cases.
        """

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

        base_house_area = L * W
        usable_per_building = base_house_area * H * la_ratio

        if target_area > 0 and usable_per_building > 0:
            building_need = int(math.ceil(target_area / usable_per_building))
        else:
            building_need = 0

        far_target = (
            (base_house_area * H) / plot_area_base
            if plot_area_base > 0
            else float("nan")
        )

        return {
            "building_length": L,
            "building_width": W,
            "floors": H,
            "plot_front": F,
            "plot_depth": plot_depth_base,
            "plot_area": plot_area_base,
            "living_per_building": float(usable_per_building),
            "building_need": int(building_need),
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
        """
        Compute capacity parameters for a single block (row) given:

        - FAR scenario `far`,
        - target column name (`la_target` / `functional_target`),
        - generation mode (residential / non_residential / mixed).
        """
        try:
            building_type = self._infer_building_type(row, mode=mode)
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
        """
        Apply compute_block(...) to each row of blocks_gdf and
        concatenate the resulting capacity columns.
        """

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
