from __future__ import annotations

import math
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import affinity
from loguru import logger

from app.logic.building_params import (
    BuildingGenParams,
    BuildingParamsProvider
)
from app.logic.generation_params import (
    GenParams,
    ParamsProvider,
)
from app.logic.building_type_resolver import infer_building_type


class PlotTuner:
    """
    Инкапсулирует:
    - расчет эффективных размеров участков;
    - генерацию дискретных опций L/W/H;
    - hill-climbing по суммарной living area / FAR;
    - запись результатов назад в GeoDataFrame.
    """

    def __init__(
        self,
        params_provider: ParamsProvider,
        building_params_provider: BuildingParamsProvider,
    ):
        self._params = params_provider
        self._building_params = building_params_provider
        self.base_L = None
        self.base_W = None
        self.base_H = None
        self.base_living_per_building = None
        self.block_area = None
        self.far_initial = None
        self.la_target = None
        self.plot_indices: list[int] = []
        self.options_per_plot: dict[int, list[dict]] = {}
        self.current_choice: dict[int, dict] = {}

    @property
    def generation_parameters(self) -> GenParams:
        return self._params.current()

    @property
    def building_generation_parameters(self) -> BuildingGenParams:
        return self._building_params.current()

    def _score_to_base(self, option: dict) -> tuple[float, float, float]:
        return (
            abs(option["living"] - self.base_living_per_building),
            abs(option["footprint"] - (self.base_L * self.base_W)),
            abs(option["H"] - self.base_H),
        )

    def _current_far(self, total_la: float) -> float:
        return (
            total_la / self.block_area
            if self.block_area and self.block_area > 0
            else math.nan
        )

    def _objective(self, total_la: float) -> tuple[float, float]:
        la_score = abs(total_la - self.la_target)
        if math.isnan(self.far_initial):
            far_score = 0.0
        else:
            far_score = abs(self._current_far(total_la) - self.far_initial)
        return la_score, far_score

    def _hill_climb(
        self,
        direction: int,
        total_la: float,
        best_la: float,
        best_far: float,
    ) -> tuple[float, float, float]:

        changed = True
        while changed:
            changed = False
            best_move = None
            best_move_score = (best_la, best_far)
            best_move_total_la = total_la

            for idx in self.plot_indices:
                cur = self.current_choice[idx]
                cur_la = cur["living"]
                options = self.options_per_plot[idx]

                for option in options:
                    if direction == 1 and option["living"] <= cur_la:
                        continue
                    if direction == -1 and option["living"] >= cur_la:
                        continue

                    new_total_la = total_la - cur_la + option["living"]
                    la_s, far_s = self._objective(new_total_la)

                    if (la_s, far_s) < best_move_score:
                        best_move_score = (la_s, far_s)
                        best_move = (idx, option)
                        best_move_total_la = new_total_la

            if best_move is not None:
                idx, option = best_move
                total_la = best_move_total_la
                best_la, best_far = best_move_score
                self.current_choice[idx] = option
                changed = True

        return total_la, best_la, best_far

    def _tune_block(
        self,
        group: gpd.GeoDataFrame,
        gdf: gpd.GeoDataFrame,
        *,
        mode: str = "residential",
        target_col: str = "la_target",
    ) -> gpd.GeoDataFrame:

        if target_col not in group.columns:
            logger.debug(
                f"[_tune_block] src_index={group['src_index'].iloc[0] if 'src_index' in group.columns else 'N/A'}: "
                f"no '{target_col}' column, skip tuning (mode={mode})"
            )
            return group

        try:
            target_val = float(group[target_col].iloc[0])
        except (TypeError, ValueError):
            target_val = 0.0

        if target_val <= 0:
            logger.debug(
                f"[_tune_block] src_index={group['src_index'].iloc[0] if 'src_index' in group.columns else 'N/A'}: "
                f"{target_col} <= 0 ({target_val}), skip tuning (mode={mode})"
            )
            return group

        angle = float(group["angle"].iloc[0])

        block_area = float(group["plot_area"].sum())
        if block_area <= 0:
            logger.debug(
                f"[_tune_block] src_index={group['src_index'].iloc[0] if 'src_index' in group.columns else 'N/A'}: "
                f"block_area <= 0, skip tuning (mode={mode})"
            )
            return group
        try:
            repr_row = group.iloc[0]
            building_type = infer_building_type(
                repr_row, mode=mode
            )
            logger.debug(
                f"[_tune_block] src_index={group['src_index'].iloc[0] if 'src_index' in group.columns else 'N/A'}: "
                f"mode={mode}, zone={repr_row.get('zone')}, floors_group={repr_row.get('floors_group')}, "
                f"floors_avg={repr_row.get('floors_avg')} -> building_type={building_type}"
            )
        except Exception as e:
            building_type = None
            logger.error(
                f"[_tune_block] src_index={group['src_index'].iloc[0] if 'src_index' in group.columns else 'N/A'}: "
                f"cannot infer BuildingType (mode={mode}, zone={group.get('zone', pd.Series([None])).iloc[0]}): {e!r}"
            )

        if building_type is None:
            logger.debug(
                f"[_tune_block] src_index={group['src_index'].iloc[0] if 'src_index' in group.columns else 'N/A'}: "
                f"building_type is None, skip tuning (mode={mode})"
            )
            return group

        try:
            building_params = self.building_generation_parameters.params_by_type[
                building_type
            ]
        except KeyError:
            logger.error(
                f"[_tune_block] src_index={group['src_index'].iloc[0] if 'src_index' in group.columns else 'N/A'}: "
                f"building_type={building_type} not in params_by_type keys="
                f"{list(self.building_generation_parameters.params_by_type.keys())} "
                f"-> skip tuning (mode={mode})"
            )
            return group

        self.base_L = float(
            group.get(
                "building_length",
                pd.Series([building_params.building_length_range[0]]),
            ).iloc[0]
        )
        self.base_W = float(
            group.get(
                "building_width",
                pd.Series([building_params.building_width_range[0]]),
            ).iloc[0]
        )
        self.base_H = float(
            group.get(
                "floors_count",
                pd.Series([building_params.building_height[0]]),
            ).iloc[0]
        )

        self.base_living_per_building = (
            self.base_L * self.base_W * self.base_H * building_params.la_coef
        )

        if "far_initial" in group.columns:
            far_initial = float(group["far_initial"].iloc[0])
        else:
            far_initial = math.nan

        self.block_area = block_area
        self.far_initial = far_initial
        self.la_target = target_val

        self.plot_indices = list(group.index)
        fronts_eff: dict[int, float] = {}
        depths_eff: dict[int, float] = {}
        max_footprints: dict[int, float] = {}

        inner_border = self.generation_parameters.INNER_BORDER
        max_coverage = self.generation_parameters.MAX_COVERAGE

        for idx, row in group.iterrows():
            poly = row.geometry
            area = float(row["plot_area"])

            if poly is None or poly.is_empty or area <= 0:
                fronts_eff[idx] = 0.0
                depths_eff[idx] = 0.0
                max_footprints[idx] = 0.0
                continue

            poly_rot = affinity.rotate(
                poly, -angle, origin=poly.centroid, use_radians=False
            )
            minx, miny, maxx, maxy = poly_rot.bounds

            front = maxx - minx
            depth = maxy - miny

            eff_front = max(front - 2 * inner_border, 0.0)
            eff_depth = max(depth - 2 * inner_border, 0.0)

            fronts_eff[idx] = eff_front
            depths_eff[idx] = eff_depth
            max_footprints[idx] = max_coverage * area

        self.options_per_plot = {}

        for idx in self.plot_indices:
            eff_front = fronts_eff[idx]
            eff_depth = depths_eff[idx]
            max_fp = max_footprints[idx]

            options: list[dict] = []

            if (
                eff_front <= 0
                or eff_depth <= 0
                or max_fp <= 0
                or gdf.loc[idx, "plot_area"] < building_params.plot_area_min
            ):
                self.options_per_plot[idx] = [
                    {
                        "L": 0.0,
                        "W": 0.0,
                        "H": 0.0,
                        "living": 0.0,
                        "footprint": 0.0,
                    }
                ]
                continue

            for L in building_params.building_length_range:
                L = float(L)
                for W in building_params.building_width_range:
                    W = float(W)
                    footprint = L * W
                    if footprint <= 0 or footprint > max_fp:
                        continue

                    fits_by_dims = (L <= eff_front and W <= eff_depth) or (
                        L <= eff_depth and W <= eff_front
                    )
                    if not fits_by_dims:
                        continue

                    for H in building_params.building_height:
                        H = float(H)
                        if H < 1:
                            continue

                        living = footprint * H * building_params.la_coef
                        if living <= 0:
                            continue

                        options.append(
                            {
                                "L": L,
                                "W": W,
                                "H": H,
                                "living": living,
                                "footprint": footprint,
                            }
                        )

            if not options:
                options = [
                    {
                        "L": 0.0,
                        "W": 0.0,
                        "H": 0.0,
                        "living": 0.0,
                        "footprint": 0.0,
                    }
                ]

            self.options_per_plot[idx] = options

        self.current_choice = {}
        total_living = 0.0

        for idx in self.plot_indices:
            options = self.options_per_plot[idx]
            best_opt = min(options, key=self._score_to_base)
            self.current_choice[idx] = best_opt
            total_living += best_opt["living"]

        best_la_score, best_far_score = self._objective(total_living)

        if total_living < target_val:
            total_living, best_la_score, best_far_score = self._hill_climb(
                +1, total_living, best_la_score, best_far_score
            )
        elif total_living > target_val:
            total_living, best_la_score, best_far_score = self._hill_climb(
                -1, total_living, best_la_score, best_far_score
            )
        if total_living < target_val:
            total_living, best_la_score, best_far_score = self._hill_climb(
                -1, total_living, best_la_score, best_far_score
            )
        elif total_living > target_val:
            total_living, best_la_score, best_far_score = self._hill_climb(
                +1, total_living, best_la_score, best_far_score
            )

        group = group.copy()
        for idx in self.plot_indices:
            choice = self.current_choice[idx]
            L = choice["L"]
            W = choice["W"]
            H = choice["H"]
            living = choice["living"]

            if living <= 0:
                group.at[idx, "building_length"] = np.nan
                group.at[idx, "building_width"] = np.nan
                group.at[idx, "floors_count"] = np.nan
                group.at[idx, "living_per_building"] = 0.0
                group.at[idx, "living_area"] = 0.0
            else:
                group.at[idx, "building_length"] = float(L)
                group.at[idx, "building_width"] = float(W)
                group.at[idx, "floors_count"] = float(H)
                group.at[idx, "living_per_building"] = float(living)
                group.at[idx, "living_area"] = float(living)

        far_final = self._current_far(total_living)
        la_diff = total_living - target_val
        la_ratio = total_living / target_val if target_val > 0 else math.nan
        far_diff = far_final - far_initial if not math.isnan(far_initial) else math.nan

        group.loc[:, "total_usable_area"] = float(total_living)
        group.loc[:, "la_diff"] = float(la_diff)
        group.loc[:, "la_ratio"] = float(la_ratio)
        group.loc[:, "far_initial"] = float(far_initial)
        group.loc[:, "far_final"] = float(far_final)
        group.loc[:, "far_diff"] = float(far_diff)

        logger.debug(
            f"[_tune_block] src_index={group['src_index'].iloc[0] if 'src_index' in group.columns else 'N/A'} "
            f"done: mode={mode}, target_col={target_col}, "
            f"target_val={target_val}, total_living={total_living}, la_diff={la_diff}, "
            f"far_initial={far_initial}, far_final={far_final}"
        )

        return group

    def _recalc_buildings_for_plots(
        self,
        plots: gpd.GeoDataFrame,
        *,
        mode: str = "residential",
        target_col: str = "la_target",
    ) -> gpd.GeoDataFrame:

        required_cols = {"src_index", "angle", target_col}
        if not required_cols.issubset(plots.columns):
            logger.debug(
                f"[_recalc_buildings_for_plots] missing required columns {required_cols - set(plots.columns)}, "
                f"skip recalc (mode={mode}, target_col={target_col})"
            )
            return plots

        gdf = plots.copy()
        gdf["plot_area"] = gdf.geometry.area

        gdf = gdf.groupby("src_index", group_keys=False).apply(
            lambda x: self._tune_block(x, gdf, mode=mode, target_col=target_col)
        )
        return gdf