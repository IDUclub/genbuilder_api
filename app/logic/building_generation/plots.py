from __future__ import annotations

import math
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import Polygon, box
from shapely.errors import TopologicalError
from shapely import affinity
from loguru import logger

from app.logic.building_generation.building_params import (
    BuildingGenParams,
    BuildingParamsProvider,
    BuildingType,
    BuildingParams,
)

from app.logic.postprocessing.generation_params import (
    GenParams,
    ParamsProvider,
)


class PlotsGenerator:
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
        self.plot_indices = []
        self.options_per_plot = {}
        self.current_choice = {}

    @property
    def generation_parameters(self) -> GenParams:
        return self._params.current()

    @property
    def building_generation_parameters(self) -> BuildingGenParams:
        return self._building_params.current()

    @staticmethod
    def _make_side_buffers_rotated(
        poly_rot: Polygon, depth: float
    ) -> List[Tuple[Polygon, str, str]]:

        if poly_rot is None or poly_rot.is_empty or depth <= 0:
            return []

        minx, miny, maxx, maxy = poly_rot.bounds
        if maxx - minx <= 0 or maxy - miny <= 0:
            return []

        left_box = box(minx, miny, minx + depth, maxy)
        right_box = box(maxx - depth, miny, maxx, maxy)
        bottom_box = box(minx, miny, maxx, miny + depth)
        top_box = box(minx, maxy - depth, maxx, maxy)

        left = poly_rot.intersection(left_box)
        right = poly_rot.intersection(right_box)
        bottom = poly_rot.intersection(bottom_box)
        top = poly_rot.intersection(top_box)

        def _safe_intersection(a, b):
            if a is None or b is None or a.is_empty or b.is_empty:
                return None
            inter = a.intersection(b)
            return None if inter.is_empty else inter

        corner_bl = _safe_intersection(left, bottom)
        corner_br = _safe_intersection(right, bottom)
        corner_tl = _safe_intersection(left, top)
        corner_tr = _safe_intersection(right, top)

        def _subtract_corners(base, corners):
            if base is None or base.is_empty:
                return None
            res = base
            for c in corners:
                if c is not None and not c.is_empty:
                    res = res.difference(c)
                    if res.is_empty:
                        return None
            return res

        left_clean = _subtract_corners(left, [corner_bl, corner_tl])
        right_clean = _subtract_corners(right, [corner_br, corner_tr])
        bottom_clean = _subtract_corners(bottom, [corner_bl, corner_br])
        top_clean = _subtract_corners(top, [corner_tl, corner_tr])

        out: List[Tuple[Polygon, str, str]] = []

        def _append_geom(geom, kind: str, side_name: str):
            if geom is None or geom.is_empty:
                return
            if geom.geom_type == "MultiPolygon":
                for g in geom.geoms:
                    if not g.is_empty:
                        out.append((g, kind, side_name))
            else:
                out.append((geom, kind, side_name))

        _append_geom(bottom_clean, "side", "bottom")
        _append_geom(top_clean, "side", "top")
        _append_geom(left_clean, "side", "left")
        _append_geom(right_clean, "side", "right")

        _append_geom(corner_bl, "corner", "corner_bl")
        _append_geom(corner_br, "corner", "corner_br")
        _append_geom(corner_tl, "corner", "corner_tl")
        _append_geom(corner_tr, "corner", "corner_tr")

        return out

    @staticmethod
    def make_block_ring(poly: Polygon, depth: float) -> Polygon:

        if poly is None or poly.is_empty or depth <= 0:
            return poly

        try:
            inner = poly.buffer(-depth)
            ring = poly.difference(inner)
            if ring.is_empty:
                return poly
            return ring
        except TopologicalError:
            return poly

    def _slice_segment_with_plots(self, row, crs=None):

        poly = row.geometry
        if poly is None or poly.is_empty:
            return gpd.GeoDataFrame(columns=list(row.index) + ["geometry"], crs=crs)

        axis = float(row["angle"])
        plot_side = float(row["plot_front"])
        plot_depth = float(row["plot_depth"])

        if plot_side <= 0 or plot_depth <= 0:
            return gpd.GeoDataFrame(columns=list(row.index) + ["geometry"], crs=crs)

        centroid = poly.centroid

        poly_rot = affinity.rotate(poly, -axis, origin=centroid, use_radians=False)

        minx, miny, maxx, maxy = poly_rot.bounds
        width = maxx - minx
        height = maxy - miny

        base_attrs = {k: row[k] for k in row.index if k != "geometry"}

        if 2 * plot_depth >= min(width, height):
            long_side = max(width, height)

            if long_side <= plot_side:
                row_dict = {k: row[k] for k in row.index}
                return gpd.GeoDataFrame([row_dict], geometry="geometry", crs=crs)

            geoms = []
            rows_data = []

            if width >= height:

                x = minx
                while x < maxx:
                    x2 = min(x + plot_side, maxx)
                    cell = box(x, miny, x2, maxy)
                    part = poly_rot.intersection(cell)
                    if not part.is_empty:
                        geom_back = affinity.rotate(
                            part, axis, origin=centroid, use_radians=False
                        )
                        data = dict(base_attrs)
                        data["segment_kind"] = "full"
                        data["segment_side"] = "long_x"
                        geoms.append(geom_back)
                        rows_data.append(data)
                    x = x2
            else:

                y = miny
                while y < maxy:
                    y2 = min(y + plot_side, maxy)
                    cell = box(minx, y, maxx, y2)
                    part = poly_rot.intersection(cell)
                    if not part.is_empty:
                        geom_back = affinity.rotate(
                            part, axis, origin=centroid, use_radians=False
                        )
                        data = dict(base_attrs)
                        data["segment_kind"] = "full"
                        data["segment_side"] = "long_y"
                        geoms.append(geom_back)
                        rows_data.append(data)
                    y = y2

            if not geoms:

                row_dict = {k: row[k] for k in row.index}
                return gpd.GeoDataFrame([row_dict], geometry="geometry", crs=crs)

            return gpd.GeoDataFrame(rows_data, geometry=geoms, crs=crs)

        parts_rot = self._make_side_buffers_rotated(poly_rot, plot_depth)
        if not parts_rot:
            return gpd.GeoDataFrame(columns=list(row.index) + ["geometry"], crs=crs)

        geoms = []
        rows_data = []

        for geom_rot, kind, side_name in parts_rot:
            if geom_rot is None or geom_rot.is_empty:
                continue

            if kind == "corner":
                geom_back = affinity.rotate(
                    geom_rot, axis, origin=centroid, use_radians=False
                )
                data = dict(base_attrs)
                data["segment_kind"] = kind
                data["segment_side"] = side_name
                geoms.append(geom_back)
                rows_data.append(data)
                continue

            gminx, gminy, gmaxx, gmaxy = geom_rot.bounds
            if gmaxx - gminx <= 0 or gmaxy - gminy <= 0:
                continue

            if side_name in ("bottom", "top"):

                x = gminx
                while x < gmaxx:
                    x2 = min(x + plot_side, gmaxx)
                    cell = box(x, gminy, x2, gmaxy)
                    part = geom_rot.intersection(cell)
                    if not part.is_empty:
                        geom_back = affinity.rotate(
                            part, axis, origin=centroid, use_radians=False
                        )
                        data = dict(base_attrs)
                        data["segment_kind"] = kind
                        data["segment_side"] = side_name
                        geoms.append(geom_back)
                        rows_data.append(data)
                    x = x2
            else:

                y = gminy
                while y < gmaxy:
                    y2 = min(y + plot_side, gmaxy)
                    cell = box(gminx, y, gmaxx, y2)
                    part = geom_rot.intersection(cell)
                    if not part.is_empty:
                        geom_back = affinity.rotate(
                            part, axis, origin=centroid, use_radians=False
                        )
                        data = dict(base_attrs)
                        data["segment_kind"] = kind
                        data["segment_side"] = side_name
                        geoms.append(geom_back)
                        rows_data.append(data)
                    y = y2

        if not geoms:
            return gpd.GeoDataFrame(columns=list(row.index) + ["geometry"], crs=crs)

        out = gpd.GeoDataFrame(rows_data, geometry=geoms, crs=crs)
        return out

    @staticmethod
    def merge_small_plots(
        plots: gpd.GeoDataFrame, area_factor: float
    ) -> gpd.GeoDataFrame:

        if plots.empty:
            return plots

        gdf = plots.copy()

        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf[~gdf.geometry.is_empty]
        if gdf.empty:
            return gpd.GeoDataFrame(columns=plots.columns, crs=plots.crs)

        gdf["_target_area"] = gdf["plot_front"] * gdf["plot_depth"]
        gdf["_area"] = gdf.geometry.area
        gdf["_is_small"] = gdf["_area"] < (area_factor * gdf["_target_area"])

        try:
            sindex = gdf.sindex
        except Exception:

            result = gdf.drop(
                columns=["_target_area", "_area", "_is_small"], errors="ignore"
            )
            result.reset_index(drop=True, inplace=True)
            return result

        processed: set[int] = set()
        new_rows = []

        for idx, row in gdf.sort_values("_area").iterrows():
            if idx in processed:
                continue

            geom = row.geometry

            if geom is None or geom.is_empty:
                new_rows.append(row)
                processed.add(idx)
                continue

            if not bool(row["_is_small"]):
                new_rows.append(row)
                processed.add(idx)
                continue

            possible_idx = [i for i in list(sindex.intersection(geom.bounds)) if i != idx and i not in processed]

            best_neighbor_idx = None
            best_shared_len = 0.0

            for j in possible_idx:
                other_geom = gdf.at[j, "geometry"]

                if other_geom is None or other_geom.is_empty:
                    continue

                try:
                    shared_geom = geom.boundary.intersection(other_geom.boundary)
                except Exception:

                    continue

                if shared_geom is None:
                    continue

                try:
                    shared_len = shared_geom.length
                except AttributeError:

                    continue

                if shared_len <= 0:
                    continue

                if shared_len > best_shared_len:
                    best_shared_len = shared_len
                    best_neighbor_idx = j

            if best_neighbor_idx is None or best_shared_len == 0.0:
                new_rows.append(row)
                processed.add(idx)
                continue

            neighbor = gdf.loc[best_neighbor_idx]
            try:
                merged_geom = geom.union(neighbor.geometry)
            except Exception:

                new_rows.append(row)
                processed.add(idx)
                continue

            merged_row = neighbor.copy()
            merged_row.geometry = merged_geom
            merged_row["_area"] = merged_geom.area
            merged_row["_is_small"] = False

            new_rows.append(merged_row)
            processed.add(idx)
            processed.add(best_neighbor_idx)

        result = gpd.GeoDataFrame(new_rows, crs=gdf.crs)
        result = result.drop(
            columns=["_target_area", "_area", "_is_small"], errors="ignore"
        )
        result.reset_index(drop=True, inplace=True)
        return result

    def merge_small_plots_iterative(self, plots, area_factor, max_iters=10):
        gdf = plots.copy()
        for _ in range(max_iters):
            gdf_new = self.merge_small_plots(gdf, area_factor=area_factor)

            target_area = gdf_new["plot_front"] * gdf_new["plot_depth"]
            small_mask = gdf_new.geometry.area < (area_factor * target_area)

            gdf = gdf_new
            if not small_mask.any():
                break

        return gdf

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
        self, direction: int, total_la: float, best_la: float, best_far: float
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

    def _tune_block(self, group: gpd.GeoDataFrame, gdf) -> gpd.GeoDataFrame:

        la_target = float(group["la_target"].iloc[0])
        if la_target <= 0:
            return group

        angle = float(group["angle"].iloc[0])

        block_area = float(group["plot_area"].sum())
        if block_area <= 0:
            return group

        if "floors_group" not in group.columns:
            return group

        floors_group = group["floors_group"].iloc[0]
        try:
            building_type = BuildingType(floors_group)
        except Exception:
            return group

        building_params = self.building_generation_parameters.params_by_type[
            building_type
        ]

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
        self.la_target = la_target

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

        if total_living < la_target:
            total_living, best_la_score, best_far_score = self._hill_climb(
                +1, total_living, best_la_score, best_far_score
            )
        elif total_living > la_target:
            total_living, best_la_score, best_far_score = self._hill_climb(
                -1, total_living, best_la_score, best_far_score
            )

        if total_living < la_target:
            total_living, best_la_score, best_far_score = self._hill_climb(
                -1, total_living, best_la_score, best_far_score
            )
        elif total_living > la_target:
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
        la_diff = total_living - la_target
        la_ratio = total_living / la_target if la_target > 0 else math.nan
        far_diff = far_final - far_initial if not math.isnan(far_initial) else math.nan

        group.loc[:, "total_living_area"] = float(total_living)
        group.loc[:, "la_diff"] = float(la_diff)
        group.loc[:, "la_ratio"] = float(la_ratio)
        group.loc[:, "far_initial"] = float(far_initial)
        group.loc[:, "far_final"] = float(far_final)
        group.loc[:, "far_diff"] = float(far_diff)

        return group

    def _recalc_buildings_for_plots(self, plots: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

        required_cols = {"src_index", "la_target", "angle"}
        if not required_cols.issubset(plots.columns):
            return plots

        gdf = plots.copy()
        gdf["plot_area"] = gdf.geometry.area

        gdf = gdf.groupby("src_index", group_keys=False).apply(
            lambda x: self._tune_block(x, gdf)
        )
        return gdf

    def generate_plots(self, segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        segments = segments.dropna(subset=["plot_depth"]).copy()
        crs = segments.crs

        parts_list = []
        for _, r in segments.iterrows():
            p = self._slice_segment_with_plots(r, crs=crs)
            if p is not None and not p.empty:
                parts_list.append(p)
        if not parts_list:
            return gpd.GeoDataFrame(
                columns=list(segments.columns),
                geometry="geometry",
                crs=crs,
            )

        result = gpd.GeoDataFrame(
            pd.concat(parts_list, ignore_index=True),
            geometry="geometry",
            crs=crs,
        )

        result = self.merge_small_plots_iterative(result, area_factor=0.5)

        result = result[result.geometry.notna() & ~result.geometry.is_empty]
        result = result[result.geom_type.isin(["Polygon", "MultiPolygon"])]

        result["plot_area"] = result.geometry.area
        result = result[result["plot_area"] > 0]

        result = self._recalc_buildings_for_plots(result)

        return result
