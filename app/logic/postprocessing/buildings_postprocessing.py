from __future__ import annotations

from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry import CAP_STYLE, JOIN_STYLE

from app.logic.postprocessing.generation_params import GenParams, ParamsProvider
from app.logic.postprocessing.grid_operations import GridOperations


class BuildingsPostProcessor:
    """
    Post-processes generated buildings on the grid.

    Finds diagonal-only building chains and turns them into buffer strips,
    creates rectangles for living cells, and returns geometries ready for
    further merging and attribute assignment.
    """
    def __init__(self, grid_operations: GridOperations, params_provider: ParamsProvider):
        self._params = params_provider
        self.grid_operations = grid_operations

    @property
    def generation_parameters(self) -> GenParams:
        return self._params.current()

    def mark_diag_only_and_buffers(
        self,
        cells: gpd.GeoDataFrame,
        *,
        neighbors_side: Dict[int, List[int]],
        neighbors_diag: Dict[int, List[int]],
    ) -> Tuple[List[List[int]], gpd.GeoDataFrame]:
        is_b = cells["is_building"].to_numpy().astype(bool)
        diag_only_nodes: List[int] = []
        for i in range(len(cells)):
            if not is_b[i]:
                continue
            side_in = any(is_b[j] for j in neighbors_side.get(i, []))
            diag_in = any(is_b[j] for j in neighbors_diag.get(i, []))
            if (not side_in) and diag_in:
                diag_only_nodes.append(i)

        diag_adj = {i: [j for j in neighbors_diag.get(i, []) if is_b[j]] for i in range(len(cells))}
        diag_components = self.grid_operations.components(diag_only_nodes, diag_adj)
        diag_components = [comp for comp in diag_components if len(comp) >= 2]

        cells = cells.copy()
        cells["is_diag_only"] = False
        for comp in diag_components:
            for i in comp:
                cells.at[i, "is_diag_only"] = True

        cap_style = CAP_STYLE.flat
        join_style = JOIN_STYLE.mitre
        buf_dist = float(self.generation_parameters.cell_size_m) / 2.0
        centroids = cells.geometry.centroid
        rec_geom: List = []
        rec_attr: List[Dict] = []
        k = 0
        for comp in diag_components:
            pts = np.array([[centroids[i].x, centroids[i].y] for i in comp], dtype=float)
            if len(pts) < 2:
                continue
            order = self.grid_operations.pca_order(pts)
            ordered = pts[order]
            keep = [True]
            for a, b in zip(ordered[:-1], ordered[1:]):
                keep.append(bool(np.any(np.abs(a - b) > 1e-12)))
            ordered = ordered[np.array(keep, dtype=bool)]
            if len(ordered) < 2:
                continue
            line = LineString(ordered)
            poly = line.buffer(buf_dist, cap_style=cap_style, join_style=join_style)
            rec_geom.append(self.grid_operations.make_valid(poly))
            k += 1
            rec_attr.append(
                {
                    "building_id": f"D{str(k).zfill(5)}",
                    "type": "diag_buffer",
                    "service": "living_house",
                    "n_cells": int(len(comp)),
                    "width_m": float(self.generation_parameters.cell_size_m),
                }
            )
        return diag_components, gpd.GeoDataFrame(rec_attr, geometry=rec_geom, crs=cells.crs)

    def living_cell_rects(
        self,
        cells: gpd.GeoDataFrame,
        *,
        zid_col: str,
    ) -> gpd.GeoDataFrame:
        simple_ids = [
            i
            for i in range(len(cells))
            if bool(cells.at[i, "is_building"]) and not bool(cells.at[i, "is_diag_only"])
        ]
        rec_geom: List = []
        rec_attr: List[Dict] = []
        for i in simple_ids:
            rec_geom.append(self.grid_operations.make_valid(cells.geometry[i]))
            zid_mode = (
                int(cells.at[i, zid_col])
                if (zid_col in cells.columns and pd.notna(cells.at[i, zid_col]))
                else None
            )
            rec_attr.append(
                {
                    "building_id": f"C{str(i).zfill(6)}",
                    "type": "cell",
                    "service": "living_house",
                    "n_cells": 1,
                    "width_m": float(self.generation_parameters.cell_size_m),
                    "row_i": int(cells.at[i, "row_i"]),
                    "col_j": int(cells.at[i, "col_j"]),
                    zid_col: zid_mode,
                }
            )
        return gpd.GeoDataFrame(rec_attr, geometry=rec_geom, crs=cells.crs)

    