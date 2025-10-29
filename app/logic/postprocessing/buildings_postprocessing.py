from __future__ import annotations

from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from shapely.ops import unary_union

from app.logic.postprocessing.generation_params import GenParams
from app.logic.postprocessing.grid_operations import GridOperations


@dataclass
class BuildingsPostProcessor:
    """
    Пост‑процессинг построек и сервисов:
      • Поиск «диагональных» компонент и построение буферов‑лент
      • Формирование прямоугольников зданий (жилые ячейки + диагональные ленты + сервисные полигоны)
      • Слияние прямоугольников по типу сервиса (жильё отдельно) с предикатом близости
      • Метрики равномерности для площадок сервисов
      • Финализация атрибутов зданий (capacity, floors_count, service, is_living)

    Зависимости:
      • GenParams — численные параметры
      • GridOps — sjoin/make_valid/components/pca_order
    """
    def __init__(self, grid_operations: GridOperations, generation_parameters: GenParams):
        self.generation_parameters = generation_parameters
        self.grid_operations = grid_operations

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

    @staticmethod
    def concat_rects(
        living_rects: gpd.GeoDataFrame,
        diag_rects: gpd.GeoDataFrame,
        service_polys_attrs: List[Dict],
        service_polys_geom: List,
        *,
        crs,
    ) -> gpd.GeoDataFrame:
        svc_rects = gpd.GeoDataFrame(service_polys_attrs, geometry=service_polys_geom, crs=crs)
        out = pd.concat([living_rects, diag_rects, svc_rects], ignore_index=True)
        return gpd.GeoDataFrame(out, geometry="geometry", crs=crs)

    def merge_by_service(
        self,
        buildings_rects: gpd.GeoDataFrame,
        *,
        zid_col: str,
    ) -> gpd.GeoDataFrame:
        left = buildings_rects[["geometry"]].reset_index().rename(columns={"index": "i"})
        right = buildings_rects[["geometry"]].reset_index().rename(columns={"index": "j"})
        pairs = self.grid_operations.sjoin(left, right, predicate=self.generation_parameters.merge_predicate)
        pairs = pairs[(pairs["i"] != pairs["j"]) & pairs["j"].notna()].copy()
        pairs["j"] = pairs["j"].astype(int)

        svc_vals = buildings_rects.get("service").astype(object).tolist()

        def _same_group(i: int, j: int) -> bool:
            return svc_vals[i] == svc_vals[j]

        pairs = pairs[pairs.apply(lambda r: _same_group(int(r["i"]), int(r["j"])), axis=1)]
        adj: DefaultDict[int, List[int]] = DefaultDict(list)
        for a, b in pairs[["i", "j"]].itertuples(index=False):
            adj[a].append(b)
            adj[b].append(a)
        groups = self.grid_operations.components(list(adj.keys()), adj)

        merged_geoms, merged_attrs = [], []
        for gid, comp in enumerate(groups):
            geoms = buildings_rects.geometry.iloc[comp].tolist()
            if self.generation_parameters.merge_fix_eps and self.generation_parameters.merge_fix_eps > 0:
                u = unary_union([self.grid_operations.make_valid(g.buffer(self.generation_parameters.merge_fix_eps)) for g in geoms]).buffer(-self.generation_parameters.merge_fix_eps)
            else:
                u = unary_union([self.grid_operations.make_valid(g) for g in geoms])
            comp_svc = list({svc_vals[i] for i in comp})
            merged_service = comp_svc[0] if len(comp_svc) == 1 else "mixed"
            types = ",".join(sorted(set(buildings_rects.loc[comp, "type"].astype(str).tolist())))
            zid_mode = None
            if zid_col in buildings_rects.columns:
                zids = buildings_rects.loc[comp, zid_col].dropna().astype(int)
                if len(zids) > 0:
                    zid_mode = int(zids.value_counts().index[0])
            merged_geoms.append(self.grid_operations.make_valid(u))
            merged_attrs.append(
                {
                    "group_id": int(gid),
                    "n_members": int(len(comp)),
                    "n_cells_sum": int(np.nansum(buildings_rects.loc[comp, "n_cells"].values)) if "n_cells" in buildings_rects else None,
                    "types": types,
                    "service": merged_service,
                    zid_col: zid_mode,
                }
            )

        buildings_merged = gpd.GeoDataFrame(merged_attrs, geometry=merged_geoms, crs=buildings_rects.crs).reset_index(drop=True)
        min_area = float(self.generation_parameters.cell_size_m) * float(self.generation_parameters.cell_size_m)
        buildings_merged = buildings_merged[buildings_merged.area > min_area]
        return buildings_merged

    def sites_uniformity_metrics(self, service_sites_gdf: gpd.GeoDataFrame, *, zid_col: str) -> pd.DataFrame:
        metrics_rows: List[Dict] = []
        if len(service_sites_gdf) == 0:
            return pd.DataFrame(metrics_rows)
        sites = service_sites_gdf.copy()
        sites["centroid"] = sites.geometry.representative_point()
        for (zid, svc), sub in sites.groupby([zid_col, "service"], dropna=True):
            pts = list(sub["centroid"])
            n = len(pts)
            if n == 1:
                min_nn = float("inf")
                mean_nn = float("inf")
                score = 1.0
            else:
                dmat = np.zeros((n, n), dtype=float)
                for i in range(n):
                    for j in range(i + 1, n):
                        d = pts[i].distance(pts[j])
                        dmat[i, j] = dmat[j, i] = d
                nn = np.min(np.where(dmat == 0, np.inf, dmat), axis=1)
                min_nn = float(np.min(nn))
                mean_nn = float(np.mean(nn))
                score = float(mean_nn / (self.generation_parameters.cell_size_m))
            metrics_rows.append(
                {
                    zid_col: int(zid) if pd.notna(zid) else None,
                    "service": svc,
                    "sites_count": int(n),
                    "min_nn_m": None if np.isinf(min_nn) else float(min_nn),
                    "mean_nn_m": None if np.isinf(mean_nn) else float(mean_nn),
                    "uniformity_score": score,
                }
            )
        return pd.DataFrame(metrics_rows)

    def finalize_buildings(
        self,
        cells: gpd.GeoDataFrame,
        buildings_merged: gpd.GeoDataFrame,
        service_sites_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        mask_b = cells["is_building"].fillna(False)
        cells.loc[mask_b, "service"] = cells.loc[mask_b, "service"].fillna("living_house")

        service_buildings = buildings_merged[buildings_merged["service"] != "living_house"].copy()
        living_buildings = buildings_merged[buildings_merged["service"] == "living_house"].copy()
        living_buildings["service"] = None
        living_buildings["is_living"] = True

        if len(service_buildings) > 0:
            joined = gpd.sjoin(service_buildings, service_sites_gdf, how="left")
            service_buildings["capacity"] = joined["capacity"].values
            floors_mapping = {"school": 3, "kindergarten": 2, "polyclinics": 4}
            service_buildings["floors_count"] = service_buildings.service.map(floors_mapping)
            service_buildings["service"] = service_buildings.apply(
                lambda x: [{x["service"]: x["capacity"]}], axis=1
            )
            service_buildings["is_living"] = False

        buildings = pd.concat([living_buildings, service_buildings], ignore_index=True)
        return gpd.GeoDataFrame(buildings, geometry="geometry", crs=buildings_merged.crs)
