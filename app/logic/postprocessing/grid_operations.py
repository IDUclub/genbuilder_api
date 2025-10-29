from __future__ import annotations

from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

from app.logic.postprocessing.generation_params import GenParams


@dataclass
class GridOperations:
    """
    Универсальные операции над регулярной сеткой и её топологией.

    Ответственности:
      • Индексация клеток в координаты (row_i, col_j)
      • Надёжное вычисление соседей по стороне/диагонали с учётом длины общей грани
      • Вставка разрывов в длинные непрерывные последовательности (enforce_line_blocks)
      • Поиск компонент связности по произвольному списку рёбер
      • PCA‑упорядочивание точек (для построения диагональных «ленточных» буферов)
      • Совместимый пространственный join для разных версий GeoPandas
      • Геометрическая починка (make_valid)

    Параметры берутся из GenParams (edge_share_frac, max_run и др.).
    """
    def __init__(self, generation_parameters: GenParams):
        self.generation_parameters = generation_parameters

    @staticmethod
    def sjoin(
        left: gpd.GeoDataFrame,
        right: gpd.GeoDataFrame,
        *,
        predicate: str,
        how: str = "left",
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """Совместимость gpd.sjoin для разных версий geopandas."""
        try:
            return gpd.sjoin(left, right, predicate=predicate, how=how, **kwargs)
        except TypeError:
            return gpd.sjoin(left, right, op=predicate, how=how, **kwargs)

    @staticmethod
    def make_valid(g: BaseGeometry) -> BaseGeometry:
        try:
            gg = g.buffer(0)
            return gg if not gg.is_empty else g
        except Exception:
            return g

    @staticmethod
    def grid_indices(gdf: gpd.GeoDataFrame) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """Возвращает (row_i, col_j, x0, y0, step_est) по центроидам ячеек."""
        c = gdf.geometry.centroid
        x = c.x.values
        y = c.y.values
        cell_step = np.median(np.sqrt(np.maximum(gdf.geometry.area.values, 1e-9)))
        x0, y0 = float(np.min(x)), float(np.min(y))
        col = np.rint((x - x0) / cell_step).astype(int)
        row = np.rint((y - y0) / cell_step).astype(int)
        return row, col, x0, y0, float(cell_step)

    def compute_neighbors(
        self, cells: gpd.GeoDataFrame
    ) -> Tuple[DefaultDict[int, List[int]], DefaultDict[int, List[int]], DefaultDict[int, List[int]], np.ndarray, np.ndarray]:
        """
        Считает:
          • все соседние клетки (любое касание) — neighbors_all
          • соседство по стороне (достаточная длина общей грани) — neighbors_side
          • соседство только по диагонали — neighbors_diag
          • число непокрытых (outside) соседей для каждой клетки — empty_neighs
          • недостающие соседи до 8‑соседства — missing
        Надёжная версия с проверкой фактической длины общей грани.
        """
        left = cells[["geometry"]].reset_index().rename(columns={"index": "ida"})
        right = cells[["geometry"]].reset_index().rename(columns={"index": "idb"})
        pairs = self.sjoin(left, right, predicate="touches", how="left")
        if "idb" not in pairs.columns and "index_right" in pairs.columns:
            pairs = pairs.rename(columns={"index_right": "idb"})
        pairs = pairs[(pairs["ida"] != pairs["idb"]) & pairs["idb"].notna()].copy()

        neighbors_all: DefaultDict[int, List[int]] = DefaultDict(list)
        neighbors_side: DefaultDict[int, List[int]] = DefaultDict(list)
        neighbors_diag: DefaultDict[int, List[int]] = DefaultDict(list)

        if len(pairs) > 0:
            pairs["idb"] = pairs["idb"].astype(int)
            geom_list = list(cells.geometry.values)
            edge_len_est = np.sqrt(np.maximum(cells.geometry.area.values, 1e-9))
            thr_len = self.generation_parameters.edge_share_frac * edge_len_est

            def _is_edge_neighbor(a: int, b: int) -> bool:
                try:
                    inter = geom_list[a].boundary.intersection(geom_list[b].boundary)
                    length = getattr(inter, "length", 0.0)
                except Exception:
                    length = 0.0
                return length >= min(thr_len[a], thr_len[b])

            for a, b in pairs[["ida", "idb"]].itertuples(index=False):
                eok = _is_edge_neighbor(a, b)
                neighbors_all[a].append(b)
                (neighbors_side if eok else neighbors_diag)[a].append(b)

            for i in range(len(cells)):
                for dct in (neighbors_all, neighbors_side, neighbors_diag):
                    for j in list(dct[i]):
                        if i not in dct[j]:
                            dct[j].append(i)

        inside = cells.get("inside_iso_closed", False)
        inside = pd.Series(inside).fillna(False).to_numpy().astype(bool)
        empty_neighs = np.zeros(len(cells), dtype=int)
        missing = np.zeros(len(cells), dtype=int)
        for i in range(len(cells)):
            nn = list(set(neighbors_all.get(i, [])))
            empty_neighs[i] = sum(1 for j in nn if not inside[j])
            missing[i] = max(0, 8 - len(nn))
        return neighbors_all, neighbors_side, neighbors_diag, empty_neighs, missing

    def enforce_line_blocks(
        self,
        df: gpd.GeoDataFrame,
        *,
        line_key: str,
        order_key: str,
        mask_key: str,
        max_run: int | None = None,
    ) -> pd.Series:
        """
        На каждой «линии» (фиксированный `line_key`), проходя по возрастанию `order_key`,
        запрещает слишком длинные сплошные блоки True в `mask_key`, вставляя «дыры».
        """
        if max_run is None:
            max_run = int(self.generation_parameters.max_run)
        out = df[mask_key].copy()
        for _, sub in df.loc[df[mask_key]].groupby(line_key):
            sub = sub.sort_values(order_key)
            idx = sub.index.to_numpy()
            ordv = sub[order_key].to_numpy()
            breaks = np.where(np.diff(ordv) != 1)[0] + 1
            segments = np.split(np.arange(len(ordv)), breaks)
            for seg in segments:
                if len(seg) <= max_run:
                    continue
                run = 0
                place_gap = False
                for k in seg:
                    i = idx[k]
                    if place_gap:
                        out.loc[i] = False
                        place_gap = False
                        run = 0
                    else:
                        if run < max_run:
                            out.loc[i] = True
                            run += 1
                            if run == max_run:
                                place_gap = True
                        else:
                            out.loc[i] = False
                            run = 0
        return out

    @staticmethod
    def components(nodes: List[int], adj: Dict[int, List[int]]) -> List[List[int]]:
        node_set = set(nodes)
        seen, comps = set(), []
        for v in nodes:
            if v in seen:
                continue
            stack, comp = [v], [v]
            seen.add(v)
            while stack:
                u = stack.pop()
                for w in adj.get(u, []):
                    if w in node_set and w not in seen:
                        seen.add(w)
                        stack.append(w)
                        comp.append(w)
            comps.append(comp)
        return comps

    @staticmethod
    def pca_order(pts: np.ndarray) -> np.ndarray:
        if len(pts) <= 2:
            return np.argsort(pts[:, 0] + pts[:, 1])
        P = pts - pts.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(P, full_matrices=False)
        axis = Vt[0]
        t = P @ axis
        return np.argsort(t)
