from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import geopandas as gpd

from shapely.geometry import Polygon, Point, box
from shapely.affinity import rotate
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

class MIR:
    """
    Упаковка максимальных вписанных прямоугольников в полигоны кварталов.

    Логика:
      1) Для каждого полигона оценивается угол его главной оси
         (через minimum_rotated_rectangle).
      2) Полигон поворачивается так, чтобы главная ось стала горизонтальной.
      3) В повернутом полигоне ищется ОДИН прямоугольник максимальной площади,
         ориентированный по осям, с минимальной стороной >= min_side.
      4) Найденный прямоугольник вырезается из полигона, цикл повторяется,
         пока возможно найти новый прямоугольник.
      5) Все прямоугольники поворачиваются обратно в исходную систему координат.

    Публичный API:
      • fit_transform(gdf) -> GeoDataFrame прямоугольников
      • статический враппер pack_inscribed_rectangles_for_gdf(...) для обратной совместимости

    Выходной GeoDataFrame содержит:
      - src_index: индекс исходного полигона (индекс строки входного gdf)
      - rect_id: порядковый номер прямоугольника внутри полигона (0,1,2,...)
      - width, height: габариты прямоугольника в повернутой системе, м
      - area: площадь прямоугольника, м²
      - angle: угол поворота в градусах (ориентация длинной стороны)
      - geometry: геометрия прямоугольника в ИСХОДНОЙ системе координат
    """

    def __init__(
        self, params_provider: ParamsProvider):
        self._params = params_provider

    @property
    def generation_parameters(self) -> GenParams:
        return self._params.current()
    
    def _pack_rectangles_for_single_geom(self, idx: int, geom: Polygon, step: float, min_side: float):
        """
        Обрабатывает один объект gdf: индекс + геометрия.
        Возвращает список records (dict), готовых к сборке в GeoDataFrame.
        """

        if geom is None or geom.is_empty:
            return []


        if geom.geom_type == "Polygon":
            polys = [geom]
        else:

            polys = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]

        records: List[Dict[str, Any]] = []
        rect_counter = 0

        for poly in polys:
            rects = self._pack_rectangles_in_poly(poly, step=step, min_side=min_side)
            for r in rects:
                r["src_index"] = idx
                r["rect_id"] = rect_counter
                rect_counter += 1
                records.append(r)

        return records



    @staticmethod
    def _main_axis_angle(poly: Polygon) -> float:
        """
        Оценка угла главной оси полигона по minimum_rotated_rectangle.
        Возвращает угол в градусах (ориентация длинной стороны).
        """
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)

        if len(coords) < 2:
            return 0.0

        best_len = 0.0
        best_angle = 0.0

        for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if length > best_len:
                best_len = length
                best_angle = angle

        return best_angle




    @staticmethod
    def _max_inscribed_rect_in_rotated_poly(
        poly_r: Polygon,
        step: float,
        min_side: float,
    ) -> Optional[Dict[str, Any]]:
        """
        poly_r — полигон уже повернут в систему, где главная ось горизонтальна.
        Ищет ОДИН прямоугольник максимальной площади, ориентированный по осям,
        с минимальной стороной >= min_side.
        Возвращает dict с geometry (в повернутой системе) и метаданными, либо None.
        """
        if poly_r.is_empty or poly_r.area == 0:
            return None

        minx, miny, maxx, maxy = poly_r.bounds
        if (maxx - minx) < min_side or (maxy - miny) < min_side:
            return None

        xs = np.arange(minx, maxx, step)
        ys = np.arange(miny, maxy, step)

        best_rect = None
        best_area = 0.0
        best_width = 0.0
        best_height = 0.0
        best_center = (None, None)

        for x in xs:
            for y in ys:
                p = Point(x, y)
                if not p.within(poly_r):
                    continue


                half_w = half_h = step / 2.0


                while True:
                    improved = False


                    cand = box(
                        x - (half_w + step / 2.0),
                        y - half_h,
                        x + (half_w + step / 2.0),
                        y + half_h,
                    )
                    if cand.within(poly_r):
                        half_w += step / 2.0
                        improved = True


                    cand = box(
                        x - half_w,
                        y - (half_h + step / 2.0),
                        x + half_w,
                        y + (half_h + step / 2.0),
                    )
                    if cand.within(poly_r):
                        half_h += step / 2.0
                        improved = True

                    if not improved:
                        break

                width = 2 * half_w
                height = 2 * half_h


                if width < min_side or height < min_side:
                    continue

                rect_r = box(x - half_w, y - half_h, x + half_w, y + half_h)
                area = rect_r.area

                if area > best_area:
                    best_area = area
                    best_rect = rect_r
                    best_width = width
                    best_height = height
                    best_center = (x, y)

        if best_rect is None:
            return None

        return {
            "geometry": best_rect,
            "center_x": float(best_center[0]),
            "center_y": float(best_center[1]),
            "width": float(best_width),
            "height": float(best_height),
            "area": float(best_area),
        }




    def _pack_rectangles_in_poly(self, 
        poly: Polygon,
        step: float,
        min_side: float,
    ) -> List[Dict[str, Any]]:
        """
        poly — исходный полигон (в CRS с метрами).
        Логика:
        1) оцениваем угол главной оси,
        2) поворачиваем полигон,
        3) пока можно:
            - ищем максимальный прямоугольник в текущем остатке,
            - вычитаем его,
            - сохраняем результат.
        Возвращает список dict-ов с geometry уже В ИСХОДНОЙ системе координат.
        """
        if poly.is_empty or poly.area == 0:
            return []

        angle = self._main_axis_angle(poly)
        origin = poly.centroid


        remaining = rotate(poly, -angle, origin=origin)

        results: List[Dict[str, Any]] = []
        tol = 0.01

        while True:
            if remaining.is_empty or remaining.area < (min_side * min_side):
                break

            best = self._max_inscribed_rect_in_rotated_poly(
                remaining, step=step, min_side=min_side
            )
            if best is None:
                break

            rect_r = best["geometry"]


            rect = rotate(rect_r, angle, origin=origin)


            results.append(
                {
                    "geometry": rect,
                    "width": best["width"],
                    "height": best["height"],
                    "area": rect.area,
                    "angle": float(angle),
                }
            )


            remaining = remaining.difference(rect_r.buffer(tol))




        return results




    def pack_inscribed_rectangles_for_gdf(self, 
        gdf: gpd.GeoDataFrame,
        step: float = 5.0,
        min_side: float = 40.0,
        geometry_column: str = "geometry",
        n_jobs: int = 1,
    ) -> gpd.GeoDataFrame:
        """
        Принимает GeoDataFrame с полигонами (CRS в метрах!),
        возвращает ОТДЕЛЬНЫЙ GeoDataFrame с упакованными прямоугольниками.

        Колонки:
        - src_index: индекс исходного полигона
        - rect_id: порядковый номер прямоугольника внутри полигона (0,1,2,...)
        - width, height, area, angle, geometry

        n_jobs:
        - 1  — без параллелизации (как раньше),
        - >1 — параллелизация по объектам gdf через ProcessPoolExecutor.
        """
        records: List[Dict[str, Any]] = []


        items = list(gdf[geometry_column].items())
        if not items:
            return gpd.GeoDataFrame(
                columns=["src_index", "rect_id", "geometry"],
                geometry="geometry",
                crs=gdf.crs,
            )


        if n_jobs == 1:
            for idx, geom in tqdm(items, desc="Packing rectangles", leave=False):
                recs = self._pack_rectangles_for_single_geom((idx, geom, step, min_side))
                records.extend(recs)

        else:


            max_workers = min(n_jobs, len(items))
            with ProcessPoolExecutor(max_workers=max_workers) as ex:

                futures = [
                    ex.submit(self._pack_rectangles_for_single_geom, (idx, geom, step, min_side))
                    for idx, geom in items
                ]

                for f in tqdm(as_completed(futures), total=len(futures), desc="Packing rectangles (parallel)", leave=False):
                    recs = f.result()
                    records.extend(recs)

        if not records:
            return gpd.GeoDataFrame(
                columns=["src_index", "rect_id", "geometry"],
                geometry="geometry",
                crs=gdf.crs,
            )

        rects_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)
        return rects_gdf