from __future__ import annotations

import math

import geopandas as gpd
from shapely.errors import GEOSException
from shapely.geometry import Polygon

from app.common.geo_utils import longest_edge_angle_mrr
from app.common.geo_utils import filter_valid_polygons, safe_float

class ResidentialBuildingsGenerator:
    """
    Генератор контуров зданий по участкам (plots).

    Вход: GeoDataFrame участков с геометрией и параметрами здания
    (длина, ширина, этажность, полезная площадь).

    Что делает:
    - фильтрует валидные геометрии (непустые полигоны с положительной площадью);
    - ориентирует здание вдоль самой длинной стороны minimum rotated rectangle;
    - ставит один прямоугольный контур здания по центру участка с заданными L, W;
    - считает:
        * living_area / functional_area — в зависимости от режима;
        * building_area – суммарная площадь по этажам = footprint_area * floors_count;
    - возвращает GeoDataFrame с контурами зданий и ключевыми атрибутами.

    Режимы:
    - mode="residential"
        * вся полезная площадь идёт в living_area, functional_area = 0;

    - mode="non_residential"
        * вся полезная площадь идёт в functional_area, living_area = 0;

    - mode="mixed"
        * на текущем этапе всё трактуем как living_area (functional_area = 0);
          реальный 1:1 баланс реализуется выше по пайплайну, когда
          многокритериальный оптимизатор будет допилен.
    """

    @staticmethod
    def generate_buildings_from_plots(
        plots_gdf: gpd.GeoDataFrame,
        *,
        mode: str = "residential",
        len_col: str = "building_length",
        width_col: str = "building_width",
        floors_col: str = "floors_count",
        area_col: str = "living_area",
    ) -> gpd.GeoDataFrame:
        """
        Преобразовать участки в здания.

        Parameters
        ----------
        plots_gdf : GeoDataFrame
            Участки с заданными параметрами здания.
        mode : {"residential", "non_residential", "mixed"}
            Режим интерпретации полезной площади.
        len_col : str
            Имя колонки с длиной здания (L).
        width_col : str
            Имя колонки с шириной здания (W).
        floors_col : str
            Имя колонки с числом этажей.
        area_col : str
            Имя колонки с «полезной» площадью на участок/здание
            (обычно living_area, но для нежилых это функциональная площадь).

        Returns
        -------
        GeoDataFrame
            Здания с колонками:
            - building_length
            - building_width
            - floors_count
            - living_area
            - functional_area
            - building_area (footprint * floors_count)
            - src_index (если был в plots_gdf)
            - geometry (Polygon)
        """

        mode = str(mode).lower()
        if mode not in {"residential", "non_residential", "mixed"}:
            raise ValueError(f"Unknown buildings generation mode: {mode!r}")

        plots = filter_valid_polygons(plots_gdf)
        if "plot_area" in plots.columns:
            plots = plots[plots["plot_area"] > 0]

        buildings = []

        for idx, r in plots.iterrows():
            poly = r.geometry
            if poly is None or poly.is_empty:
                continue

            L = r.get(len_col)
            W = r.get(width_col)
            if L is None or W is None:
                continue

            try:
                L = float(L)
                W = float(W)
            except (TypeError, ValueError):
                continue

            if L <= 0 or W <= 0:
                continue

            angle = longest_edge_angle_mrr(poly, degrees=False)
            if angle == 0.0 and poly.minimum_rotated_rectangle.is_empty:
                continue

            cx, cy = poly.centroid.x, poly.centroid.y
            ux, uy = math.cos(angle), math.sin(angle)
            vx, vy = -uy, ux

            hL = L / 2.0
            hW = W / 2.0

            p1 = (cx - ux * hL - vx * hW, cy - uy * hL - vy * hW)
            p2 = (cx + ux * hL - vx * hW, cy + uy * hL - vy * hW)
            p3 = (cx + ux * hL + vx * hW, cy + uy * hL + vy * hW)
            p4 = (cx - ux * hL + vx * hW, cy - uy * hL + vy * hW)

            pts = [p1, p2, p3, p4]

            if any((not math.isfinite(x)) or (not math.isfinite(y)) for (x, y) in pts):
                continue

            try:
                building_geom = Polygon(pts)
            except GEOSException:
                try:
                    building_geom = Polygon(pts + [pts[0]])
                except GEOSException:
                    continue

            if building_geom.is_empty:
                continue

            floors_val = r.get(floors_col)
            floors = safe_float(floors_val, default=float("nan"))
            usable_raw = r.get(area_col)
            usable_area = safe_float(usable_raw, default=0.0)
            if mode == "residential":
                living_area = usable_area
                functional_area = 0.0
            elif mode == "non_residential":
                living_area = 0.0
                functional_area = usable_area
            else:
                living_area = usable_area
                functional_area = 0.0
                
            footprint_area = building_geom.area
            if math.isfinite(floors) and floors > 0:
                building_area = footprint_area * floors
            else:
                building_area = float("nan")

            buildings.append(
                {
                    "geometry": building_geom,
                    "building_length": L,
                    "building_width": W,
                    "floors_count": floors_val,
                    "living_area": living_area,
                    "functional_area": functional_area,
                    "building_area": building_area,
                    "src_index": r.get("src_index"),
                }
            )

        if not buildings:
            return gpd.GeoDataFrame(
                columns=[
                    "building_length",
                    "building_width",
                    "floors_count",
                    "living_area",
                    "functional_area",
                    "building_area",
                    "src_index",
                    "geometry",
                ],
                geometry="geometry",
                crs=plots_gdf.crs,
            )

        return gpd.GeoDataFrame(buildings, geometry="geometry", crs=plots_gdf.crs)
