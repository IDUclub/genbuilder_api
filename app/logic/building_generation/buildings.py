from __future__ import annotations

import math

import geopandas as gpd
from shapely.errors import GEOSException
from shapely.geometry import Polygon


class ResidentialBuildingsGenerator:
    """
    Generates residential building footprints inside plot polygons.

    Given a GeoDataFrame of plots with preset building dimensions and attributes
    (length, width, floors, living area), this class:
    - filters valid plot geometries (non-empty polygons with positive area);
    - orients each building along the longest edge of the plot’s minimum
        rotated rectangle;
    - places a single rectangular building at the plot centroid with the
        requested length and width;
    - returns a GeoDataFrame of building footprints with geometry and
        key attributes (dimensions, floors, living_area, src_index).

    Main entry point:
    - generate_buildings_from_plots(...) – static method that converts plots
        GeoDataFrame into a buildings GeoDataFrame.
    """

    @staticmethod
    def generate_buildings_from_plots(
        plots_gdf: gpd.GeoDataFrame,
        len_col: str = "building_length",
        width_col: str = "building_width",
        floors_col: str = "floors_count",
        la_col: str = "living_area",
    ) -> gpd.GeoDataFrame:

        plots = plots_gdf.copy()
        plots = plots[plots.geometry.notna() & ~plots.geometry.is_empty]
        plots = plots[plots.geom_type.isin(["Polygon", "MultiPolygon"])]
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

            mrr = poly.minimum_rotated_rectangle

            if mrr.geom_type == "Polygon":
                coords = list(mrr.exterior.coords)[:-1]
            elif mrr.geom_type == "LineString":
                coords = list(mrr.coords)
            else:
                continue

            if len(coords) < 2:
                continue

            max_d = -1.0
            dx = dy = 0.0
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                d = (x2 - x1) ** 2 + (y2 - y1) ** 2
                if d > max_d:
                    max_d = d
                    dx, dy = x2 - x1, y2 - y1

            if max_d <= 0:
                continue

            angle = math.atan2(dy, dx)

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

            buildings.append(
                {
                    "geometry": building_geom,
                    "building_length": L,
                    "building_width": W,
                    "floors_count": r.get(floors_col),
                    "living_area": r.get(la_col),
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
                    "src_index",
                    "geometry",
                ],
                geometry="geometry",
                crs=plots_gdf.crs,
            )

        return gpd.GeoDataFrame(buildings, geometry="geometry", crs=plots_gdf.crs)
