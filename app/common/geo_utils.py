import math
from typing import Optional

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

def longest_edge_angle_mrr(
    poly: BaseGeometry,
    *,
    degrees: bool = False,
) -> float:
    if poly is None or poly.is_empty:
        return 0.0

    try:
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        if len(coords) < 2:
            return 0.0

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
            return 0.0

        angle = math.atan2(dy, dx)
        if degrees:
            return math.degrees(angle)
        return angle
    except Exception:
        return 0.0

def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        if not math.isfinite(v):
            return default
        return v
    except (TypeError, ValueError):
        return default

def safe_make_valid(geom: Optional[BaseGeometry]) -> Optional[BaseGeometry]:
    if geom is None or geom.is_empty:
        return geom

    try:
        if not geom.is_valid:
            geom = make_valid(geom)
            if not geom.is_valid:
                geom = geom.buffer(0)
    except Exception:
        try:
            geom = geom.buffer(0)
        except Exception:
            return None

    if geom is None or geom.is_empty:
        return None
    return geom

def filter_valid_polygons(
    gdf: gpd.GeoDataFrame,
    *,
    min_area: float = 0.0,
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf

    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]

    if min_area > 0:
        area = gdf.geometry.area
        gdf = gdf[area > min_area]

    return gdf

def ensure_crs(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("GeoDataFrame wthout CRS.")
    if str(gdf.crs) != str(target_crs):
        return gdf.to_crs(target_crs)
    return gdf