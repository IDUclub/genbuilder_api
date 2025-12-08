from __future__ import annotations

import os
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import geopandas as gpd

from shapely.geometry import Polygon, Point, box
from shapely.affinity import rotate
from shapely.prepared import prep
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


class MIR:
    """
    Packs maximal axis-aligned rectangles inside polygon geometries using
    grid-based rasterization and a largest-rectangle-in-histogram algorithm.
    """
    @staticmethod
    def _main_axis_angle(poly: Polygon) -> float:

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
    def _max_rectangle_in_histogram(
        heights: np.ndarray,
    ) -> Tuple[int, Optional[int], Optional[int], Optional[int]]:
        stack: List[Tuple[int, int]] = []
        best_area = 0
        best_left = None
        best_right = None
        best_height = None

        n = len(heights)
        for i in range(n + 1):
            h = heights[i] if i < n else 0
            start = i

            while stack and stack[-1][1] > h:
                idx, height = stack.pop()
                area = height * (i - idx)
                if area > best_area:
                    best_area = area
                    best_left = idx
                    best_right = i
                    best_height = height
                start = idx

            stack.append((start, h))

        return best_area, best_left, best_right, best_height

    def _find_largest_rect_in_mask(
        self,
        mask: np.ndarray,
        min_cells: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        rows, cols = mask.shape
        heights = np.zeros(cols, dtype=int)

        best_area = 0
        best_top = best_bottom = best_left = best_right = None

        for i in range(rows):
            row = mask[i]
            heights = np.where(row == 1, heights + 1, 0)

            area_cells, left, right, height = self._max_rectangle_in_histogram(heights)
            if area_cells == 0 or left is None or right is None or height is None:
                continue

            width_cells = right - left
            height_cells = height
            if width_cells < min_cells or height_cells < min_cells:
                continue

            if area_cells > best_area:
                best_area = area_cells
                best_left = left
                best_right = right
                best_bottom = i
                best_top = i - height_cells + 1

        if best_area == 0 or best_left is None or best_right is None:
            return None

        return best_top, best_bottom, best_left, best_right

    def _rasterize_polygon_to_mask(
        self,
        poly_r: Polygon,
        step: float,
    ) -> Tuple[np.ndarray, float, float, float, float]:
        minx, miny, maxx, maxy = poly_r.bounds

        if maxx <= minx or maxy <= miny:
            return np.zeros((0, 0), dtype=np.uint8), minx, miny, maxx, maxy

        cols = max(1, int(math.ceil((maxx - minx) / step)))
        rows = max(1, int(math.ceil((maxy - miny) / step)))

        mask = np.zeros((rows, cols), dtype=np.uint8)

        prepared = prep(poly_r)

        for i in range(rows):
            y_center = miny + (i + 0.5) * step
            for j in range(cols):
                x_center = minx + (j + 0.5) * step
                if prepared.contains(Point(x_center, y_center)):
                    mask[i, j] = 1

        return mask, minx, miny, maxx, maxy

    def _pack_rectangles_in_poly(
        self,
        poly: Polygon,
        step: float,
        min_side: float,
    ) -> List[Dict[str, Any]]:

        if poly.is_empty or poly.area == 0:
            return []

        angle = self._main_axis_angle(poly)
        origin = poly.centroid

        poly_r = rotate(poly, -angle, origin=origin)

        mask, minx, miny, maxx, maxy = self._rasterize_polygon_to_mask(poly_r, step=step)
        if mask.size == 0:
            return []

        rows, cols = mask.shape

        min_cells = max(1, int(math.ceil(min_side / step)))

        results: List[Dict[str, Any]] = []

        while True:
            best = self._find_largest_rect_in_mask(mask, min_cells=min_cells)
            if best is None:
                break

            top, bottom, left, right = best

            width_cells = right - left
            height_cells = bottom - top + 1

            if width_cells < min_cells or height_cells < min_cells:
                break

            rect_minx = minx + left * step
            rect_maxx = minx + right * step 
            rect_miny = miny + top * step
            rect_maxy = miny + (bottom + 1) * step

            rect_r = box(rect_minx, rect_miny, rect_maxx, rect_maxy)
            rect = rotate(rect_r, angle, origin=origin)

            width = width_cells * step
            height = height_cells * step
            area = rect.area

            results.append(
                {
                    "geometry": rect,
                    "width": float(width),
                    "height": float(height),
                    "area": float(area),
                    "angle": float(angle),
                }
            )

            mask[top : bottom + 1, left:right] = 0

        return results

    def _pack_rectangles_for_single_geom(
        self, idx: int, geom: Polygon, step: float, min_side: float
    ) -> List[Dict[str, Any]]:
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

    def pack_inscribed_rectangles_for_gdf(
        self,
        gdf: gpd.GeoDataFrame,
        step: float = 5.0,
        min_side: float = 40.0,
        n_jobs: int = 1,
    ) -> gpd.GeoDataFrame:

        records: List[Dict[str, Any]] = []

        items = list(gdf["geometry"].items())
        if not items:
            return gpd.GeoDataFrame(
                columns=["src_index", "rect_id", "geometry"],
                geometry="geometry",
                crs=gdf.crs,
            )

        if n_jobs == 1:
            for idx, geom in tqdm(items, desc="Packing rectangles", leave=False):
                recs = self._pack_rectangles_for_single_geom(idx, geom, step, min_side)
                records.extend(recs)
        else:
            max_workers = min(n_jobs, len(items), (os.cpu_count() or 1) * 2)
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(
                        self._pack_rectangles_for_single_geom,
                        idx,
                        geom,
                        step,
                        min_side,
                    )
                    for idx, geom in items
                ]

                for f in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Packing rectangles (parallel)",
                    leave=False,
                ):
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
