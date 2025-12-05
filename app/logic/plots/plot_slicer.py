from __future__ import annotations

from typing import List, Tuple

import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.errors import TopologicalError
from shapely import affinity


class PlotSegmentSlicer:
    """
    Отвечает только за геометрию:
    - делает кольцо квартала (ring);
    - нарезает один сегмент на участки по plot_front/plot_depth.
    """

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
        _append_geom(corner_tr, "corner_tr", "corner_tr")

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