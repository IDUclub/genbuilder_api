from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
from shapely.geometry import (LineString, MultiLineString, MultiPolygon, Point,
                              Polygon)
from shapely.ops import unary_union


@dataclass
class Snapper:
    """
    Centroid snapper to the median line of the “second ring” within a block.

    **Concept:**
        - `ring1`: the zone between 0 and `ring1_width` meters inward from the outer boundary.
        - `ring2`: the zone between `ring2_inner` and `ring2_outer` meters inward.
        - `midline`: the intersection of `boundary(buffer(-mid_offset))` with `ring2`.

    The result provides a median alignment curve (midline) representing an outer border within a block.
    Centroids located within ring1 or ring2 are snapped to the nearest point on this midline.

    **Assumptions:**
        - All geometries are expressed in meters (i.e., projected CRS such as EPSG:XXXX).
        - Works with both Polygon and MultiPolygon geometries.
        - Each block is assumed to be topologically closed and valid.

    **Parameters:**
        - `ring1_width` (float): width of the outer ring (meters).
        - `ring2_inner` (float): inner distance of the second ring (meters).
        - `ring2_outer` (float): outer distance of the second ring (meters).
        - `mid_offset` (float): offset distance used to generate the median line (meters).
    """

    ring1_width: float = 10.0
    ring2_inner: float = 10.0
    ring2_outer: float = 30.0
    mid_offset: float = 20.0

    @staticmethod
    def _lines_of(geom) -> List[LineString]:
        if geom is None or geom.is_empty:
            return []
        if isinstance(geom, LineString):
            return [geom]
        if isinstance(geom, MultiLineString):
            return list(geom.geoms)
        return []

    @staticmethod
    def _nearest_on_lines(
        lines: List[LineString], p: Point
    ) -> Tuple[int, Point, float]:
        best_i = -1
        best_q = None
        best_d = float("inf")
        best_m = 0.0
        for i, ln in enumerate(lines):
            m = ln.project(p)
            q = ln.interpolate(m)
            d = q.distance(p)
            if d < best_d:
                best_d = d
                best_q = q
                best_i = i
                best_m = m
        if best_q is None:
            return -1, p, 0.0
        return best_i, best_q, best_m

    @staticmethod
    def _safe_buffer(poly: Union[Polygon, MultiPolygon], dist: float):
        if poly is None or poly.is_empty:
            return poly
        return poly.buffer(0).buffer(dist)

    def _build_rings(
        self, block_geom: Union[Polygon, MultiPolygon]
    ) -> Tuple[
        Union[Polygon, MultiPolygon],
        Union[Polygon, MultiPolygon],
        Union[LineString, MultiLineString],
    ]:

        if block_geom is None or block_geom.is_empty:
            return Polygon(), Polygon(), LineString()

        block = block_geom.buffer(0)

        inner_r1 = self._safe_buffer(block, -max(self.ring1_width, 0.0))
        ring1 = (
            block
            if (inner_r1 is None or inner_r1.is_empty)
            else block.difference(inner_r1)
        )

        inner_in = self._safe_buffer(block, -max(self.ring2_inner, 0.0))
        inner_out = self._safe_buffer(block, -max(self.ring2_outer, 0.0))

        if inner_in is None or inner_in.is_empty:
            ring2 = Polygon()
        elif inner_out is None or inner_out.is_empty:
            ring2 = inner_in
        else:
            ring2 = inner_in.difference(inner_out)

        mid_poly = self._safe_buffer(block, -self.mid_offset)
        if mid_poly is None or mid_poly.is_empty:
            midline = LineString()
        else:
            midline = mid_poly.boundary.intersection(ring2)

        return ring1, ring2, midline

    def run(
        self,
        centroids: gpd.GeoDataFrame,
        blocks: gpd.GeoDataFrame,
        *,
        block_id_col: str = "block_id",
    ) -> Dict[str, object]:

        if centroids.empty:
            return {
                "centroids": centroids.copy(),
                "midlines": gpd.GeoSeries([], crs=blocks.crs),
                "midline": LineString(),
            }

        blocks_gdf = blocks.copy()
        cents = centroids.copy()
        if blocks_gdf.crs is None:
            raise ValueError("blocks GeoDataFrame must have a projected CRS (метры).")
        if cents.crs != blocks_gdf.crs:
            cents = cents.to_crs(blocks_gdf.crs)

        if block_id_col not in blocks_gdf.columns:
            blocks_gdf = blocks_gdf.copy()
            blocks_gdf[block_id_col] = blocks_gdf.index.astype("int64")

        if block_id_col not in cents.columns:
            joined = gpd.sjoin(
                cents,
                blocks_gdf[[block_id_col, "geometry"]],
                predicate="within",
                how="left",
            )
            if joined[block_id_col].isna().all():
                joined = gpd.sjoin(
                    cents,
                    blocks_gdf[[block_id_col, "geometry"]],
                    predicate="intersects",
                    how="left",
                )
            cents = joined.drop(
                columns=[c for c in joined.columns if c.startswith("index_")],
                errors="ignore",
            )

        block_geoms: Dict[object, Union[Polygon, MultiPolygon]] = dict(
            zip(
                blocks_gdf[block_id_col].values.tolist(),
                blocks_gdf.geometry.values.tolist(),
            )
        )
        rings_cache: Dict[object, Tuple[object, object, object]] = {}
        midline_items: List[Tuple[object, Union[LineString, MultiLineString]]] = []

        for bid, geom in block_geoms.items():
            r1, r2, mid = self._build_rings(geom)
            rings_cache[bid] = (r1, r2, mid)
            midline_items.append((bid, mid))

        midlines = gpd.GeoSeries(
            data=[m for _, m in midline_items],
            index=[bid for bid, _ in midline_items],
            crs=blocks_gdf.crs,
            name="midline",
        )

        new_points: List[Point] = []
        snap_flags: List[str] = []
        zones: List[str] = []
        bids: List[object] = []

        for i, row in cents.iterrows():
            p: Point = row.geometry
            bid = row.get(block_id_col, None)
            q = p
            status = "none"
            snap = "none"

            if bid in rings_cache:
                ring1, ring2, midline = rings_cache[bid]
                lines = self._lines_of(midline)

                if lines and (ring1 is not None) and (ring2 is not None):
                    in_r1 = ring1.contains(p) or ring1.touches(p)
                    in_r2 = ring2.contains(p) or ring2.touches(p)
                    if in_r1 or in_r2:
                        status = (
                            "first"
                            if (in_r1 and not in_r2)
                            else ("second" if in_r2 else "first")
                        )
                        li, q_on, _ = self._nearest_on_lines(lines, p)
                        if li >= 0:
                            q = q_on
                            snap = "to_ring2_midline"

            new_points.append(q)
            snap_flags.append(snap)
            zones.append(status)
            bids.append(bid)

        snapped = cents.copy()
        snapped = snapped.set_geometry(new_points)
        snapped["ring_snap"] = snap_flags
        snapped["ring_zone"] = zones
        snapped[block_id_col] = bids
        snapped = snapped.set_crs(blocks_gdf.crs, allow_override=True)
        not_empty = [g for g in midlines if (g is not None and not g.is_empty)]
        union_mid = unary_union(not_empty) if not_empty else LineString()

        return {
            "centroids": snapped,
            "midlines": midlines,
            "midline": union_mid,
        }


snapper = Snapper()
