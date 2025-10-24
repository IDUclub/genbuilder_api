from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon


@dataclass
class GridGenerator:
    """
    Cell tagging based on isolines.

    **Workflow:**
      1) Convert isoline geometries into buffered “bands” (via `buffer`),
         then evaluate their nesting depth (`iso_level`).
      2) Construct “inner ring” polygons from closed isoline loops
         and compute their nesting (`fill_level`).
      3) Tag grid cells that intersect isoline bands or whose centroids fall
         inside inner polygons.
      4) Apply the “3-side rule” — a cell is considered inside if at least
         three of its edge-neighbors are inside.
      5) Propagate iso levels to promoted cells by taking the mode of their
         neighboring levels.

    **Input:**
        - `grid_gdf`: GeoDataFrame of grid cells (Polygon geometry)
        - `isolines_gdf`: GeoDataFrame of isolines (LineString, MultiLineString, or Polygon)

    **Output:**
        The same grid GeoDataFrame with additional columns:
        - `cell_id`: int — unique cell identifier
        - `iso_pids`: list[int] — IDs of intersected isoline bands
        - `inside_iso_raw`: bool — cell intersects a band or inner polygon
        - `inside_iso_closed`: bool — adjusted using the “3-side rule”
        - `iso_level_raw`: float — level derived from isoline bands or inner polygons
        - `iso_level`: float — final level after propagation
        - `fill_level`: float — nesting level of inner polygons, if any

    **Parameters:**
        - `iso_buffer_m` (float): buffer radius (in meters) to expand isoline lines into bands
        - `edge_share_frac` (float): minimum shared edge length fraction for detecting neighbors
        - `auto_reproject` (bool): automatically reproject isolines to the grid CRS
        - `fallback_epsg` (int): CRS assigned to the grid if undefined
        - `verbose` (bool): print log information during processing

    **Notes:**
        - The method is CRS-aware and supports both projected and geographic coordinate systems.
        - Suitable for generating isoline-based zoning masks or multi-level heatmaps over regular grids.
        - The “3-side rule” ensures morphological continuity along polygon boundaries.
    """

    iso_buffer_m: float = 1.0
    edge_share_frac: float = 0.2
    auto_reproject: bool = True
    fallback_epsg: int = 32636
    verbose: bool = True

    def fit_transform(
        self,
        grid_gdf: gpd.GeoDataFrame,
        isolines_gdf: gpd.GeoDataFrame,
        output_crs: Optional[str | int] = None,
    ) -> gpd.GeoDataFrame:

        grid = self._ensure_grid_crs(grid_gdf)
        iso = self._align_isolines_to_grid_crs(isolines_gdf, grid.crs)

        iso_polys = self._isolines_to_polys(iso, buffer_m=self.iso_buffer_m)
        iso_polys["iso_pid"] = pd.to_numeric(
            iso_polys["iso_pid"], errors="coerce"
        ).astype("Int64")
        iso_polys = gpd.GeoDataFrame(iso_polys, geometry="geometry", crs=iso.crs)
        iso_polys = self._attach_nesting_level(
            iso_polys, id_col="iso_pid", out_level_col="iso_level"
        )

        iso_fill = self._rings_to_fill_polys(iso)
        if len(iso_fill) > 0:
            iso_fill = self._attach_nesting_level(
                iso_fill, id_col="fill_id", out_level_col="fill_level"
            )

        grid = grid.reset_index(drop=True)
        if "cell_id" in grid.columns:
            grid["cell_id"] = pd.to_numeric(grid["cell_id"], errors="coerce")
            mask_na = grid["cell_id"].isna()
            if mask_na.any():
                grid.loc[mask_na, "cell_id"] = np.arange(len(grid))[mask_na]
            grid["cell_id"] = grid["cell_id"].astype(int)
        else:
            grid["cell_id"] = np.arange(len(grid), dtype=int)
        cells = grid[["cell_id", "geometry"]].copy()

        hit = gpd.sjoin(
            cells,
            iso_polys[["iso_pid", "iso_level", "geometry"]],
            predicate="intersects",
            how="left",
        )
        hit_nonnull = hit[hit["iso_pid"].notna()].copy()
        agg_inter = (
            hit_nonnull.groupby("cell_id", dropna=False)
            .agg(
                iso_pids=(
                    "iso_pid",
                    lambda s: sorted(
                        set(
                            int(x)
                            for x in pd.to_numeric(s, errors="coerce").dropna().tolist()
                        )
                    ),
                ),
                iso_level_raw=("iso_level", "max"),
            )
            .reset_index()
        )
        grid = grid.merge(agg_inter, on="cell_id", how="left")
        grid["iso_pids"] = grid["iso_pids"].apply(
            lambda v: v if isinstance(v, list) else []
        )
        grid["inside_iso_raw"] = grid["iso_pids"].apply(lambda v: len(v) > 0)
        grid["iso_level_raw"] = pd.to_numeric(
            grid["iso_level_raw"], errors="coerce"
        ).astype("Float64")

        if len(iso_fill) > 0:
            cells_pts = cells.copy()
            cells_pts["geometry"] = grid.geometry.representative_point().values
            hit_fill = gpd.sjoin(
                cells_pts,
                iso_fill[["fill_id", "fill_level", "geometry"]],
                predicate="within",
                how="left",
            )
            hit_fill_nonnull = hit_fill[hit_fill["fill_id"].notna()].copy()
            agg_fill = (
                hit_fill_nonnull.groupby("cell_id", dropna=False)
                .agg(fill_level=("fill_level", "max"))
                .reset_index()
            )
            grid = grid.merge(agg_fill, on="cell_id", how="left")
            inside_by_fill = grid["fill_level"].notna()
            grid.loc[inside_by_fill, "inside_iso_raw"] = True
            grid["iso_level_raw"] = np.fmax(
                grid["iso_level_raw"].astype(float).fillna(-1),
                pd.to_numeric(grid["fill_level"], errors="coerce").fillna(-1),
            )
            grid["iso_level_raw"].replace(-1, np.nan, inplace=True)
        else:
            grid["fill_level"] = np.nan

        neighbors = self._edge_neighbors(grid)
        inside_map = dict(zip(grid["cell_id"].values, grid["inside_iso_raw"].values))
        promote = []
        for rid, neighs in neighbors.items():
            if not inside_map.get(rid, False):
                if sum(inside_map.get(nb, False) for nb in neighs) >= 3:
                    promote.append(rid)

        grid["inside_iso_closed"] = grid.apply(
            lambda r: bool(r["inside_iso_raw"] or (r["cell_id"] in promote)), axis=1
        )

        level_map = dict(zip(grid["cell_id"].values, grid["iso_level_raw"].values))

        def _neighbor_level_mode(rid: int) -> float:
            vals = [
                level_map.get(nb)
                for nb in neighbors.get(rid, [])
                if inside_map.get(nb, False) and pd.notna(level_map.get(nb))
            ]
            if not vals:
                return np.nan
            return int(pd.Series(vals).value_counts().index[0])

        grid["iso_level"] = grid["iso_level_raw"]
        need_fill = grid["inside_iso_closed"] & (~grid["inside_iso_raw"])
        grid.loc[need_fill, "iso_level"] = grid.loc[need_fill, "cell_id"].apply(
            _neighbor_level_mode
        )

        cols_out = [
            "cell_id",
            "geometry",
            "iso_pids",
            "inside_iso_raw",
            "inside_iso_closed",
            "iso_level_raw",
            "iso_level",
            "fill_level",
        ]
        grid_out = gpd.GeoDataFrame(
            grid[cols_out].copy(), geometry="geometry", crs=grid.crs
        )

        if output_crs is not None:
            grid_out = grid_out.to_crs(output_crs)

        if self.verbose:
            print(
                f"OK | cells={len(grid_out)}, raw True={int(grid_out.inside_iso_raw.sum())}, "
                f"closed True={int(grid_out.inside_iso_closed.sum())}"
            )
        return grid_out

    def make_grid_for_blocks(
        self,
        blocks_gdf: gpd.GeoDataFrame,
        cell_size_m: float = 15.0,
        midlines: Optional[gpd.GeoDataFrame | gpd.GeoSeries | List] = None,
        block_id_col: Optional[str] = None,
        offset_m: float = 20.0,
        output_crs: Optional[str | int] = None,
    ) -> gpd.GeoDataFrame:

        from shapely.geometry import box as _box

        if blocks_gdf is None or len(blocks_gdf) == 0:
            return gpd.GeoDataFrame(
                {"block_id": [], "cell_id": []}, geometry=[], crs=self.fallback_epsg
            )

        blocks = blocks_gdf.copy()
        blocks = self._ensure_grid_crs(blocks)
        if midlines is not None and self.auto_reproject:
            try:
                if (
                    isinstance(midlines, (gpd.GeoDataFrame, gpd.GeoSeries))
                    and midlines.crs != blocks.crs
                ):
                    midlines = midlines.to_crs(blocks.crs)
            except Exception:
                pass

        if block_id_col and block_id_col in blocks.columns:
            block_ids = list(blocks[block_id_col].values)
        else:
            block_ids = list(range(len(blocks)))

        out_rows = []
        out_geoms = []

        for i, (bid, geom) in enumerate(zip(block_ids, blocks.geometry.values)):
            if geom is None or geom.is_empty:
                continue
            minx, miny, maxx, maxy = geom.bounds
            if maxx - minx <= 0 or maxy - miny <= 0:
                continue

            start_x = np.floor(minx / cell_size_m) * cell_size_m
            start_y = np.floor(miny / cell_size_m) * cell_size_m
            end_x = np.ceil(maxx / cell_size_m) * cell_size_m
            end_y = np.ceil(maxy / cell_size_m) * cell_size_m

            clip_area = None
            try:
                if offset_m > 0:
                    clip_area = geom.buffer(-float(offset_m))
            except Exception:
                clip_area = None

            if clip_area is None or clip_area.is_empty:
                try:
                    clip_area = geom.buffer(-1e-6)
                except Exception:
                    clip_area = None

            if clip_area is None or clip_area.is_empty:
                clip_area = geom
            cell_id = 0
            y = start_y
            while y < end_y - 1e-9:
                x = start_x
                while x < end_x - 1e-9:
                    cell = _box(x, y, x + cell_size_m, y + cell_size_m)
                    if not cell.intersects(clip_area):
                        x += cell_size_m
                        continue
                    try:
                        clipped = cell.intersection(clip_area)
                        if not clipped.is_empty:
                            try:
                                clipped = clipped.buffer(0)
                            except Exception:
                                pass
                    except Exception:
                        clipped = cell.intersection(geom)

                    if (
                        clipped is not None
                        and (not clipped.is_empty)
                        and clipped.area > 1e-6
                    ):
                        out_rows.append({"block_id": bid, "cell_id": cell_id})
                        out_geoms.append(clipped)
                        cell_id += 1

                    x += cell_size_m
                y += cell_size_m

        grid = gpd.GeoDataFrame(out_rows, geometry=out_geoms, crs=blocks.crs)

        if output_crs is not None:
            grid = grid.to_crs(output_crs)

        if self.verbose:
            n_blocks = len(set(grid["block_id"])) if len(grid) else 0
            print(
                f"GRID | blocks={n_blocks}, cells={len(grid)}, cell_size={cell_size_m} m, offset={offset_m} m"
            )

        return grid

    def _ensure_grid_crs(self, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if grid.crs is None:
            warnings.warn(f"CRS сетки отсутствует, назначаю EPSG:{self.fallback_epsg}")
            return grid.set_crs(epsg=self.fallback_epsg, allow_override=True)
        return grid

    def _align_isolines_to_grid_crs(
        self, iso: gpd.GeoDataFrame, target_crs
    ) -> gpd.GeoDataFrame:
        if not self.auto_reproject:
            return iso
        if iso.crs != target_crs:
            if self.verbose:
                print(f"Reproject isolines: {iso.crs} → {target_crs}")
            return iso.to_crs(target_crs)
        return iso

    def _isolines_to_polys(
        self, iso_gdf: gpd.GeoDataFrame, buffer_m: float
    ) -> gpd.GeoDataFrame:

        iso_gdf = iso_gdf.copy()
        line_mask = iso_gdf.geom_type.isin(["LineString", "MultiLineString"])
        poly_mask = iso_gdf.geom_type.isin(["Polygon", "MultiPolygon"])

        parts = []
        if line_mask.any():
            lines = iso_gdf.loc[line_mask].copy()
            lines["geometry"] = lines.geometry.buffer(buffer_m)
            parts.append(lines)
        if poly_mask.any():
            parts.append(iso_gdf.loc[poly_mask].copy())

        out = pd.concat(parts, ignore_index=True) if parts else iso_gdf.copy()
        if not parts:
            out["geometry"] = out.geometry.buffer(buffer_m)

        out = gpd.GeoDataFrame(out, geometry="geometry", crs=iso_gdf.crs).reset_index(
            drop=True
        )
        out["iso_pid"] = np.arange(len(out))
        return out[["iso_pid", "geometry"]]

    def _attach_nesting_level(
        self, polys: gpd.GeoDataFrame, id_col: str, out_level_col: str
    ) -> gpd.GeoDataFrame:

        pts = polys[[id_col, "geometry"]].copy()
        pts["geometry"] = pts.geometry.representative_point()
        pairs = gpd.sjoin(
            pts,
            polys[[id_col, "geometry"]],
            predicate="within",
            how="left",
            lsuffix="pt",
            rsuffix="poly",
        )
        cnt = pairs.groupby(f"{id_col}_pt", dropna=False).size() - 1
        levels = (
            cnt.reset_index()
            .rename(columns={f"{id_col}_pt": id_col, 0: out_level_col})
            .astype({id_col: "Int64"})
        )
        out = polys.merge(levels, on=id_col, how="left")
        out[out_level_col] = out[out_level_col].fillna(0).astype(int)
        return out

    def _rings_to_fill_polys(self, iso_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

        geoms: List[Polygon] = []
        for geom in iso_gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            gtype = geom.geom_type
            if gtype == "Polygon":
                geoms.append(Polygon(geom.exterior))
            elif gtype == "MultiPolygon":
                for p in geom.geoms:
                    geoms.append(Polygon(p.exterior))
            elif gtype == "LineString":
                if getattr(geom, "is_ring", False):
                    geoms.append(Polygon(geom.coords))
            elif gtype == "MultiLineString":
                for ls in geom.geoms:
                    if getattr(ls, "is_ring", False):
                        geoms.append(Polygon(ls.coords))
        if not geoms:
            return gpd.GeoDataFrame(
                {"fill_id": [], "geometry": []}, geometry="geometry", crs=iso_gdf.crs
            )

        geoms = [
            g.buffer(0) if isinstance(g, (Polygon, MultiPolygon)) else g for g in geoms
        ]
        fill = gpd.GeoDataFrame(
            {"geometry": geoms}, geometry="geometry", crs=iso_gdf.crs
        )
        fill = (
            fill[~fill.geometry.is_empty & fill.geometry.notna()]
            .copy()
            .reset_index(drop=True)
        )
        fill["fill_id"] = np.arange(len(fill))
        return fill[["fill_id", "geometry"]]

    def _edge_neighbors(self, grid: gpd.GeoDataFrame) -> Dict[int, List[int]]:

        pairs = gpd.sjoin(
            grid[["cell_id", "geometry"]],
            grid[["cell_id", "geometry"]],
            predicate="touches",
            how="left",
            lsuffix="a",
            rsuffix="b",
        )
        pairs = pairs[
            (pairs["cell_id_a"] != pairs["cell_id_b"]) & pairs["cell_id_b"].notna()
        ].copy()
        pairs["cell_id_b"] = pairs["cell_id_b"].astype(int)

        geom_list = list(grid.geometry.values)
        pos = {rid: i for i, rid in enumerate(grid["cell_id"].values)}
        edge_len_est = np.sqrt(np.maximum(grid.geometry.area.values, 1e-9))
        thr_len = self.edge_share_frac * edge_len_est

        def _is_edge_neighbor(a: int, b: int) -> bool:
            ia, ib = pos[a], pos[b]
            inter = geom_list[ia].boundary.intersection(geom_list[ib].boundary)
            length = getattr(inter, "length", 0.0)
            return length >= min(thr_len[ia], thr_len[ib])

        pairs["edge_ok"] = pairs.apply(
            lambda r: _is_edge_neighbor(int(r["cell_id_a"]), int(r["cell_id_b"])),
            axis=1,
        )
        pairs = pairs[pairs["edge_ok"]]

        neighbors: Dict[int, List[int]] = {rid: [] for rid in grid["cell_id"].values}
        for ra, rb in pairs[["cell_id_a", "cell_id_b"]].itertuples(index=False):
            neighbors[int(ra)].append(int(rb))
        return neighbors


grid_generator = GridGenerator()
