from __future__ import annotations

import geopandas as gpd


class PlotMerger:
    """
    Merges undersized plots by unioning each small polygon with its best
    shared-boundary neighbour, with an optional iterative pass until plots
    reach a target area based on plot_front * plot_depth.
    """
    @staticmethod
    def merge_small_plots(
        plots: gpd.GeoDataFrame, area_factor: float
    ) -> gpd.GeoDataFrame:

        if plots.empty:
            return plots

        gdf = plots.copy()

        gdf = gdf[gdf.geometry.notna()]
        gdf = gdf[~gdf.geometry.is_empty]
        if gdf.empty:
            return gpd.GeoDataFrame(columns=plots.columns, crs=plots.crs)

        gdf["_target_area"] = gdf["plot_front"] * gdf["plot_depth"]
        gdf["_area"] = gdf.geometry.area
        gdf["_is_small"] = gdf["_area"] < (area_factor * gdf["_target_area"])

        try:
            sindex = gdf.sindex
        except Exception:
            result = gdf.drop(
                columns=["_target_area", "_area", "_is_small"], errors="ignore"
            )
            result.reset_index(drop=True, inplace=True)
            return result

        processed: set[int] = set()
        new_rows = []

        for idx, row in gdf.sort_values("_area").iterrows():
            if idx in processed:
                continue

            geom = row.geometry

            if geom is None or geom.is_empty:
                new_rows.append(row)
                processed.add(idx)
                continue

            if not bool(row["_is_small"]):
                new_rows.append(row)
                processed.add(idx)
                continue

            possible_idx = [
                i
                for i in list(sindex.intersection(geom.bounds))
                if i != idx and i not in processed
            ]

            best_neighbor_idx = None
            best_shared_len = 0.0

            for j in possible_idx:
                other_geom = gdf.at[j, "geometry"]

                if other_geom is None or other_geom.is_empty:
                    continue

                try:
                    shared_geom = geom.boundary.intersection(other_geom.boundary)
                except Exception:
                    continue

                if shared_geom is None:
                    continue

                try:
                    shared_len = shared_geom.length
                except AttributeError:
                    continue

                if shared_len <= 0:
                    continue

                if shared_len > best_shared_len:
                    best_shared_len = shared_len
                    best_neighbor_idx = j

            if best_neighbor_idx is None or best_shared_len == 0.0:
                new_rows.append(row)
                processed.add(idx)
                continue

            neighbor = gdf.loc[best_neighbor_idx]
            try:
                merged_geom = geom.union(neighbor.geometry)
            except Exception:
                new_rows.append(row)
                processed.add(idx)
                continue

            merged_row = neighbor.copy()
            merged_row.geometry = merged_geom
            merged_row["_area"] = merged_geom.area
            merged_row["_is_small"] = False

            new_rows.append(merged_row)
            processed.add(idx)
            processed.add(best_neighbor_idx)

        result = gpd.GeoDataFrame(new_rows, crs=gdf.crs)
        result = result.drop(
            columns=["_target_area", "_area", "_is_small"], errors="ignore"
        )
        result.reset_index(drop=True, inplace=True)
        return result

    def merge_small_plots_iterative(
        self,
        plots: gpd.GeoDataFrame,
        area_factor: float,
        max_iters: int = 10,
    ) -> gpd.GeoDataFrame:
        gdf = plots.copy()
        for _ in range(max_iters):
            gdf_new = self.merge_small_plots(gdf, area_factor=area_factor)

            target_area = gdf_new["plot_front"] * gdf_new["plot_depth"]
            small_mask = gdf_new.geometry.area < (area_factor * target_area)

            gdf = gdf_new
            if not small_mask.any():
                break

        return gdf