from __future__ import annotations

import random
import asyncio
from typing import DefaultDict, Dict, List, Optional, Set

from loguru import logger
from tqdm.auto import tqdm
import geopandas as gpd
import numpy as np
import pandas as pd

from app.logic.postprocessing.generation_params import GenParams, ParamsProvider
from app.logic.postprocessing.grid_operations import GridOperations
from app.logic.postprocessing.shapes_library import ShapesLibrary
from app.logic.postprocessing.site_planner import SitePlanner
from app.logic.postprocessing.buildings_postprocessing import BuildingsPostProcessor


class BuildingGenerator:
    """
    Generates building polygons on a regular grid.

    Takes cell and zone layers, marks building cells, merges them into
    building polygons and applies post-processing. Use `fit_transform`
    to run the pipeline and get the result.
    """
    def __init__(self, grid_operations: GridOperations, 
                 buildings_postprocessor: BuildingsPostProcessor,
                 params_provider: ParamsProvider):
        self._params = params_provider
        self.grid_operations = grid_operations
        self.buildings_postprocessor = buildings_postprocessor

    @property
    def generation_parameters(self) -> GenParams:
        return self._params.current()

    async def fit_transform(
        self,
        cells_gdf: gpd.GeoDataFrame,
        zones_gdf: gpd.GeoDataFrame,
        zone_name_aliases: Optional[List[str]] = None
    ) -> gpd.GeoDataFrame | Dict[str, object]:
        if zone_name_aliases is None:
            zone_name_aliases = ["functional_zone_type_name", "zone_type", "zone_name"]
        cells = cells_gdf.copy().reset_index(drop=True)
        cells["inside_iso_closed"] = cells["inside_iso_closed"].fillna(False).astype(bool)
        cells["iso_level"] = pd.to_numeric(cells.get("iso_level"), errors="coerce")
        cells["service"] = None
        row_i, col_j, x0, y0, step_est = self.grid_operations.grid_indices(cells)
        cells["row_i"], cells["col_j"] = row_i, col_j
        zones = zones_gdf.to_crs(cells.crs).reset_index(drop=True)
        zid_col = self.generation_parameters.zone_id_col
        if zid_col not in zones.columns:
            zones[zid_col] = zones["id"] if "id" in zones.columns else np.arange(len(zones))
        zname_col = self.generation_parameters.zone_name_col
        if zname_col not in zones.columns:
            for alt in zone_name_aliases:
                if alt in zones.columns:
                    zones[zname_col] = zones[alt]
                    break
        if zname_col not in zones.columns:
            zones[zname_col] = "unknown"
        cells_zone = self.grid_operations.sjoin(
            cells[["geometry"]].reset_index().rename(columns={"index": "cell_idx"}),
            zones[[zid_col, zname_col, "geometry"]],
            predicate="within",
            how="left",
        ).drop_duplicates("cell_idx")

        cells = cells.merge(
            cells_zone[["cell_idx", zid_col, zname_col]],
            left_index=True,
            right_on="cell_idx",
            how="left",
        ).drop(columns=["cell_idx"])

        cells[zname_col] = cells[zname_col].astype(str).str.lower().str.strip()
        cells["is_residential_zone"] = cells[zname_col].eq("residential")

        neighbors_all, neighbors_side, neighbors_diag, empty_neighs, missing = self.grid_operations.compute_neighbors(cells)

        inside_mask = cells["inside_iso_closed"].values
        is_external = np.zeros(len(cells), dtype=bool)
        for i in range(len(cells)):
            if not inside_mask[i]:
                continue
            is_external[i] = (empty_neighs[i] >= self.generation_parameters.neigh_empty_thr) or (missing[i] > 0)
        cells["candidate_building"] = inside_mask & is_external

        cells["candidate_building"] = self.grid_operations.enforce_line_blocks(
            cells, line_key="row_i", order_key="col_j", mask_key="candidate_building", max_run=self.generation_parameters.max_run
        )
        cells["is_building"] = self.grid_operations.enforce_line_blocks(
            cells, line_key="col_j", order_key="row_i", mask_key="candidate_building", max_run=self.generation_parameters.max_run
        )

        bbox = np.array([
            cells.geometry.bounds.minx.values,
            cells.geometry.bounds.miny.values,
            cells.geometry.bounds.maxx.values,
            cells.geometry.bounds.maxy.values,
        ]).T
        w = bbox[:, 2] - bbox[:, 0]
        h = bbox[:, 3] - bbox[:, 1]
        is_small = (w < self.generation_parameters.cell_size_m - 1e-6) | (h < self.generation_parameters.cell_size_m - 1e-6)
        external_score = empty_neighs + missing

        is_b = cells["is_building"].to_numpy().astype(bool)
        promote_targets: List[int] = []
        for i in range(len(cells)):
            if not (is_b[i] and is_external[i] and is_small[i]):
                continue
            cand = [j for j in neighbors_side.get(i, []) if inside_mask[j]]
            if not cand:
                cand = [j for j in neighbors_all.get(i, []) if inside_mask[j]]
            cand = [j for j in cand if not is_b[j]]
            if not cand:
                continue
            j_best = min(cand, key=lambda j: (external_score[j], -empty_neighs[j]))
            promote_targets.append(j_best)

        if promote_targets:
            for j in tqdm(
                promote_targets,
                desc="Promote neighbor cells",
                unit="cell",
                leave=False,
            ):
                is_b[j] = True
            cells["is_building"] = is_b
            cells["is_building"] = self.grid_operations.enforce_line_blocks(
                cells, line_key="row_i", order_key="col_j", mask_key="is_building", max_run=self.generation_parameters.max_run
            )
            cells["is_building"] = self.grid_operations.enforce_line_blocks(
                cells, line_key="col_j", order_key="row_i", mask_key="is_building", max_run=self.generation_parameters.max_run
            )

        diag_components, diag_rects = self.buildings_postprocessor.mark_diag_only_and_buffers(
            cells, neighbors_side=neighbors_side, neighbors_diag=neighbors_diag
        )
        cells["is_diag_only"] = False
        if diag_components:
            idx = [i for comp in diag_components for i in comp]
            cells.loc[idx, "is_diag_only"] = True

        living_rects = self.buildings_postprocessor.living_cell_rects(cells, zid_col=zid_col)
        buildings_rects =  gpd.GeoDataFrame(pd.concat([living_rects, diag_rects], ignore_index=True), geometry='geometry', crs=cells.crs)
        buildings_merged = (
            buildings_rects[["geometry"]]    
                .dissolve()                   
                .explode(index_parts=False)   
                .reset_index(drop=True)
        )
        buildings_merged['is_living'] = True

        return {"buildings":buildings_merged, "cells":cells}