from __future__ import annotations

import random
import asyncio
from typing import DefaultDict, Dict, List, Optional, Set

from loguru import logger
from tqdm.auto import tqdm
import geopandas as gpd
import numpy as np
import pandas as pd

from app.logic.postprocessing.generation_params import GenParams
from app.logic.postprocessing.grid_operations import GridOperations
from app.logic.postprocessing.shapes_library import ShapesLibrary
from app.logic.postprocessing.site_panner import SitePlanner
from app.logic.postprocessing.buildings_postprocessing import BuildingsPostProcessor


class BuildingGenerator:
    """
    Оркестратор генерации зданий и площадок сервисов на регулярной сетке.

    Компоненты:
      • GridOps — базовые операции над сеткой/геометрией/топологией
      • ServiceShapes — библиотека форм сервисов + утилиты вариантов
      • SitePlanner — размещение площадок и «ядер» сервисов
      • PostProcessor — диагональные буферы, слияние, метрики и финализация

    Публичный API:
      • fit_transform(cells_gdf, zones_gdf, zone_name_aliases=None) -> gpd.GeoDataFrame
        Возвращает итоговый GeoDataFrame зданий (жилые + сервисы) с атрибутами.
        При необходимости легко расширяется до возврата словаря промежуточных результатов.
    """
    def __init__(self, grid_operations: GridOperations, shapes_library: ShapesLibrary, 
                 buildings_postprocessor: BuildingsPostProcessor, planner: SitePlanner, 
                 generation_parameters: GenParams):
        self.generation_parameters = generation_parameters
        self.grid_operations = grid_operations
        self.shapes_library = shapes_library
        self.planner = planner
        self.buildings_postprocessor = buildings_postprocessor

    async def fit_transform(
        self,
        cells_gdf: gpd.GeoDataFrame,
        zones_gdf: gpd.GeoDataFrame,
        zone_name_aliases: Optional[List[str]] = None,
        *,
        return_all: bool = False,
    ) -> gpd.GeoDataFrame | Dict[str, object]:
        if zone_name_aliases is None:
            zone_name_aliases = ["functional_zone_type_name", "zone_type", "zone_name"]
        cells = cells_gdf.copy().reset_index(drop=True)
        if "inside_iso_closed" not in cells.columns:
            raise ValueError("Во входном слое клеток отсутствует колонка 'inside_iso_closed'.")
        cells["inside_iso_closed"] = cells["inside_iso_closed"].fillna(False).astype(bool)
        cells["iso_level"] = pd.to_numeric(cells.get("iso_level"), errors="coerce")
        cells["service"] = None
        row_i, col_j, x0, y0, step_est = await asyncio.to_thread(self.grid_operations.grid_indices, cells)
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

        assigned = cells[zid_col].notna().sum()
        neighbors_all, neighbors_side, neighbors_diag, empty_neighs, missing = await asyncio.to_thread(self.grid_operations.compute_neighbors, cells)
        avg_side = np.mean([len(v) for v in neighbors_side.values()]) if neighbors_side else 0.0
        avg_diag = np.mean([len(v) for v in neighbors_diag.values()]) if neighbors_diag else 0.0

        inside_mask = cells["inside_iso_closed"].values
        is_external = np.zeros(len(cells), dtype=bool)
        for i in range(len(cells)):
            if not inside_mask[i]:
                continue
            is_external[i] = (empty_neighs[i] >= self.generation_parameters.neigh_empty_thr) or (missing[i] > 0)
        cells["candidate_building"] = inside_mask & is_external

        cells["candidate_building"] = await asyncio.to_thread(self.grid_operations.enforce_line_blocks,
            cells, line_key="row_i", order_key="col_j", mask_key="candidate_building", max_run=self.generation_parameters.max_run
        )
        cells["is_building"] = await asyncio.to_thread(self.grid_operations.enforce_line_blocks,
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
            cells["is_building"] = await asyncio.to_thread(self.grid_operations.enforce_line_blocks,
                cells, line_key="row_i", order_key="col_j", mask_key="is_building", max_run=self.generation_parameters.max_run
            )
            cells["is_building"] = await asyncio.to_thread(self.grid_operations.enforce_line_blocks,
                cells, line_key="col_j", order_key="row_i", mask_key="is_building", max_run=self.generation_parameters.max_run
            )

        idx_by_rc = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(cells["row_i"], cells["col_j"]))}
        is_res = cells["is_residential_zone"].fillna(False).values
        not_house = ~cells["is_building"].fillna(False).values
        inside_true = cells["inside_iso_closed"].fillna(False).values
        inside_false = ~inside_true
        ok_iso = cells["iso_level"].fillna(0).values >= 0

        zone_groups: Dict[int, Dict] = {}
        zids_series = (cells[zid_col].astype("Int64") if zid_col in cells.columns else pd.Series([None] * len(cells)))
        for zid, sub_all in cells[pd.notna(zids_series)].groupby(zids_series):
            zid_int = int(zid)
            idxs = sub_all.index
            in_ids = sub_all.index[(is_res[idxs]) & (inside_true[idxs]) & ok_iso[idxs] & not_house[idxs]].to_list()
            out_ids = sub_all.index[(is_res[idxs]) & (inside_false[idxs]) & not_house[idxs]].to_list()
            if not in_ids and not out_ids:
                continue
            sub_res = sub_all.index[is_res[idxs]].to_list()
            r_center = float(cells.loc[sub_res, "row_i"].median()) if len(sub_res) else float(sub_all["row_i"].median())
            c_center = float(cells.loc[sub_res, "col_j"].median()) if len(sub_res) else float(sub_all["col_j"].median())
            Lmax = int(pd.to_numeric(cells.loc[in_ids, "iso_level"], errors="coerce").fillna(0).max()) if in_ids else 0
            zone_groups[zid_int] = {
                "inside_ids": in_ids,
                "outside_ids": out_ids,
                "r_center": r_center,
                "c_center": c_center,
                "Lmax": Lmax,
            }
        svc_count_by_zone: DefaultDict[int, DefaultDict[str, int]] = DefaultDict(lambda: DefaultDict(int))
        rng = random.Random(self.generation_parameters.service_random_seed)
        shape_variants = await asyncio.to_thread(self.shapes_library.build_shape_variants_from_library, rng=rng)

        reserved_site_cells: Set[int] = set()
        reserved_service_cells: Set[int] = set()
        service_sites_geom: List = []
        service_sites_attrs: List[Dict] = []
        service_polys_geom: List = []
        service_polys_attrs: List[Dict] = []
        placed_site_sets: List[List[int]] = []
        placed_sites_by_type: DefaultDict[str, List[List[int]]] = DefaultDict(list)

        for zid, meta in zone_groups.items():
            r_cen, c_cen = meta["r_center"], meta["c_center"]
            Lmax = meta["Lmax"]
            z_inside = meta["inside_ids"]
            z_outside = meta["outside_ids"]

            limits = {svc: self.generation_parameters.max_services_per_zone.get(svc, 0)
                      for svc in self.generation_parameters.svc_order}
            total_targets = sum(limits.values())
            if total_targets == 0:
                continue

            placed_any = False
            pbar = tqdm(
                total=total_targets,
                desc=f"Zone {zid} service placement",
                unit="svc",
                disable=not getattr(self.generation_parameters, "verbose", True),
                leave=False,
            )
            while True:
                progress = False
                for svc in self.generation_parameters.svc_order:
                    if svc_count_by_zone[zid][svc] >= limits[svc]:
                        continue
                    placed_here = False

                    for L in range(Lmax, -1, -1):
                        allowed_ids = [
                            i for i in z_inside
                            if (i not in reserved_site_cells)
                            and (not bool(cells.at[i, "is_building"]))
                            and pd.notna(cells.at[i, "iso_level"]) and (int(cells.at[i, "iso_level"]) >= L)
                        ]
                        if not allowed_ids:
                            continue
                        ok = await asyncio.to_thread(self.planner.try_place_site_and_service_in_zone_level,
                            cells, zid, svc, allowed_ids, r_cen, c_cen,
                            placed_site_sets, placed_sites_by_type, rng,
                            shape_variants, idx_by_rc,
                            reserved_site_cells, reserved_service_cells,
                            neighbors_all,
                            service_sites_geom, service_sites_attrs,
                            service_polys_geom, service_polys_attrs,
                        )
                        if ok:
                            svc_count_by_zone[zid][svc] += 1
                            pbar.update(1)
                            placed_here = True
                            placed_any = True
                            break

                    if (not placed_here) and z_outside:
                        ok = await asyncio.to_thread(self.planner.try_place_site_and_service_fallback_outside,
                            cells, zid, svc, z_outside, r_cen, c_cen,
                            placed_site_sets, placed_sites_by_type, rng,
                            shape_variants, idx_by_rc, reserved_site_cells, reserved_service_cells,
                            neighbors_all,
                            service_sites_geom, service_sites_attrs,
                            service_polys_geom, service_polys_attrs,
                        )
                        if ok:
                            svc_count_by_zone[zid][svc] += 1
                            pbar.update(1)
                            placed_here = True
                            placed_any = True

                    if placed_here:
                        progress = True

                if not progress:
                    break
            pbar.close()

        diag_components, diag_rects = await asyncio.to_thread(self.buildings_postprocessor.mark_diag_only_and_buffers,
            cells, neighbors_side=neighbors_side, neighbors_diag=neighbors_diag
        )
        cells["is_diag_only"] = False
        if diag_components:
            idx = [i for comp in diag_components for i in comp]
            cells.loc[idx, "is_diag_only"] = True

        living_rects = await asyncio.to_thread(self.buildings_postprocessor.living_cell_rects, cells, zid_col=zid_col)
        buildings_rects = await asyncio.to_thread(self.buildings_postprocessor.concat_rects, 
            living_rects, diag_rects, service_polys_attrs, service_polys_geom, crs=cells.crs
        )

        buildings_merged = await asyncio.to_thread(self.buildings_postprocessor.merge_by_service, buildings_rects, zid_col=zid_col)
        service_sites_gdf = gpd.GeoDataFrame(service_sites_attrs, geometry=service_sites_geom, crs=cells.crs).reset_index(drop=True)
        buildings = await asyncio.to_thread(self.buildings_postprocessor.finalize_buildings, cells, buildings_merged, service_sites_gdf)
        if self.generation_parameters.verbose:
            svc_rects = buildings_rects[buildings_rects.get("service").isin(self.generation_parameters.svc_order)]
            logger.debug(
                f"OK | cells={len(cells)}, living={int(cells['is_building'].sum())}, diag-only={int(cells.get('is_diag_only', False).sum() if 'is_diag_only' in cells.columns else 0)}\n"
                f"Rects={len(buildings_rects)} (service rects: {len(svc_rects)}) | Merged={len(buildings_merged)} | Sites={len(service_sites_gdf)}"
            )

        if return_all:
            return {
                "cells": cells,
                "buildings_rects": buildings_rects,
                "buildings_merged": buildings_merged,
                "service_sites": service_sites_gdf,
                "buildings": buildings,
            }
        return buildings
