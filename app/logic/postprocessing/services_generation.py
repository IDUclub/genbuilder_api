from __future__ import annotations

import random
from typing import DefaultDict, Dict, List, Set

from loguru import logger
from tqdm.auto import tqdm
import geopandas as gpd
import pandas as pd

from app.logic.postprocessing.generation_params import GenParams, ParamsProvider
from app.logic.postprocessing.shapes_library import ShapesLibrary
from app.logic.postprocessing.site_planner import SitePlanner
from app.logic.postprocessing.service_types import ServiceType


class ServiceGenerator:
    """
    Generates service buildings on a regular grid.

    Uses living_area_per_block and service_normatives to compute required
    capacity per zone, then calls SitePlanner to place service sites and
    polygons. Use `fit_transform` to get a GeoDataFrame of service buildings.
    """
    def __init__(
        self,
        shapes_library: ShapesLibrary,
        planner: SitePlanner,
        params_provider: ParamsProvider,
    ):
        self._params = params_provider
        self.shapes_library = shapes_library
        self.planner = planner

    @property
    def generation_parameters(self) -> GenParams:
        return self._params.current()

    async def fit_transform(
        self,
        cells_gdf: gpd.GeoDataFrame,
        living_area_per_block: Dict,        
        service_normatives: pd.DataFrame,    
    ) -> gpd.GeoDataFrame | Dict[str, object]:
        idx_by_rc = {
            (int(r), int(c)): i
            for i, (r, c) in enumerate(zip(cells_gdf["row_i"], cells_gdf["col_j"]))
        }

        is_res = cells_gdf["is_residential_zone"].fillna(False).values
        not_house = ~cells_gdf["is_building"].fillna(False).values
        inside_true = cells_gdf["inside_iso_closed"].fillna(False).values
        inside_false = ~inside_true
        ok_iso = cells_gdf["iso_level"].fillna(0).values >= 0

        zid_col = self.generation_parameters.zone_id_col

        people_per_zone = {
            block: round(
                (living_area / self.generation_parameters.living_area_normative),
                0,
            )
            for block, living_area in living_area_per_block.items()
        }
        zone_groups: Dict[int, Dict] = {}
        zids_series = (
            cells_gdf[zid_col].astype("Int64")
            if zid_col in cells_gdf.columns
            else pd.Series([None] * len(cells_gdf))
        )

        for zid, sub_all in cells_gdf[pd.notna(zids_series)].groupby(zids_series):
            zid_int = int(zid)
            idxs = sub_all.index

            in_ids = sub_all.index[
                (is_res[idxs]) & (inside_true[idxs]) & ok_iso[idxs] & not_house[idxs]
            ].to_list()
            out_ids = sub_all.index[
                (is_res[idxs]) & (inside_false[idxs]) & not_house[idxs]
            ].to_list()

            if not in_ids and not out_ids:
                continue

            sub_res = sub_all.index[is_res[idxs]].to_list()
            if len(sub_res):
                r_center = float(cells_gdf.loc[sub_res, "row_i"].median())
                c_center = float(cells_gdf.loc[sub_res, "col_j"].median())
            else:
                r_center = float(sub_all["row_i"].median())
                c_center = float(sub_all["col_j"].median())

            zone_name = sub_all["zone"].mode().iat[0]

            zone_groups[zid_int] = {
                "inside_ids": in_ids,
                "outside_ids": out_ids,
                "r_center": r_center,
                "c_center": c_center,
                "zone_name": zone_name,
            }

        svc_capacity_by_zone: DefaultDict[int, DefaultDict[ServiceType, float]] = DefaultDict(
            lambda: DefaultDict(float)
        )

        rng = random.Random(self.generation_parameters.service_random_seed)
        shape_variants = self.shapes_library.build_shape_variants_from_library(rng=rng)

        reserved_site_cells: Set[int] = set()
        reserved_service_cells: Set[int] = set()
        service_sites_geom: List = []
        service_sites_attrs: List[Dict] = []
        service_polys_geom: List = []
        service_polys_attrs: List[Dict] = []
        placed_site_sets: List[List[int]] = []
        placed_sites_by_type: DefaultDict[ServiceType, List[List[int]]] = DefaultDict(list)

        for zid, meta in zone_groups.items():
            r_cen, c_cen = meta["r_center"], meta["c_center"]
            z_inside = meta["inside_ids"]
            z_outside = meta["outside_ids"]
            zone_name = meta["zone_name"]

            limits: Dict[ServiceType, float] = {}
            for service in self.generation_parameters.created_services:
                row = service_normatives.loc[
                    service_normatives["service_id"] == service.value,
                    "service_capacity",
                ]
                if row.empty:
                    logger.warning(
                        f"Service '{service}' has no normative capacity in service_normatives; skipping"
                    )
                    continue
                cap_per_1000 = float(row.iloc[0])
                people = float(people_per_zone.get(zid, 0.0))
                target_cap = cap_per_1000 * (people / 1000.0)
                limits[service] = target_cap

            total_targets = sum(limits.values())
            if total_targets == 0:
                continue

            placed_any = False
            pbar = tqdm(
                total=total_targets,
                desc=f"Zone {zone_name}-{zid} service placement",
                unit="cap",
                disable=not getattr(self.generation_parameters, "verbose", True),
                leave=False,
            )

            while True:
                progress = False

                for svc in self.generation_parameters.svc_order:
                    if svc not in limits:
                        continue
                    if svc_capacity_by_zone[zid][svc] >= limits[svc]:
                        continue

                    cand_inside = [i for i in z_inside if not_house[i] and ok_iso[i]]
                    cand_outside = [i for i in z_outside if not_house[i]]
                    candidate_ids = cand_inside + cand_outside
                    if not candidate_ids:
                        continue

                    ok, added_capacity = self.planner.place_service(
                        cells=cells_gdf,
                        zid=zid,
                        svc=svc,
                        candidate_ids=candidate_ids,
                        r_cen=r_cen,
                        c_cen=c_cen,
                        placed_site_sets=placed_site_sets,
                        placed_sites_by_type=placed_sites_by_type,
                        rng=rng,
                        shape_variants_by_svc=shape_variants,
                        idx_by_rc=idx_by_rc,
                        reserved_site_cells=reserved_site_cells,
                        reserved_service_cells=reserved_service_cells,
                        service_sites_geom=service_sites_geom,
                        service_sites_attrs=service_sites_attrs,
                        service_polys_geom=service_polys_geom,
                        service_polys_attrs=service_polys_attrs,
                        prefer_center=True,
                    )

                    if ok and added_capacity > 0:
                        svc_capacity_by_zone[zid][svc] += float(added_capacity)
                        pbar.update(float(added_capacity))
                        progress = True
                        placed_any = True

                if not progress:
                    break

            pbar.close()

        logger.debug(
            f"[check] service placements: polys={len(service_polys_attrs)}, "
            f"sites={len(service_sites_geom)}"
        )

        service_rects = gpd.GeoDataFrame(
            service_polys_attrs,
            geometry=service_polys_geom,
            crs=cells_gdf.crs,
        )
        service_rects["is_living"] = False

        return service_rects
