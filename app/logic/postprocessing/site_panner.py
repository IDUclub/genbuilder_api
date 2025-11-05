from __future__ import annotations

import random
from typing import Dict, List, Tuple, Set

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union

from app.logic.postprocessing.generation_params import GenParams, ParamsProvider
from app.logic.postprocessing.shapes_library import ShapesLibrary
from app.logic.postprocessing.grid_operations import GridOperations


class SitePlanner:
    """
    Планировщик площадок и размещения сервисных зданий внутри площадок.

    Ответственности:
      • Выбор позиций площадок относительно центра зоны/из краёв
      • Проверки разнесения по Чебышёву до жилых зданий и других площадок
      • Подбор прямоугольной площадки нужного размера и размещение «ядра» сервиса внутри
      • Создание геометрий площадок и сервисных полигонов, накопление атрибутов

    Зависимости:
      • GenParams — численные пороги и флаги рандомизации
      • GridOps — make_valid, вспомогательная топология (опционально)
      • ServiceShapes — библиотека форм и нормативы площадок
    """
    def __init__(self, grid_operations: GridOperations, shapes_library: ShapesLibrary, params_provider: ParamsProvider):
        self._params = params_provider
        self.grid_operations = grid_operations
        self.shapes_library = shapes_library

    @property
    def generation_parameters(self) -> GenParams:
        return self._params.current()

    @staticmethod
    def _min_cheb_between_sets(
        cells: gpd.GeoDataFrame, A: List[int], B: List[int]
    ) -> int:
        if not A or not B:
            return 10 ** 9
        ar = cells.loc[A, "row_i"].to_numpy()
        ac = cells.loc[A, "col_j"].to_numpy()
        br = cells.loc[B, "row_i"].to_numpy()
        bc = cells.loc[B, "col_j"].to_numpy()
        best = 10 ** 9
        for i in range(len(ar)):
            dr = np.abs(br - ar[i])
            dc = np.abs(bc - ac[i])
            d = int(np.min(np.maximum(dr, dc)))
            if d < best:
                best = d
            if best == 0:
                break
        return best

    @staticmethod
    def _house_indices(cells: gpd.GeoDataFrame) -> np.ndarray:
        return np.where(cells.get("is_building", False))[0]

    def _cheb_gap_ok_to_houses(
        self, cells: gpd.GeoDataFrame, candidate_idxs: List[int]
    ) -> bool:
        H = np.where(cells["is_building"].fillna(False).to_numpy())[0]
        if len(H) == 0 or len(candidate_idxs) == 0:
            return True
        d = self._min_cheb_between_sets(cells, candidate_idxs, list(H))
        return d >= self.generation_parameters.gap_to_houses_cheb

    def _cheb_gap_ok_to_sites(
        self,
        cells: gpd.GeoDataFrame,
        candidate_idxs: List[int],
        placed_site_sets: List[List[int]],
        placed_sites_by_type: Dict[str, List[List[int]]],
        svc: str,
    ) -> bool:
        for S in placed_site_sets:
            if (
                self._min_cheb_between_sets(cells, candidate_idxs, S)
                < self.generation_parameters.gap_between_sites_cheb
            ):
                return False
        for S in placed_sites_by_type.get(svc, []):
            if (
                self._min_cheb_between_sets(cells, candidate_idxs, S)
                < self.generation_parameters.same_type_site_gap_cheb
            ):
                return False
        return True

    @staticmethod
    def _positions_from_center_or_edges(
        svc: str,
        rmin: int,
        rmax: int,
        cmin: int,
        cmax: int,
        r_center: float,
        c_center: float,
        invert: bool = False,
    ) -> List[Tuple[int, int]]:
        pos = [(r0, c0) for r0 in range(rmin, rmax + 1) for c0 in range(cmin, cmax + 1)]

        def d2(rc: Tuple[int, int]):
            return (rc[0] - r_center) ** 2 + (rc[1] - c_center) ** 2

        pos.sort(key=d2, reverse=invert and (svc == "kindergarten"))
        return pos

    def try_place_service_inside_site(
        self,
        cells: gpd.GeoDataFrame,
        svc: str,
        zid: int,
        site_idxs: List[int],
        site_id: str,
        shape_variants_by_svc: Dict[str, List[Tuple[str, List[List[Tuple[int, int]]]]]],
        reserved_service_cells: Set[int],
        rng: random.Random,
        idx_by_rc: Dict[Tuple[int, int], int],
        *,
        inner_margin_cells: int | None = None,
    ) -> Tuple[bool, List[int], str]:
        if inner_margin_cells is None:
            inner_margin_cells = int(self.generation_parameters.inner_margin_cells)
        site_set = set(site_idxs)
        rvals = cells.loc[site_idxs, "row_i"].to_numpy()
        cvals = cells.loc[site_idxs, "col_j"].to_numpy()
        rmin, rmax = int(rvals.min()), int(rvals.max())
        cmin, cmax = int(cvals.min()), int(cvals.max())
        rmin_core = rmin + inner_margin_cells
        rmax_core = rmax - inner_margin_cells
        cmin_core = cmin + inner_margin_cells
        cmax_core = cmax - inner_margin_cells
        if (rmin_core > rmax_core) or (cmin_core > cmax_core):
            return False, [], ""
        core_h = rmax_core - rmin_core + 1
        core_w = cmax_core - cmin_core + 1
        cen_r = 0.5 * (rmin_core + rmax_core)
        cen_c = 0.5 * (cmin_core + cmax_core)

        for (pat_name, variants) in shape_variants_by_svc.get(svc, []):
            vars_iter = list(variants)
            if self.generation_parameters.randomize_service_forms:
                rng.shuffle(vars_iter)
            vars_iter = self.shapes_library.sort_variants_by_core_fit(vars_iter, core_h, core_w)
            for var in vars_iter:
                vr = [dr for (dr, dc) in var]
                vc = [dc for (dr, dc) in var]
                h, w = (max(vr) - min(vr) + 1), (max(vc) - min(vc) + 1)
                if h > core_h or w > core_w:
                    continue
                positions = [
                    (r0, c0)
                    for r0 in range(rmin_core, rmax_core - h + 2)
                    for c0 in range(cmin_core, cmax_core - w + 2)
                ]
                positions.sort(key=lambda rc: (rc[0] - cen_r) ** 2 + (rc[1] - cen_c) ** 2)
                for (r0, c0) in positions:
                    idxs: List[int] = []
                    ok = True
                    for (dr, dc) in var:
                        rr, cc = r0 + dr, c0 + dc
                        idx = idx_by_rc.get((rr, cc))
                        if (
                            (idx is None)
                            or (idx in reserved_service_cells)
                            or (idx not in site_set)
                        ):
                            ok = False
                            break
                        idxs.append(idx)
                    if not ok:
                        continue
                    return True, idxs, pat_name
        return False, [], ""

    def try_place_site_and_service_in_zone_level(
        self,
        cells: gpd.GeoDataFrame,
        zid: int,
        svc: str,
        allowed_ids: List[int],
        r_cen: float,
        c_cen: float,
        placed_site_sets: List[List[int]],
        placed_sites_by_type: Dict[str, List[List[int]]],
        rng: random.Random,
        shape_variants_by_svc: Dict[str, List[Tuple[str, List[List[Tuple[int, int]]]]]],
        idx_by_rc: Dict[Tuple[int, int], int],
        reserved_site_cells: Set[int],
        reserved_service_cells: Set[int],
        neighbors_all: Dict[int, List[int]],
        service_sites_geom: List,
        service_sites_attrs: List[Dict],
        service_polys_geom: List,
        service_polys_attrs: List[Dict],
    ) -> bool:
        if not allowed_ids:
            return False
        allowed_set = set(allowed_ids)
        coord_to_idx = {
            (int(cells.at[i, "row_i"]), int(cells.at[i, "col_j"])): i
            for i in allowed_ids
        }
        sub = cells.loc[allowed_ids, ["row_i", "col_j"]]
        rmin, rmax = int(sub["row_i"].min()), int(sub["row_i"].max())
        cmin, cmax = int(sub["col_j"].min()), int(sub["col_j"].max())
        positions = self._positions_from_center_or_edges(
            svc, rmin, rmax, cmin, cmax, r_cen, c_cen, invert=False
        )

        service_variants = list(shape_variants_by_svc.get(svc, []))
        if self.generation_parameters.randomize_service_forms:
            rng.shuffle(service_variants)

        for (pat_name, _service_vars) in service_variants:
            site_area_m2, capacity = self.shapes_library.service_site_spec(svc, pat_name)
            territory_variants = self.shapes_library.territory_shape_variants(site_area_m2)
            if self.generation_parameters.randomize_service_forms:
                rng.shuffle(territory_variants)

            for (site_form_name, site_offsets) in territory_variants:
                vrr = [dr for (dr, dc) in site_offsets]
                vcc = [dc for (dr, dc) in site_offsets]
                Hs, Ws = (max(vrr) - min(vrr) + 1), (max(vcc) - min(vcc) + 1)
                for (r0, c0) in positions:
                    if r0 + Hs - 1 > rmax or c0 + Ws - 1 > cmax:
                        continue
                    site_idxs: List[int] = []
                    ok = True
                    for (dr, dc) in site_offsets:
                        rr, cc = r0 + dr, c0 + dc
                        idx = coord_to_idx.get((rr, cc))
                        if (
                            (idx is None)
                            or (idx in reserved_site_cells)
                            or (idx in reserved_service_cells)
                        ):
                            ok = False
                            break
                        if idx not in allowed_set:
                            ok = False
                            break
                        if cells.at[idx, "is_building"]:
                            ok = False
                            break
                        site_idxs.append(idx)
                    if not ok:
                        continue
                    if not self._cheb_gap_ok_to_houses(cells, site_idxs):
                        continue
                    if not self._cheb_gap_ok_to_sites(
                        cells, site_idxs, placed_site_sets, placed_sites_by_type, svc
                    ):
                        continue
                    (
                        ok_svc,
                        svc_cell_idxs,
                        chosen_pat,
                    ) = self.try_place_service_inside_site(
                        cells,
                        svc,
                        zid,
                        site_idxs,
                        site_id="__tmp__",
                        shape_variants_by_svc=shape_variants_by_svc,
                        reserved_service_cells=reserved_service_cells,
                        rng=rng,
                        idx_by_rc=idx_by_rc,
                        inner_margin_cells=int(self.generation_parameters.inner_margin_cells),
                    )
                    if not ok_svc:
                        continue

                    site_id = f"SITE_{svc.upper()}_{str(len(service_sites_attrs) + 1).zfill(4)}"
                    service_id = f"{svc.upper()}_{str(len(service_polys_attrs) + 1).zfill(4)}"
                    for idx in site_idxs:
                        reserved_site_cells.add(idx)
                        cells.loc[idx, "is_service_site"] = True
                        cells.loc[idx, "site_id"] = site_id
                        cells.loc[idx, "service_site_type"] = svc
                    for idx in svc_cell_idxs:
                        reserved_service_cells.add(idx)
                        cells.loc[idx, "service"] = svc

                    site_poly = self.grid_operations.make_valid(
                        unary_union([cells.geometry[i] for i in site_idxs])
                    )
                    svc_poly = self.grid_operations.make_valid(
                        unary_union([cells.geometry[i] for i in svc_cell_idxs])
                    )

                    service_sites_geom.append(site_poly)
                    service_sites_attrs.append(
                        {
                            "site_id": site_id,
                            "service": svc,
                            "zone_id": int(zid),
                            "site_form": site_form_name,
                            "pattern_for_norms": pat_name,
                            "site_cells": int(len(site_idxs)),
                            "site_area_target_m2": float(site_area_m2),
                            "site_area_actual_m2": float(getattr(site_poly, "area", 0.0)),
                            "capacity": int(capacity),
                        }
                    )
                    service_polys_geom.append(svc_poly)
                    service_polys_attrs.append(
                        {
                            "building_id": service_id,
                            "site_id": site_id,
                            "service": svc,
                            "pattern": chosen_pat,
                            "zone_id": int(zid),
                            "n_cells": int(len(svc_cell_idxs)),
                            "width_m": float(self.generation_parameters.cell_size_m),
                        }
                    )

                    placed_site_sets.append(site_idxs)
                    placed_sites_by_type.setdefault(svc, []).append(site_idxs)
                    return True
        return False

    def try_place_site_and_service_fallback_outside(
        self,
        cells: gpd.GeoDataFrame,
        zid: int,
        svc: str,
        outside_ids: List[int],
        r_cen: float,
        c_cen: float,
        placed_site_sets: List[List[int]],
        placed_sites_by_type: Dict[str, List[List[int]]],
        rng: random.Random,
        shape_variants_by_svc: Dict[str, List[Tuple[str, List[List[Tuple[int, int]]]]]],
        idx_by_rc: Dict[Tuple[int, int], int],
        reserved_site_cells: Set[int],
        reserved_service_cells: Set[int],
        neighbors_all: Dict[int, List[int]],
        service_sites_geom: List,
        service_sites_attrs: List[Dict],
        service_polys_geom: List,
        service_polys_attrs: List[Dict],
    ) -> bool:
        if not outside_ids:
            return False
        allowed_ids = [
            i
            for i in outside_ids
            if (i not in reserved_site_cells) and (not cells.at[i, "is_building"])
        ]
        if not allowed_ids:
            return False
        allowed_set = set(allowed_ids)
        coord_to_idx = {
            (int(cells.at[i, "row_i"]), int(cells.at[i, "col_j"])): i
            for i in allowed_ids
        }
        sub = cells.loc[allowed_ids, ["row_i", "col_j"]]
        rmin, rmax = int(sub["row_i"].min()), int(sub["row_i"].max())
        cmin, cmax = int(sub["col_j"].min()), int(sub["col_j"].max())
        positions = self._positions_from_center_or_edges(
            svc, rmin, rmax, cmin, cmax, r_cen, c_cen, invert=False
        )

        service_variants = list(shape_variants_by_svc.get(svc, []))
        if self.generation_parameters.randomize_service_forms:
            rng.shuffle(service_variants)

        for (pat_name, _service_vars) in service_variants:
            site_area_m2, capacity = self.shapes_library.service_site_spec(svc, pat_name)
            min_cells = self.shapes_library.min_site_cells_for_service_with_margin(
                svc, shape_variants_by_svc, inner_margin_cells=int(self.generation_parameters.inner_margin_cells)
            )
            ncells = max(self.shapes_library.site_cells_required(site_area_m2), min_cells)
            territory_variants = self.shapes_library.rect_variants_for_cells(ncells, max_variants=12)
            if self.generation_parameters.randomize_service_forms:
                rng.shuffle(territory_variants)

            for (site_form_name, site_offsets) in territory_variants:
                vrr = [dr for (dr, dc) in site_offsets]
                vcc = [dc for (dr, dc) in site_offsets]
                Hs, Ws = (max(vrr) - min(vrr) + 1), (max(vcc) - min(vcc) + 1)
                for (r0, c0) in positions:
                    if r0 + Hs - 1 > rmax or c0 + Ws - 1 > cmax:
                        continue
                    site_idxs: List[int] = []
                    ok = True
                    for (dr, dc) in site_offsets:
                        rr, cc = r0 + dr, c0 + dc
                        idx = coord_to_idx.get((rr, cc))
                        if (
                            (idx is None)
                            or (idx in reserved_site_cells)
                            or (idx in reserved_service_cells)
                        ):
                            ok = False
                            break
                        if idx not in allowed_set:
                            ok = False
                            break
                        if cells.at[idx, "is_building"]:
                            ok = False
                            break
                        site_idxs.append(idx)
                    if not ok:
                        continue
                    if not self._cheb_gap_ok_to_houses(cells, site_idxs):
                        continue
                    if not self._cheb_gap_ok_to_sites(
                        cells, site_idxs, placed_site_sets, placed_sites_by_type, svc
                    ):
                        continue
                    (
                        ok_svc,
                        svc_cell_idxs,
                        chosen_pat,
                    ) = self.try_place_service_inside_site(
                        cells,
                        svc,
                        zid,
                        site_idxs,
                        site_id="__tmp__",
                        shape_variants_by_svc=shape_variants_by_svc,
                        reserved_service_cells=reserved_service_cells,
                        rng=rng,
                        idx_by_rc=idx_by_rc,
                        inner_margin_cells=int(self.generation_parameters.inner_margin_cells),
                    )
                    if not ok_svc:
                        continue

                    site_id = f"SITE_{svc.upper()}_{str(len(service_sites_attrs) + 1).zfill(4)}"
                    service_id = f"{svc.upper()}_{str(len(service_polys_attrs) + 1).zfill(4)}"
                    for idx in site_idxs:
                        reserved_site_cells.add(idx)
                        cells.loc[idx, "is_service_site"] = True
                        cells.loc[idx, "site_id"] = site_id
                        cells.loc[idx, "service_site_type"] = svc
                    for idx in svc_cell_idxs:
                        reserved_service_cells.add(idx)
                        cells.loc[idx, "service"] = svc

                    site_poly = self.grid_operations.make_valid(
                        unary_union([cells.geometry[i] for i in site_idxs])
                    )
                    svc_poly = self.grid_operations.make_valid(
                        unary_union([cells.geometry[i] for i in svc_cell_idxs])
                    )

                    service_sites_geom.append(site_poly)
                    service_sites_attrs.append(
                        {
                            "site_id": site_id,
                            "service": svc,
                            "zone_id": int(zid),
                            "site_form": site_form_name,
                            "pattern_for_norms": pat_name,
                            "site_cells": int(len(site_idxs)),
                            "site_area_target_m2": float(site_area_m2),
                            "site_area_actual_m2": float(getattr(site_poly, "area", 0.0)),
                            "capacity": int(capacity),
                            "fallback_outside": True,
                        }
                    )
                    service_polys_geom.append(svc_poly)
                    service_polys_attrs.append(
                        {
                            "building_id": service_id,
                            "site_id": site_id,
                            "service": svc,
                            "pattern": chosen_pat,
                            "zone_id": int(zid),
                            "n_cells": int(len(svc_cell_idxs)),
                            "width_m": float(self.generation_parameters.cell_size_m),
                            "fallback_outside": True,
                        }
                    )

                    placed_site_sets.append(site_idxs)
                    placed_sites_by_type.setdefault(svc, []).append(site_idxs)
                    return True
        return False
