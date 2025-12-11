from __future__ import annotations

import math
from typing import Any, Optional, Dict, Tuple, List

import geopandas as gpd
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from app.logic.building_params import (
    BuildingType,
    BuildingParams,
    PARAMS_BY_TYPE,
)


def _normalize_building_type(value: Any) -> Optional[BuildingType]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, BuildingType):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return BuildingType(value)
        except ValueError:
            return BuildingType[value]

    raise TypeError(f"Unsupported building_type value: {value!r}")


def _facade_and_gable_edges(geom: Polygon) -> Tuple[List[LineString], List[LineString]]:
    if geom is None or geom.is_empty:
        return [], []
    mrr = geom.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    if len(coords) < 4:
        return [], []
    base = coords[:4]
    segments: List[Tuple[float, LineString]] = []
    for i in range(4):
        p1 = base[i]
        p2 = base[(i + 1) % 4]
        line = LineString([p1, p2])
        length = line.length
        segments.append((length, line))
    if len(segments) < 4:
        return [], []
    segments.sort(key=lambda x: x[0], reverse=True)
    facades = [segments[0][1], segments[1][1]]
    gables = [segments[2][1], segments[3][1]]

    return facades, gables


def _min_distance_between_sets(a: List[LineString], b: List[LineString]) -> float:
    if not a or not b:
        return math.inf
    best = math.inf
    for g1 in a:
        for g2 in b:
            d = g1.distance(g2)
            if d < best:
                best = d
    return best


def _union_lines(lines: List[LineString]) -> LineString | Polygon:
    if len(lines) == 1:
        return lines[0]
    return unary_union(lines)


def check_buildings_setbacks(
    buildings_all: gpd.GeoDataFrame,
    *,
    setbacks_geojson_path: str | None = None,
    buildings_geojson_path: str | None = None,
) -> gpd.GeoDataFrame:

    gdf = buildings_all.copy()

    norm_types: Dict[int, Optional[BuildingType]] = {}
    for idx, val in gdf["building_type"].items():
        norm_types[idx] = _normalize_building_type(val)

    for idx, bt in norm_types.items():
        if bt is None:
            continue
        if bt not in PARAMS_BY_TYPE:
            raise KeyError(f"No BuildingParams found for building_type={bt!r}")

    facade_edges: Dict[int, List[LineString]] = {}
    gable_edges: Dict[int, List[LineString]] = {}

    for idx, geom in gdf.geometry.items():
        if not isinstance(geom, Polygon) or geom.is_empty:
            raise ValueError(f"Building {idx}: geometry is empty or not a Polygon")

        f_edges, t_edges = _facade_and_gable_edges(geom)
        facade_edges[idx] = f_edges
        gable_edges[idx] = t_edges

        if not f_edges and not t_edges:
            raise ValueError(
                f"Building {idx}: cannot derive facades/gables from geometry "
                f"(degenerate MRR or invalid footprint)"
            )

    sindex = gdf.sindex

    status_flags: Dict[int, Optional[str]] = {}
    setback_records: List[Dict[str, Any]] = []

    for idx, row in gdf.iterrows():
        bt = norm_types[idx]

        if bt is None:
            status_flags[idx] = None
            continue

        params: BuildingParams = PARAMS_BY_TYPE[bt]

        ff = params.setback_ff
        ft = params.setback_ft
        tt = params.setback_tt

        geom = row.geometry
        if geom is None or geom.is_empty:
            raise ValueError(f"Building {idx}: geometry is empty or None")

        fac_i = facade_edges[idx]
        gab_i = gable_edges[idx]

        radius = max(ff, ft, tt)
        if radius <= 0:
            raise ValueError(
                f"Building {idx}: non-positive setback radius for building_type={bt!r}: "
                f"ff={ff}, ft={ft}, tt={tt}"
            )

        minx, miny, maxx, maxy = geom.bounds
        query_bounds = (minx - radius, miny - radius, maxx + radius, maxy + radius)
        candidate_idxs = list(sindex.intersection(query_bounds))

        if not candidate_idxs or (len(candidate_idxs) == 1 and candidate_idxs[0] == idx):
            status_flags[idx] = "ok"
        else:
            checked_any_neighbor = False
            broke = False

            for j in candidate_idxs:
                if j == idx:
                    continue

                geom_j = gdf.at[j, "geometry"]
                if geom_j is None or geom_j.is_empty:
                    raise ValueError(f"Building {j}: geometry is empty or None")

                fac_j = facade_edges[j]
                gab_j = gable_edges[j]
                if not fac_j and not gab_j:
                    raise ValueError(
                        f"Building {j}: cannot derive facades/gables from geometry "
                        f"(degenerate MRR or invalid footprint)"
                    )

                checked_any_neighbor = True

                d_ff = _min_distance_between_sets(fac_i, fac_j)
                d_fi_tj = _min_distance_between_sets(fac_i, gab_j)
                d_ti_fj = _min_distance_between_sets(gab_i, fac_j)
                d_ft = min(d_fi_tj, d_ti_fj)
                d_tt = _min_distance_between_sets(gab_i, gab_j)

                if (d_ff < ff) or (d_ft < ft) or (d_tt < tt):
                    broke = True
                    break

            if not checked_any_neighbor:
                status_flags[idx] = "ok"
            else:
                status_flags[idx] = "broke" if broke else "ok"

        bt_str = bt.value if isinstance(bt, BuildingType) else str(bt)

        if fac_i and ff > 0:
            fac_union = _union_lines(fac_i)
            ff_zone = fac_union.buffer(ff)
            setback_records.append(
                {
                    "src_index": idx,
                    "building_type": bt_str,
                    "setback_kind": "ff",
                    "setback_value": ff,
                    "geometry": ff_zone,
                }
            )

        if gab_i and tt > 0:
            gab_union = _union_lines(gab_i)
            tt_zone = gab_union.buffer(tt)
            setback_records.append(
                {
                    "src_index": idx,
                    "building_type": bt_str,
                    "setback_kind": "tt",
                    "setback_value": tt,
                    "geometry": tt_zone,
                }
            )
        if (fac_i or gab_i) and ft > 0:
            ft_lines: List[LineString] = []
            ft_lines.extend(fac_i)
            ft_lines.extend(gab_i)
            ft_union = _union_lines(ft_lines)
            ft_zone = ft_union.buffer(ft)
            setback_records.append(
                {
                    "src_index": idx,
                    "building_type": bt_str,
                    "setback_kind": "ft",
                    "setback_value": ft,
                    "geometry": ft_zone,
                }
            )
    if set(status_flags.keys()) != set(gdf.index):
        missing = set(gdf.index) - set(status_flags.keys())
        raise RuntimeError(f"Missing setback status for buildings: {sorted(missing)}")

    gdf["setback_status"] = gdf.index.map(status_flags.get)
    gdf["broke_restriction_zone"] = gdf["setback_status"] == "broke"

    return gdf
