from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union


@dataclass
class BuildingAttributes:
    """
    Calculation of building floor counts and allocation of total living area for RESIDENTIAL buildings.

    The model relies ONLY on the boolean attribute `is_living` in `buildings_gdf`.

    Rules:
      1) For is_living=True:
         floors_count = round(area_m2 / area_per_floor_m2),
         then clipped to [1, floors_cap(zname)],
         where floors_cap is taken from targets_by_zone['floors_avg'][zname] (rounded);
         if missing, `default_zone_max_floor` is used.
      2) Living area: living_area = K * area_m2 * floors_count, with K ∈ [K_min, K_max].
      3) The target value targets_by_zone['la_target'][zname] is distributed by weights
         proportional to 1 / distance_to_zone_boundary, considering cap_min/cap_max (=K_min/K_max).
      4) If no target is defined for zname or it is ≤ 0 → living_area=0, K=NaN for all buildings in that zone.
      5) For is_living=False → living_area=0, K=NaN, _alloc_status="non_living → living=0".

    Returns:
      dict:
        - buildings: GeoDataFrame containing area_m2, floors_count, K, living_area, _alloc_status
        - summary:  DataFrame aggregated by zone name:
                    zone_name, target_requested, target_used, cap_min, cap_max,
                    n_buildings, mean_K, sum_living_area, status
    """

    area_per_floor_m2: float = 100.0
    K_min: float = 0.5
    K_max: float = 0.75
    default_zone_max_floor: int = 9

    dist_eps: float = 1e-6

    fallback_epsg: int = 3857

    verbose: bool = True

    def fit_transform(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        zones_gdf: gpd.GeoDataFrame,
        targets_by_zone: Dict[str, Dict[str, float | int]],
    ) -> Dict[str, object]:

        la_target_map = self._normalize_zone_map(targets_by_zone.get("la_target", {}))
        floors_avg_map = self._normalize_zone_map(targets_by_zone.get("floors_avg", {}))

        buildings = buildings_gdf.copy()
        zones = self._to_metric_crs(
            zones_gdf.copy(), like=buildings, fallback_epsg=self.fallback_epsg
        )
        buildings = self._to_metric_crs(
            buildings, like=zones, fallback_epsg=self.fallback_epsg
        )

        if "is_living" not in buildings.columns:
            raise KeyError(
                "Column 'is_living' is required in buildings_gdf when using is_living-only mode."
            )
        is_living = buildings["is_living"].astype(bool)

        zid_col, zname_col = self._zone_cols(zones)

        need_zone_join = (
            (zid_col not in buildings.columns)
            or buildings.get(zid_col).isna().any()
            or (zname_col not in buildings.columns)
            or buildings.get(zname_col).isna().any()
        )
        if need_zone_join:
            cent = buildings.geometry.representative_point()
            j = (
                gpd.sjoin(
                    gpd.GeoDataFrame(
                        {"_i": np.arange(len(buildings))},
                        geometry=cent,
                        crs=buildings.crs,
                    ),
                    zones[[zid_col, zname_col, "geometry"]],
                    how="left",
                    predicate="within",
                )
                .drop_duplicates("_i")
                .set_index("_i")[[zid_col, zname_col]]
            )
            buildings = buildings.drop(
                columns=[zid_col, zname_col], errors="ignore"
            ).join(j, how="left")

        buildings[zname_col] = buildings[zname_col].astype(str).str.lower().str.strip()
        buildings["area_m2"] = buildings.geometry.area.astype(float)

        buildings["max_floor_zone"] = (
            buildings[zname_col]
            .map(
                lambda zn: self._safe_int_round(
                    floors_avg_map.get(str(zn), self.default_zone_max_floor)
                )
            )
            .astype(int)
        )

        prev_floors = (
            buildings["floors_count"].copy()
            if "floors_count" in buildings.columns
            else None
        )

        if "floors_count" not in buildings.columns:
            buildings["floors_count"] = pd.Series(
                pd.NA, index=buildings.index, dtype="Float64"
            )

        mask_liv = is_living
        if mask_liv.any():
            floors_raw = (
                (buildings.loc[mask_liv, "area_m2"] / float(self.area_per_floor_m2))
                .round()
                .clip(lower=1)
            )
            floors_cap = buildings.loc[mask_liv, "max_floor_zone"].astype(float)
            floors_capped = pd.concat([floors_raw, floors_cap], axis=1).min(axis=1)
            buildings.loc[mask_liv, "floors_count"] = floors_capped

        if prev_floors is not None:
            buildings.loc[~mask_liv, "floors_count"] = prev_floors.loc[~mask_liv]

        buildings["floors_count"] = buildings["floors_count"].astype("Int64")

        buildings["K"] = pd.Series(np.nan, index=buildings.index, dtype="float64")
        buildings["living_area"] = 0.0

        zones[zname_col] = zones[zname_col].astype(str).str.lower().str.strip()
        zones["_boundary"] = zones.geometry.boundary

        summary_rows: List[Dict[str, Any]] = []

        mask_non = ~is_living
        if mask_non.any():
            buildings.loc[mask_non, "_alloc_status"] = "non_living → living=0"

        liv = buildings[mask_liv & buildings[zname_col].notna()].copy()
        if not liv.empty:
            for zname, sub in liv.groupby(liv[zname_col]):
                zname_str = str(zname)
                targ = la_target_map.get(zname_str, np.nan)
                targ = float(targ) if pd.notna(targ) else np.nan
                if (not np.isfinite(targ)) or targ <= 0:
                    buildings.loc[sub.index, "_alloc_status"] = "no_target → living=0"
                    summary_rows.append(
                        {
                            "zone_name": zname_str,
                            "target_requested": (
                                None if (not np.isfinite(targ)) else float(targ)
                            ),
                            "target_used": 0.0,
                            "cap_min": float(
                                (
                                    self.K_min * sub["area_m2"] * sub["floors_count"]
                                ).sum()
                            ),
                            "cap_max": float(
                                (
                                    self.K_max * sub["area_m2"] * sub["floors_count"]
                                ).sum()
                            ),
                            "n_buildings": int(len(sub)),
                            "mean_K": np.nan,
                            "sum_living_area": 0.0,
                            "status": "no_target",
                        }
                    )
                    continue

                cap_min = (self.K_min * sub["area_m2"] * sub["floors_count"]).astype(
                    float
                )
                cap_max = (self.K_max * sub["area_m2"] * sub["floors_count"]).astype(
                    float
                )
                cap_min_tot = float(cap_min.sum())
                cap_max_tot = float(cap_max.sum())
                T = float(np.clip(targ, cap_min_tot, cap_max_tot))
                Z = zones.loc[zones[zname_col] == zname_str]
                if len(Z) == 0:
                    boundary = unary_union(list(zones["_boundary"].values))
                else:
                    boundary = unary_union(list(Z["_boundary"].values))
                dists = (
                    sub.geometry.representative_point().distance(boundary).astype(float)
                )
                weights = 1.0 / (self.dist_eps + dists)
                if (not np.isfinite(weights).any()) or (weights.sum() == 0):
                    weights = pd.Series(1.0, index=sub.index)

                deltas = (cap_max - cap_min).astype(float)
                D = T - cap_min_tot
                x = self._waterfill_with_caps(
                    weights.to_numpy(), deltas.to_numpy(), float(D)
                )

                living = cap_min.values + x
                denom = sub["area_m2"].values * sub["floors_count"].values
                K = np.divide(living, denom, out=np.zeros_like(living), where=denom > 0)
                K = np.clip(K, self.K_min, self.K_max)

                buildings.loc[sub.index, "K"] = K
                buildings.loc[sub.index, "living_area"] = living
                status = f"T_used={T:.2f}; cap_min={cap_min_tot:.2f}; cap_max={cap_max_tot:.2f}"
                buildings.loc[sub.index, "_alloc_status"] = status

                summary_rows.append(
                    {
                        "zone_name": zname_str,
                        "target_requested": float(targ),
                        "target_used": float(T),
                        "cap_min": cap_min_tot,
                        "cap_max": cap_max_tot,
                        "n_buildings": int(len(sub)),
                        "mean_K": float(np.mean(K)) if len(K) else np.nan,
                        "sum_living_area": float(np.sum(living)),
                        "status": status,
                    }
                )

        summary_df = pd.DataFrame(summary_rows)
        zones.drop(columns=["_boundary"], inplace=True, errors="ignore")
        if self.verbose:
            meanK = (
                summary_df["mean_K"].mean(skipna=True)
                if len(summary_df) > 0
                else float("nan")
            )
            print(
                f"OK | objects={len(buildings)}, living_houses={int(is_living.sum())}, "
                f"zones_with_target={summary_df.shape[0]} | mean(K)={meanK:.3f}"
            )

        return {"buildings": buildings, "summary": summary_df}

    @staticmethod
    def _normalize_zone_map(d: Dict[str, Any]) -> Dict[str, Any]:
        if not d:
            return {}
        return {str(k).lower().strip(): v for k, v in d.items()}

    @staticmethod
    def _safe_int_round(x: float | int | None) -> int:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return 0
        try:
            return int(round(float(x)))
        except Exception:
            return 0

    @staticmethod
    def _to_metric_crs(
        gdf: gpd.GeoDataFrame,
        like: gpd.GeoDataFrame | None = None,
        fallback_epsg: int = 3857,
    ) -> gpd.GeoDataFrame:
        if like is not None and like.crs is not None:
            try:
                return gdf.to_crs(like.crs)
            except Exception:
                if gdf.crs is None:
                    return gdf.set_crs(like.crs, allow_override=True)
                raise
        if gdf.crs is None:
            return gdf.set_crs(epsg=fallback_epsg, allow_override=True)
        if getattr(gdf.crs, "is_projected", None) is True:
            return gdf
        return gdf.to_crs(epsg=fallback_epsg)

    @staticmethod
    def _zone_cols(zones: gpd.GeoDataFrame) -> tuple[str, str]:
        zid = (
            "zone_id"
            if "zone_id" in zones.columns
            else ("id" if "id" in zones.columns else "ZONE_ID")
        )
        if zid not in zones.columns:
            zones[zid] = np.arange(len(zones))
        zname = "zone"
        if zname not in zones.columns:
            for alt in ["zone_name", "zone_type", "functional_zone_type_name"]:
                if alt in zones.columns:
                    zname = alt
                    break
            else:
                zones["zone"] = "unknown"
                zname = "zone"
        return zid, zname

    @staticmethod
    def _waterfill_with_caps(
        weights: np.ndarray, caps: np.ndarray, demand: float, eps: float = 1e-9
    ) -> np.ndarray:
        n = len(weights)
        x = np.zeros(n, dtype=float)
        remain = np.arange(n)
        D = float(max(demand, 0.0))
        w = np.asarray(weights, dtype=float).copy()
        w[w < 0] = 0.0
        caps = np.asarray(caps, dtype=float)

        while len(remain) > 0 and D > eps:
            sw = float(w[remain].sum())
            if sw <= eps:
                break
            inc = np.zeros_like(x)
            share = D / sw
            for i in remain:
                quota = share * w[i]
                inc[i] = min(quota, caps[i] - x[i])
            inc_sum = float(inc[remain].sum())
            if inc_sum <= eps:
                break
            x += inc
            D -= inc_sum
            remain = np.array([i for i in remain if (caps[i] - x[i]) > eps], dtype=int)
        return x


attributes_calculator = BuildingAttributes()
