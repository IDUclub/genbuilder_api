from dataclasses import dataclass
from typing import Any, Iterable

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from loguru import logger

def _keep_polygonal(geom: BaseGeometry | None) -> BaseGeometry | None:
    """Return only polygonal part of geometry."""
    if geom is None or geom.is_empty:
        return None

    gtype = geom.geom_type
    if gtype in {"Polygon", "MultiPolygon"}:
        return geom

    if gtype == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in {"Polygon", "MultiPolygon"}]
        if not polys:
            return None
        return unary_union(polys)

    return None

@dataclass(frozen=True)
class PhysicalObjectsService:
    """Select physical objects features by their ids from GeoJSON FeatureCollection."""
    def select_features_by_ids(
        self,
        fc: dict[str, Any] | None,
        requested_ids: Iterable[int],
    ) -> list[dict[str, Any]]:
        """Return list of GeoJSON features matching requested physical_object_id."""
        if not fc:
            return []

        features = fc.get("features") or []
        if not features:
            return []

        ids_set = {int(x) for x in requested_ids if x is not None}
        if not ids_set:
            return []

        selected: list[dict[str, Any]] = []
        missing_id_count = 0

        for f in features:
            props = f.get("properties") or {}
            obj_id = props.get("physical_object_id")

            if obj_id is None:
                missing_id_count += 1
                continue

            try:
                if int(obj_id) in ids_set:
                    selected.append(f)
            except Exception:
                continue

        if missing_id_count:
            logger.debug(
                "PhysicalObjectsSelector: some features had no physical_object_id in properties "
                f"(count={missing_id_count})"
            )

        return selected


    def exclude(
        self,
        blocks: gpd.GeoDataFrame,
        physical_objects: gpd.GeoDataFrame,
        buffer_m: float = 0.0,
    ) -> gpd.GeoDataFrame:
        """Return blocks GeoDataFrame with excluded areas removed."""
        if blocks.empty or physical_objects.empty:
            return blocks

        objs = physical_objects.copy()
        objs = objs[objs.geometry.notna() & ~objs.geometry.is_empty]
        if objs.empty:
            return blocks

        if buffer_m and float(buffer_m) > 0:
            logger.info(
                f"PhysicalObjectsExcluder.exclude: buffering physical objects by {float(buffer_m)}m"
            )
            objs["geometry"] = objs.geometry.buffer(float(buffer_m))

        exclusion_geom = unary_union(list(objs["geometry"]))
        if exclusion_geom is None or exclusion_geom.is_empty:
            return blocks

        out = blocks.copy()

        def _diff(g: BaseGeometry | None) -> BaseGeometry | None:
            if g is None or g.is_empty:
                return None
            try:
                d = g.difference(exclusion_geom)
            except Exception as e:
                logger.warning(
                    "PhysicalObjectsExcluder.exclude: difference failed, keeping original geometry: "
                    f"{e}"
                )
                return g

            d = _keep_polygonal(d)
            if d is None or d.is_empty:
                return None

            try:
                return d.buffer(0)
            except Exception:
                return d

        out["geometry"] = out.geometry.apply(_diff)
        out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()

        logger.info(
            "PhysicalObjectsExcluder.exclude: blocks reduced "
            f"from {len(blocks)} to {len(out)} after exclusion"
        )
        return out
