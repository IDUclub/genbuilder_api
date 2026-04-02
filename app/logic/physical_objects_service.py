from copy import deepcopy
from typing import Any, Iterable, Optional, TYPE_CHECKING

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from loguru import logger

if TYPE_CHECKING:
    from app.logic.building_params import BuildingParamsProvider, BuildingType


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


class PhysicalObjectsService:
    """Select physical objects features by their ids from GeoJSON FeatureCollection."""

    def select_features_by_ids(
            self,
            fc: dict[str, Any] | None,
            requested_ids: Iterable[int],
    ) -> list[dict[str, Any]]:
        """
        Return excluded physical objects normalized to generated_buildings schema.

        Contract:
        - same flat properties shape as generated_buildings
        - custom flags: is_excluded, physical_object_id
        - living_area > 0 only for residential physical objects
          (physical_object_type_id == 4 and name == "Жилой дом")
        """
        if not fc:
            return []

        features = fc.get("features") or []
        if not features:
            return []

        ids_set = {int(x) for x in requested_ids if x is not None}
        if not ids_set:
            return []

        selected: list[dict[str, Any]] = []

        for feature in features:
            feature_properties = feature.get("properties") or {}
            raw_physical_object_id = feature_properties.get("physical_object_id")
            if raw_physical_object_id is None:
                continue

            try:
                physical_object_id = int(raw_physical_object_id)
            except (TypeError, ValueError):
                continue

            if physical_object_id not in ids_set:
                continue

            source_props = feature_properties.get("properties") or {}
            building = feature_properties.get("building") or {}
            physical_object_type = feature_properties.get("physical_object_type") or {}

            physical_object_type_id = physical_object_type.get("physical_object_type_id")
            physical_object_type_name = physical_object_type.get("name")

            is_residential_physical_object = (
                    physical_object_type_id == 4
                    and physical_object_type_name == "Жилой дом"
            )

            if is_residential_physical_object:
                try:
                    living_area = float(source_props.get("living_area"))
                except (TypeError, ValueError):
                    living_area = 1.0

                if living_area <= 0:
                    living_area = 1.0
            else:
                living_area = 0.0

            try:
                building_area = float(source_props.get("building_area"))
            except (TypeError, ValueError):
                official_area = building.get("building_area_official")
                try:
                    building_area = float(official_area)
                except (TypeError, ValueError):
                    building_area = 0.0

            try:
                floors_count = float(
                    source_props.get("floors_count", building.get("floors", 0.0))
                )
            except (TypeError, ValueError):
                floors_count = 0.0

            try:
                residents_number = float(source_props.get("residents_number"))
            except (TypeError, ValueError):
                residents_number = 0.0

            if not is_residential_physical_object:
                residents_number = 0.0

            service_value = source_props.get("service")
            if not isinstance(service_value, list):
                service_value = []

            result_properties = {
                "floors_count": floors_count,
                "living_area": living_area,
                "building_area": building_area,
                "service": service_value,
                "broke_restriction_zone": bool(
                    source_props.get("broke_restriction_zone", False)
                ),
                "building_type": source_props.get("building_type"),
                "zone": source_props.get("zone"),
                "residents_number": residents_number,
                "is_excluded": True,
                "physical_object_id": physical_object_id,
            }

            selected.append(
                {
                    "type": "Feature",
                    "id": feature.get("id"),
                    "geometry": deepcopy(feature.get("geometry")),
                    "properties": result_properties,
                }
            )

        logger.info(
            "PhysicalObjectsService.select_features_by_ids: selected {} features by ids={}",
            len(selected),
            sorted(ids_set),
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

        objs = physical_objects
        objs = objs[objs.geometry.notna() & ~objs.geometry.is_empty]
        if objs.empty:
            return blocks

        objs = objs.copy()

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
            "from {} to {} after exclusion",
            len(blocks),
            len(out)
        )
        return out

    def apply_dynamic_buffer(
        self,
        physical_objects: gpd.GeoDataFrame,
        building_params_provider: "BuildingParamsProvider",
        min_buffer_m: float = 3.0,
        max_buffer_m: float = 60.0,
        building_types: Optional[list["BuildingType"]] = None,
    ) -> gpd.GeoDataFrame:
        """Apply per-object buffer based on geometry size and building_params.

        The method assumes that geometries are already in a metric CRS (e.g., UTM),
        so polygon area is in square meters and buffering value is in meters.

        Heuristic:
        - For each BuildingType, compute a 'typical' footprint area based on the
          midpoints of length/width ranges.
        - For each object geometry, choose BuildingType with the closest relative
          footprint area.
        - Use max(setback_ff, setback_ft, setback_tt) as buffer and clamp it to
          [min_buffer_m, max_buffer_m].
        """
        if physical_objects.empty:
            return physical_objects

        params_by_type = building_params_provider.current().params_by_type
        scope = building_types or list(params_by_type.keys())
        scope = [bt for bt in scope if bt in params_by_type]

        if not scope:
            logger.warning(
                "PhysicalObjectsService.apply_dynamic_buffer: empty building type scope; using min_buffer"
            )
            scope = [BuildingType.IZH] if BuildingType.IZH in params_by_type else list(params_by_type.keys())

        typical_area: dict[BuildingType, float] = {}
        typical_buffer: dict[BuildingType, float] = {}
        for bt in scope:
            p = params_by_type[bt]
            length_mid = (min(p.building_length_range) + max(p.building_length_range)) / 2.0
            width_mid = (min(p.building_width_range) + max(p.building_width_range)) / 2.0
            typical_area[bt] = float(length_mid * width_mid)
            typical_buffer[bt] = float(max(p.setback_ff, p.setback_ft, p.setback_tt))

        def _choose_buffer_m(geom: BaseGeometry | None) -> float:
            if geom is None or geom.is_empty:
                return float(min_buffer_m)

            area = float(getattr(geom, "area", 0.0) or 0.0)
            if area <= 0:
                return float(min_buffer_m)

            best_bt: Optional[BuildingType] = None
            best_score = float("inf")
            for bt, ta in typical_area.items():
                if ta <= 0:
                    continue
                score = abs(area - ta) / ta
                if score < best_score:
                    best_score = score
                    best_bt = bt

            if best_bt is None:
                return float(min_buffer_m)

            buf = typical_buffer.get(best_bt, float(min_buffer_m))
            buf = max(float(min_buffer_m), min(float(max_buffer_m), float(buf)))
            return float(buf)

        out = physical_objects.copy()
        out["__buffer_m"] = out.geometry.apply(_choose_buffer_m)
        out["geometry"] = [
            (g.buffer(float(bm)) if g is not None and not g.is_empty else g)
            for g, bm in zip(out.geometry, out["__buffer_m"])
        ]

        logger.info(
            "PhysicalObjectsService.apply_dynamic_buffer: applied per-object buffers "
            "(min={}, max={}), objects_count={}",
            min_buffer_m,
            max_buffer_m,
            len(out)
        )

        return out
