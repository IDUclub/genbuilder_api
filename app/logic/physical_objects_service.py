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
        Return list of GeoJSON features matching requested physical_object_id.

        Supports both payload shapes:
        1. flat feature.properties.physical_object_id
        2. nested feature.properties.physical_objects[]
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

            # Case 1: flat schema
            raw_feature_object_id = feature_properties.get("physical_object_id")
            if raw_feature_object_id is not None:
                try:
                    feature_object_id = int(raw_feature_object_id)
                except (TypeError, ValueError):
                    feature_object_id = None

                if feature_object_id in ids_set:
                    result_properties = deepcopy(feature_properties)

                    source_props = result_properties.get("properties") or {}
                    raw_living_area = source_props.get("living_area", result_properties.get("living_area"))

                    try:
                        living_area = float(raw_living_area)
                    except (TypeError, ValueError):
                        living_area = 1.0

                    if living_area <= 0:
                        living_area = 1.0

                    building = result_properties.get("building") or {}
                    physical_object_type = result_properties.get("physical_object_type") or {}

                    result_properties["is_excluded"] = True
                    result_properties["physical_object_id"] = feature_object_id
                    result_properties["living_area"] = living_area

                    if result_properties.get("floors_count") is None and building.get("floors") is not None:
                        result_properties["floors_count"] = building.get("floors")

                    if result_properties.get("building_area") is None:
                        nested_building_area = source_props.get("building_area")
                        official_building_area = building.get("building_area_official")
                        if nested_building_area is not None:
                            result_properties["building_area"] = nested_building_area
                        elif official_building_area is not None:
                            result_properties["building_area"] = official_building_area

                    if physical_object_type and result_properties.get("building_type") is None:
                        result_properties["building_type"] = physical_object_type.get("name")

                    selected.append(
                        {
                            "type": "Feature",
                            "id": feature.get("id"),
                            "geometry": deepcopy(feature.get("geometry")),
                            "properties": result_properties,
                        }
                    )
                    continue

            # Case 2: nested schema
            physical_objects = feature_properties.get("physical_objects") or []
            if not isinstance(physical_objects, list) or not physical_objects:
                continue

            for physical_object in physical_objects:
                raw_physical_object_id = physical_object.get("physical_object_id")
                if raw_physical_object_id is None:
                    continue

                try:
                    physical_object_id = int(raw_physical_object_id)
                except (TypeError, ValueError):
                    continue

                if physical_object_id not in ids_set:
                    continue

                result_properties = deepcopy(feature_properties)

                building = physical_object.get("building") or {}
                building_properties = building.get("properties") or {}
                physical_object_type = physical_object.get("physical_object_type") or {}

                raw_living_area = (
                    building_properties.get("living_area")
                    if building_properties.get("living_area") is not None
                    else result_properties.get("living_area")
                )

                try:
                    living_area = float(raw_living_area)
                except (TypeError, ValueError):
                    living_area = 1.0

                if living_area <= 0:
                    living_area = 1.0

                building_area = (
                    building.get("building_area_official")
                    if building.get("building_area_official") is not None
                    else result_properties.get("building_area")
                )

                floors_count = (
                    building.get("floors")
                    if building.get("floors") is not None
                    else result_properties.get("floors_count")
                )

                result_properties["is_excluded"] = True
                result_properties["physical_object_id"] = physical_object_id
                result_properties["living_area"] = living_area

                if floors_count is not None:
                    result_properties["floors_count"] = floors_count

                if building_area is not None:
                    result_properties["building_area"] = building_area

                if physical_object_type and result_properties.get("building_type") is None:
                    result_properties["building_type"] = physical_object_type.get("name")

                selected.append(
                    {
                        "type": "Feature",
                        "id": feature.get("id"),
                        "geometry": deepcopy(feature.get("geometry")),
                        "properties": result_properties,
                    }
                )
                break

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
