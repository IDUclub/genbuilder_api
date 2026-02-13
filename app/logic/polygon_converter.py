import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import math
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer

from shapely.geometry import shape, mapping, Polygon, MultiPolygon

from app.constants.constants import MIN_AREA_M2_BY_ZONE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolygonPart:
    """Single polygon part extracted from an input geometry."""
    index: int
    geometry: Dict[str, Any]
    area_weight: float


def _explode_to_polygons(geometry: Dict[str, Any], min_area_weight: float = 0.0) -> List[PolygonPart]:
    """
    Explode Polygon/MultiPolygon GeoJSON geometry into polygon parts.

    min_area_weight:
        Drop very small parts by relative area weight threshold.
        Example: 0.02 drops polygons smaller than 2% of total area.
    """
    geom = shape(geometry)

    if isinstance(geom, Polygon):
        return [PolygonPart(index=0, geometry=geometry, area_weight=1.0)]

    if not isinstance(geom, MultiPolygon):
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")

    polys = list(geom.geoms)
    if not polys:
        return []

    areas = [p.area for p in polys]
    total = sum(areas) or 0.0
    if total <= 0:
        w = 1.0 / len(polys)
        return [PolygonPart(index=i, geometry=mapping(p), area_weight=w) for i, p in enumerate(polys)]

    parts: List[PolygonPart] = []
    for i, (p, a) in enumerate(zip(polys, areas)):
        w = a / total
        if w >= min_area_weight:
            parts.append(PolygonPart(index=i, geometry=mapping(p), area_weight=w))

    if not parts:
        i_max = max(range(len(polys)), key=lambda i: areas[i])
        parts = [PolygonPart(index=i_max, geometry=mapping(polys[i_max]), area_weight=1.0)]

    w_sum = sum(p.area_weight for p in parts) or 1.0
    parts = [PolygonPart(p.index, p.geometry, p.area_weight / w_sum) for p in parts]
    return parts


def _scale_numeric_targets(targets_by_zone: Optional[Dict[str, Any]], weight: float) -> Optional[Dict[str, Any]]:
    """
    Scale numeric leaf targets by weight.
    Keeps non-numeric values unchanged.

    This is a pragmatic approach for targets like:
    - residents_number / residents_target
    - la_target / living_area_target
    - coverage_area
    - etc.
    """
    if not targets_by_zone:
        return targets_by_zone

    def scale(v: Any) -> Any:
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, int):
            return int(round(v * weight))
        if isinstance(v, float):
            return float(v * weight)
        if isinstance(v, dict):
            return {k: scale(val) for k, val in v.items()}
        if isinstance(v, list):
            return [scale(x) for x in v]
        return v

    return scale(targets_by_zone)


def _make_block_feature(base_props: Dict[str, Any], zone_type: str, zone_id: int, part_index: int, geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Build a single block feature with a unique block_id."""
    return {
        "type": "Feature",
        "properties": {
            **base_props,
            "block_id": f"{zone_id}:{part_index}",
            "zone": zone_type,
            "functional_zone_id": zone_id,
        },
        "geometry": geometry,
    }

def _utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """
    Pick UTM zone EPSG for a lon/lat point.
    Northern hemisphere: EPSG:326xx, southern: EPSG:327xx.
    """
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    return 32600 + zone if lat >= 0 else 32700 + zone

def _area_m2_geojson(geometry: Dict[str, Any]) -> float:
    """
    Compute area in m^2 for GeoJSON geometry in WGS84 lon/lat.
    Uses UTM zone chosen by centroid.
    """
    geom = shape(geometry)
    if geom.is_empty:
        return 0.0

    c = geom.centroid
    utm_epsg = _utm_epsg_for_lonlat(c.x, c.y)

    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_epsg(utm_epsg),
        always_xy=True,
    )

    geom_m = shp_transform(transformer.transform, geom)
    return float(geom_m.area)

def _filter_polygon_parts_by_min_area(
    parts: List[Dict[str, Any]],
    zone_type: str,
) -> List[Dict[str, Any]]:
    """
    Keep only polygon parts whose area >= min threshold for the zone_type.
    If zone_type isn't in the table, no filtering is applied.
    """
    min_m2 = MIN_AREA_M2_BY_ZONE.get((zone_type or "").lower())
    if min_m2 is None:
        return parts

    kept: List[Dict[str, Any]] = []
    dropped = 0

    for g in parts:
        try:
            a = _area_m2_geojson(g)
        except Exception:
            # safer to keep than to silently drop on area calculation failure
            kept.append(g)
            continue

        if a >= min_m2:
            kept.append(g)
        else:
            dropped += 1

    if dropped:
        logger.info("Dropped %s small polygon parts for zone_type=%s (min_m2=%s)", dropped, zone_type, min_m2)

    return kept

@dataclass(frozen=True)
class DroppedPart:
    """Info about a polygon part dropped by filtering rules."""
    part_index: int
    area_m2: float
    threshold_m2: float
    reason: str

@dataclass(frozen=True)
class DroppedPartsReport:
    """Summary of filtering results for one functional zone."""
    zone_id: int
    zone_type: str
    dropped_count: int
    kept_count: int
    dropped: List[DroppedPart]


def _renormalize_weights(parts: List["PolygonPart"]) -> List["PolygonPart"]:
    """Renormalize area_weight to sum to 1.0."""
    if not parts:
        return parts
    w_sum = sum(p.area_weight for p in parts) or 1.0
    return [PolygonPart(p.index, p.geometry, p.area_weight / w_sum) for p in parts]


def _filter_parts_by_zone_min_area(
    *,
    zone_id: int,
    zone_type: str,
    parts: List["PolygonPart"],
    min_area_m2_by_zone: Dict[str, float],
) -> Tuple[List["PolygonPart"], Optional[DroppedPartsReport]]:
    """
    Filter PolygonPart list by minimal area threshold (m²) for the given zone type.
    Returns kept parts and a report (or None if nothing dropped / no rule).
    """
    if not parts:
        return parts, None

    zone_type_norm = (zone_type or "").lower()
    threshold = min_area_m2_by_zone.get(zone_type_norm)
    if threshold is None:
        return parts, None

    kept: List["PolygonPart"] = []
    dropped: List[DroppedPart] = []

    for p in parts:
        area_m2 = _area_m2_geojson(p.geometry)
        if area_m2 < threshold:
            dropped.append(
                DroppedPart(
                    part_index=p.index,
                    area_m2=area_m2,
                    threshold_m2=threshold,
                    reason="below_min_area_m2",
                )
            )
        else:
            kept.append(p)

    if not kept:
        biggest = max(parts, key=lambda x: _area_m2_geojson(x.geometry))
        kept = [PolygonPart(biggest.index, biggest.geometry, 1.0)]
        dropped = [
            DroppedPart(
                part_index=p.index,
                area_m2=_area_m2_geojson(p.geometry),
                threshold_m2=threshold,
                reason="below_min_area_m2_but_kept_biggest_fallback",
            )
            for p in parts
            if p.index != biggest.index
        ]

    kept = _renormalize_weights(kept)

    report = DroppedPartsReport(
        zone_id=zone_id,
        zone_type=zone_type_norm,
        dropped_count=len(dropped),
        kept_count=len(kept),
        dropped=dropped,
    )

    # “Вывод” в лог: компактно, без спама
    if report.dropped_count > 0:
        min_area = min(d.area_m2 for d in report.dropped)
        max_area = max(d.area_m2 for d in report.dropped)
        logger.info(
            "Zone %s (%s): dropped %s polygon parts below %s m² (dropped_area_m2 min=%.1f max=%.1f), kept=%s",
            zone_id,
            zone_type_norm,
            report.dropped_count,
            threshold,
            min_area,
            max_area,
            report.kept_count,
        )

    return kept, report
