import math
from shapely.geometry.base import BaseGeometry


def usable_per_building(L: float, W: float, floors: float, la_coef: float) -> float:
    return max(L, 0.0) * max(W, 0.0) * max(floors, 0.0) * max(la_coef, 0.0)


def far_from_dims(L: float, W: float, floors: float, plot_area: float) -> float:
    if plot_area <= 0:
        return float("nan")
    return (L * W * floors) / plot_area


def building_need(target_area: float, usable_one: float) -> int:
    if target_area <= 0 or usable_one <= 0:
        return 0
    return int(math.ceil(target_area / usable_one))


def building_area(geom: BaseGeometry, floors: float) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    try:
        f = float(floors)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f) or f <= 0:
        return 0.0
    return geom.area * f