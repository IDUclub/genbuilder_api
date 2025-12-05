import math
import pandas as pd

from app.logic.building_params import BuildingType
from app.common.geo_utils import safe_float

def infer_building_type(row: pd.Series, mode: str) -> BuildingType | None:
    zone = str(row.get("zone", "")).strip().lower()
    floors_group = row.get("floors_group")
    floors_avg_raw = row.get("floors_avg")
    floors_avg_val = safe_float(floors_avg_raw, default=math.nan)
    floors_avg: float | None = None if math.isnan(floors_avg_val) else floors_avg_val

    try:
        geom = row.geometry
        block_area = safe_float(getattr(geom, "area", 0.0), default=0.0)
    except Exception:
        block_area = 0.0
    if zone == "residential" or mode == "residential":
        if floors_avg is None:
            return BuildingType.MKD_5_8
        if floors_avg < 3:
            return BuildingType.MKD_2_4
        if floors_avg < 9:
            return BuildingType.MKD_5_8
        if floors_avg < 17:
            return BuildingType.MKD_9_16
        return BuildingType.HIGHRISE
    if zone in {"business", "unknown"}:
        if floors_avg is None:
            return BuildingType.BIZ_MID

        if floors_avg <= 4:
            if block_area >= 15000.0:
                return BuildingType.BIZ_MALL
            else:
                return BuildingType.BIZ_LOW
        if floors_avg <= 8:
            return BuildingType.BIZ_MID
        return BuildingType.BIZ_TOWER

    if zone == "industrial":
        if floors_avg is None:
            return BuildingType.IND_LIGHT
        if floors_avg < 2:
            return BuildingType.IND_WAREHOUSE
        if floors_avg < 4:
            return BuildingType.IND_LIGHT
        return BuildingType.IND_HEAVY

    if zone == "transport":
        if floors_avg is None:
            return BuildingType.TR_STATION
        if floors_avg <= 2:
            return BuildingType.TR_DEPOT
        if floors_avg <= 4:
            return BuildingType.TR_STATION
        return BuildingType.TR_PARKING

    if zone == "special":
        if block_area >= 20000.0:
            return BuildingType.SPEC_WASTE
        return BuildingType.SPEC_TECH
    return None