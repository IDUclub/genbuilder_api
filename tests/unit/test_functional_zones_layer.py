import asyncio

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from app.logic.functional_zones_service import FunctionalZonesService
from app.logic.zone_taxonomy import normalize_zone_column


def _square(x: float) -> Polygon:
    return Polygon([(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)])


def _zones_frame() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "zone": ["residential_individual", "mixed_use", "industrial"],
            "functional_zone_id": [11, 22, 33],
            "geometry": [_square(0), _square(2), _square(4)],
        },
        geometry="geometry",
        crs=4326,
    )


class _FakeUrbanDB:
    """Stands in for the UrbanDB gateway; records how it was called."""

    def __init__(self, frame: gpd.GeoDataFrame) -> None:
        self._frame = frame
        self.calls: list[tuple] = []

    async def get_territories_for_buildings(self, scenario_id, year, source, token):
        self.calls.append((scenario_id, year, source, token))
        return self._frame.copy()


def _prepare(frame: gpd.GeoDataFrame, zone_types=None):
    api = _FakeUrbanDB(frame)
    service = FunctionalZonesService(api)
    layer = asyncio.run(
        service.prepare_zones_layer(
            scenario_id=198,
            year=2024,
            source="OSM",
            token="user-token",
            functional_zone_types=zone_types,
        )
    )
    return layer, api


def test_normalize_zone_column_collapses_subtypes_and_derives_floor_group():
    frame = normalize_zone_column(_zones_frame())

    assert list(frame["zone"]) == ["residential", "business", "industrial"]
    assert frame["floors_group"].tolist()[0] == "private"
    assert pd.isna(frame["floors_group"].tolist()[1])


def test_normalize_zone_column_keeps_an_explicit_floor_group():
    frame = _zones_frame()
    frame["floors_group"] = ["high", None, None]

    frame = normalize_zone_column(frame)

    assert frame["floors_group"].tolist()[0] == "high"


def test_normalize_zone_column_is_a_noop_on_canonical_names():
    frame = _zones_frame()
    frame["zone"] = ["residential", "business", "industrial"]

    frame = normalize_zone_column(frame)

    assert list(frame["zone"]) == ["residential", "business", "industrial"]


def test_prepare_zones_layer_returns_a_feature_collection():
    layer, _ = _prepare(_zones_frame())

    assert layer["type"] == "FeatureCollection"
    assert len(layer["features"]) == 3
    assert layer["features"][0]["geometry"]["type"] == "Polygon"


def test_prepare_zones_layer_normalizes_zone_names():
    layer, _ = _prepare(_zones_frame())

    zones = [f["properties"]["zone"] for f in layer["features"]]
    assert zones == ["residential", "business", "industrial"]


def test_prepare_zones_layer_filters_by_requested_types():
    layer, _ = _prepare(_zones_frame(), ["residential", "business"])

    zones = [f["properties"]["zone"] for f in layer["features"]]
    assert zones == ["residential", "business"]


def test_prepare_zones_layer_keeps_functional_zone_id_for_joining():
    layer, _ = _prepare(_zones_frame(), ["residential"])

    assert layer["features"][0]["properties"]["functional_zone_id"] == 11


def test_prepare_zones_layer_forwards_the_caller_token():
    """UrbanDB must enforce access with the user's own token, not a service one."""
    _, api = _prepare(_zones_frame(), ["residential"])

    assert api.calls == [(198, 2024, "OSM", "user-token")]


def test_prepare_zones_layer_returns_empty_collection_when_nothing_matches():
    layer, _ = _prepare(_zones_frame(), ["transport"])

    assert layer == {"type": "FeatureCollection", "features": []}
