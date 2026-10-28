"""Service building templates keep their orientation and fit their own plots."""

import math

import geopandas as gpd
import pytest
from shapely.geometry import box

from app.logic.generation_params import GenParams, ParamsProvider
from app.logic.service_generation import ServiceGenerator

KAZAN_UTM = 32639
LENINGRAD_UTM = 32636


def _long_axis_angle(geom) -> float:
    coords = list(geom.minimum_rotated_rectangle.exterior.coords)
    edges = sorted(
        (
            (math.dist(coords[i], coords[i + 1]), coords[i], coords[i + 1])
            for i in range(2)
        ),
        reverse=True,
    )
    _, p0, p1 = edges[0]
    angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 180.0
    return min(angle, 180.0 - angle)


def _shipped_templates() -> gpd.GeoDataFrame:
    return ServiceGenerator(ParamsProvider(GenParams())).load_service_projects()


def test_distant_template_is_not_rotated_by_the_scenario_utm_zone():
    east_west_footprint = box(353_000, 6_184_000, 353_060, 6_184_012)
    template = gpd.GeoDataFrame(
        {"type_id": ["polyclinic"]}, geometry=[east_west_footprint], crs=KAZAN_UTM
    ).to_crs(LENINGRAD_UTM)

    footprint = ServiceGenerator._local_footprints(template)["polyclinic"]

    assert _long_axis_angle(template.geometry.iloc[0]) > 10.0
    assert _long_axis_angle(footprint) < 0.5


def test_every_shipped_template_has_a_service_type_id():
    templates = _shipped_templates()

    assert templates["service_type_id"].notna().all()
    assert templates["type_id"].is_unique


@pytest.mark.parametrize(
    "row", list(_shipped_templates().itertuples()), ids=lambda r: r.type_id
)
def test_shipped_template_is_aligned_and_fits_its_minimum_plot(row):
    border = GenParams().INNER_BORDER
    footprint = ServiceGenerator._local_footprints(
        gpd.GeoDataFrame({"type_id": [row.type_id]}, geometry=[row.geometry], crs=4326)
    )[row.type_id]
    minx, miny, maxx, maxy = footprint.bounds

    assert _long_axis_angle(footprint) < 1.0
    assert maxx - minx <= row.plot_length_min - 2 * border
    assert maxy - miny <= row.plot_width_min - 2 * border
