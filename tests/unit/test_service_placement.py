import asyncio

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from app.logic.generation_params import GenParams, ParamsProvider
from app.logic.service_generation import ServiceGenerator


CRS = 32636


def _generator(**overrides) -> ServiceGenerator:
    params = GenParams(
        seed=42,
        max_service_attempts=500,
        max_sites_per_service_per_block=10,
    ).patched(overrides)
    return ServiceGenerator(ParamsProvider(params))


def _blocks(size: float = 200.0) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [{"zone": "residential", "geometry": box(0, 0, size, size)}],
        geometry="geometry",
        crs=CRS,
    )


def _plots(size: float = 200.0) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [{"src_index": 0, "geometry": box(0, 0, size, size)}],
        geometry="geometry",
        crs=CRS,
    )


def _residential_buildings() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "src_index": 0,
                "living_area": 18_000.0,
                "geometry": box(5, 5, 15, 15),
            }
        ],
        geometry="geometry",
        crs=CRS,
    )


def _normatives(service_name: str, capacity: float) -> pd.DataFrame:
    return pd.DataFrame(
        [{"service_name": service_name, "service_capacity": capacity}]
    )


def _projects(
    service_name: str,
    *,
    capacity: float = 100.0,
    plot_size: float = 30.0,
) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "service": service_name,
                "type_id": f"{service_name}-100",
                "capacity": capacity,
                "floors_count": 1,
                "plot_length_min": plot_size,
                "plot_length_max": plot_size,
                "plot_width_min": plot_size,
                "plot_width_max": plot_size,
                "address": None,
                "osm_type": "way",
                "osm_id": 1,
                "osm_url": None,
                "geometry": box(-4, -4, 4, 4),
            }
        ],
        geometry="geometry",
        crs=CRS,
    )


def _generate(
    generator: ServiceGenerator,
    *,
    service_name: str = "Школа",
    normative_capacity: float = 100.0,
    project_service_name: str = "Школа",
    project_capacity: float = 100.0,
    plot_size: float = 30.0,
    block_size: float = 200.0,
) -> gpd.GeoDataFrame:
    projects = _projects(
        project_service_name,
        capacity=project_capacity,
        plot_size=plot_size,
    )
    generator.load_service_projects = lambda: projects
    return asyncio.run(
        generator.generate_services(
            _blocks(block_size),
            _plots(block_size),
            _residential_buildings(),
            _normatives(service_name, normative_capacity),
            CRS,
        )
    )


def test_service_placement_uses_building_footprints_not_residential_plots():
    residential_plot = _plots().geometry.iloc[0]
    result = _generate(_generator())

    assert len(result) == 1
    assert result.geometry.iloc[0].within(residential_plot)
    assert not result.geometry.iloc[0].intersects(
        _residential_buildings().geometry.iloc[0]
    )
    assert result.attrs["service_diagnostics"]["services_placed"] == 1


def test_diagnostics_distinguish_missing_template():
    result = _generate(
        _generator(),
        service_name="Библиотека",
        project_service_name="Школа",
    )

    diagnostics = result.attrs["service_diagnostics"]
    assert result.empty
    assert diagnostics["unplaced_no_template"] == 1
    assert diagnostics["unplaced_no_space"] == 0
    assert diagnostics["unplaced_site_limit"] == 0
    assert diagnostics["unplaced_by_reason"]["no_template"] == ["Библиотека"]


def test_diagnostics_distinguish_insufficient_space():
    result = _generate(
        _generator(),
        plot_size=80.0,
        block_size=50.0,
    )

    diagnostics = result.attrs["service_diagnostics"]
    assert result.empty
    assert diagnostics["unplaced_no_template"] == 0
    assert diagnostics["unplaced_no_space"] == 1
    assert diagnostics["unplaced_site_limit"] == 0
    assert diagnostics["unplaced_by_reason"]["no_space"] == ["Школа"]


def test_diagnostics_distinguish_per_block_site_limit():
    result = _generate(
        _generator(max_sites_per_service_per_block=1),
        normative_capacity=300.0,
        project_capacity=100.0,
        block_size=300.0,
    )

    diagnostics = result.attrs["service_diagnostics"]
    assert len(result) == 1
    assert diagnostics["services_unplaced"] == 1
    assert diagnostics["unplaced_no_template"] == 0
    assert diagnostics["unplaced_no_space"] == 0
    assert diagnostics["unplaced_site_limit"] == 1
    assert diagnostics["unplaced_by_reason"]["site_limit"] == ["Школа"]
