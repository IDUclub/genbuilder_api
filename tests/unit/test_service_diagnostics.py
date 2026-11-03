import geopandas as gpd
from shapely.geometry import Point

from app.logic.service_generation import ServiceGenerator
from app.schema.dto import BuildingFeatureCollection


def test_service_summary_counts_fulfilled_and_unfulfilled_targets():
    limits = {
        "residential": {
            "Школа": 100.0,
            "Детский сад": 50.0,
            "Музей": 0.0,
        }
    }
    placed = gpd.GeoDataFrame(
        [
            {
                "zone": "residential",
                "service": "Школа",
                "capacity": 100.0,
                "geometry": Point(0, 0),
            },
            {
                "zone": "residential",
                "service": "Детский сад",
                "capacity": 20.0,
                "geometry": Point(1, 1),
            },
        ],
        geometry="geometry",
        crs=4326,
    )

    summary = ServiceGenerator.summarize_service_generation(limits, placed)

    assert summary == {
        "services_requested": 2,
        "services_placed": 1,
        "services_unplaced": 1,
        "service_buildings_placed": 2,
        "capacity_requested": 150.0,
        "capacity_placed": 120.0,
        "capacity_unplaced": 30.0,
        "unplaced_type_not_supported": 0,
        "unplaced_no_template": 0,
        "unplaced_demand_below_template": 0,
        "unplaced_no_space": 1,
        "unplaced_site_limit": 0,
        "unplaced_by_reason": {
            "type_not_supported": [],
            "no_template": [],
            "demand_below_template": [],
            "no_space": ["Детский сад"],
            "site_limit": [],
        },
    }


def test_response_model_keeps_service_diagnostics():
    payload = BuildingFeatureCollection.model_validate(
        {
            "type": "FeatureCollection",
            "features": [],
            "service_diagnostics": {
                "territory_id_provided": False,
                "territory_id": None,
                "normatives_found": 0,
                "services_requested": 0,
                "services_placed": 0,
                "services_unplaced": 0,
                "service_buildings_placed": 0,
                "capacity_requested": 0.0,
                "capacity_placed": 0.0,
                "capacity_unplaced": 0.0,
                "status": "territory_not_provided",
                "warning": "territory_id не передан; сервисы не генерировались",
            },
        }
    )

    assert payload.service_diagnostics is not None
    assert payload.service_diagnostics.status == "territory_not_provided"
    assert payload.service_diagnostics.unplaced_no_template == 0
    assert payload.service_diagnostics.unplaced_by_reason["no_template"] == []


def test_openapi_exposes_strict_territory_contract_and_diagnostics():
    from app.main import app

    schema = app.openapi()
    territory_request = schema["components"]["schemas"]["TerritoryRequest"]
    generation_response = schema["components"]["schemas"][
        "BuildingFeatureCollection"
    ]

    assert schema["info"]["version"] == "0.1.3"
    assert territory_request["additionalProperties"] is False
    assert "territory_id" in territory_request["properties"]
    assert "service_diagnostics" in generation_response["properties"]
    diagnostics = schema["components"]["schemas"]["ServiceGenerationDiagnostics"]
    assert "unplaced_type_not_supported" in diagnostics["properties"]
    assert "unplaced_no_template" in diagnostics["properties"]
    assert "unplaced_demand_below_template" in diagnostics["properties"]
    assert "unplaced_no_space" in diagnostics["properties"]
    assert "unplaced_site_limit" in diagnostics["properties"]
