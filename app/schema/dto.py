from __future__ import annotations

from typing import Any, Dict, Optional, Union, Annotated

from geojson_pydantic import Feature, FeatureCollection, Polygon
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

from app.schema._blocks_example import blocks as EXAMPLE_BLOCKS


class BlockProperties(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
    )

    block_id: Union[int, str] | None = Field(
        default=None,
        description="Block identifier"
    )

    zone: Annotated[str, Field(description="Zone label (required)")]

    @field_validator("zone")
    @classmethod
    def _zone_required_nonempty(cls, v: str) -> str:
        s = str(v).strip()
        if not s or s.lower() in {"none", "nan"}:
            raise ValueError("feature.properties should have 'zone'.")
        return s


class BlockFeature(Feature[Polygon, BlockProperties]):
    """GeoJSON Feature representing an urban block (Polygon)."""

    pass


class BlockFeatureCollection(FeatureCollection[BlockFeature]):
    """GeoJSON FeatureCollection of block polygons."""

    pass


class ScenarioBody(BaseModel):
    targets_by_zone: Optional[Dict[str, Dict[str, Any]]] = Field(
        default={
            "residents": {
                "residential": 1100,
                "business": 0,
                "unknown": 0,
            },
            "coverage_area": {
                "business": 10000,
                "unknown": 10000,
                "industrial": 20000,
                "transport": 10000,
                "special": 10000,
            },
            "floors_avg": {
                "business": 7,
                "unknown": 5,
                "industrial": 5,
                "transport": 1,
                "special": 3,
            },
            "density_scenario": {
                "residential": "min",
                "business": "min",
                "unknown": "min",
            },
            "default_floor_group": {
                "residential": "medium",
                "business": "high",
                "unknown": "high",
            },
        },
        description=(
            "Spatial and target parameters for functional zone types.\n"
            "- coverage_area: non-res functional area targets (m²)\n"
            "- floors_avg: mean floors for each zone\n"
            "- density_scenario: FAR scenario for residential component\n"
            "- default_floor_group: default floors_group for residential component"
        ),
        json_schema_extra={
            "examples": [
                {
                "residents": {
                    "residential": 1100,
                    "business": 0,
                    "unknown": 0,
                },
                "coverage_area": {
                    "business": 10000,
                    "unknown": 10000,
                    "industrial": 20000,
                    "transport": 10000,
                    "special": 10000,
                },
                "floors_avg": {
                    "business": 7,
                    "unknown": 5,
                    "industrial": 5,
                    "transport": 1,
                    "special": 3,
                },
                "density_scenario": {
                    "residential": "min",
                    "business": "min",
                    "unknown": "min",
                },
                "default_floor_group": {
                    "residential": "medium",
                    "business": "high",
                    "unknown": "high",
                },
            }
            ]
        },
    )
    generation_parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Generation parameters, override base ones",
        json_schema_extra={"examples": [{"rectangle_finder_step": 5}]},
    )


class TerritoryRequest(BaseModel):
    blocks: BlockFeatureCollection = Field(
        ...,
        description=(
            "GeoJSON FeatureCollection of blocks. "
            "Each Feature must include `properties.zone`."
        ),
        json_schema_extra={"examples": [EXAMPLE_BLOCKS]},
    )

    targets_by_zone: Optional[Dict[str, Dict[str, Any]]] = Field(
        default={
            "residents": {
                "residential": 1100,
                "business": 0,
                "industrial": 0,
                "transport": 0,
                "special": 0,
                "agriculture": 0,
                "recreation": 0,
            },
            "coverage_area": {
                "business": 10000,
                "industrial": 20000,
                "transport": 10000,
                "special": 10000,
                "agriculture": 0,
                "recreation": 0,
            },
            "floors_avg": {
                "residential": 12,
                "business": 7,
                "industrial": 5,
                "transport": 1,
                "special": 3,
                "agriculture": 3,
                "recreation": 1,
            },
            "density_scenario": {
                "residential": "min",
                "business": "min",
                "unknown": "min",
            },
            "default_floor_group": {
                "residential": "medium",
                "business": "high",
                "unknown": "high",
            },
        },
        description=(
            "Per-zone targets for generation:\n"
            "- coverage_area: non-res functional area (m²)\n"
            "- floors_avg: mean floors per zone\n"
            "- density_scenario: FAR scenario for residential component\n"
            "- default_floor_group: default floors_group for residential component"
        ),
        json_schema_extra={
            "examples": [
                {
                    "residents": {
                        "residential": 1100,
                        "business": 0,
                        "industrial": 0,
                    },
                    "coverage_area": {
                        "business": 6000,
                        "industrial": 20000,
                    },
                    "floors_avg": {
                        "residential": 12,
                        "business": 7,
                        "industrial": 5,
                    },
                    "density_scenario": {
                        "residential": "min",
                        "business": "mean",
                    },
                    "default_floor_group": {
                        "residential": "medium",
                        "business": "high",
                    },
                }
            ]
        },
    )

    params: Optional[Dict[str, Any]] = Field(
        default={
            "knn": 8,
            "e_thr": 0.8,
            "il_thr": 0.5,
            "sv1_thr": 0.5,
            "slots": 5000,
        },
        description="Inference hyperparameters (kept for backward compatibility)",
        json_schema_extra={
            "examples": [
                {
                    "knn": 8,
                    "e_thr": 0.8,
                    "il_thr": 0.5,
                    "sv1_thr": 0.5,
                    "slots": 5000,
                }
            ]
        },
    )

    generation_parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Generation parameters, override base ones",
        json_schema_extra={"examples": [{"cell_size_m": 10}]},
    )

    @model_validator(mode="after")
    def _ensure_polygons(self):
        for feature in self.blocks.features:
            if getattr(feature, "geometry", None) is None or feature.geometry.type != "Polygon":
                raise ValueError(
                    "Each Feature in `blocks` must have geometry of type Polygon"
                )
        return self


class BuildingFeatureCollection(BaseModel):
    type: str = Field(...)
    features: list = Field(
        ...,
        description=(
            "List of building features. "
            "`properties` should at least contain "
            "`floors_count`, `living_area`, `functional_area`, `building_area`, "
            "`zone`; `service` is optional (else - empty list or null)."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "id": "0",
                            "type": "Feature",
                            "properties": {
                                "living_area": 10636.33,
                                "functional_area": 0.0,
                                "building_area": 74453.31,
                                "floors_count": 7,
                                "service": [{"Школа": 350}],
                                "zone": "residential",
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [31.03664649794203, 59.920231764344585],
                                        [31.04196685516627, 59.920231764344585],
                                        [31.04196685516627, 59.922346520462526],
                                        [31.03664649794203, 59.922346520462526],
                                        [31.03664649794203, 59.920231764344585],
                                    ]
                                ],
                            },
                        }
                    ],
                }
            ]
        }
    }


__all__ = [
    "BlockFeatureCollection",
    "TerritoryRequest",
    # "PIPELINE_EXAMPLE",
    # "ScenarioRequest",
    "BuildingFeatureCollection",
]
