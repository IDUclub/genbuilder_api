from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Annotated

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
            raise ValueError("`zone` обязательно в feature.properties и не может быть пустым.")
        return s


class BlockFeature(Feature[Polygon, BlockProperties]):
    """GeoJSON Feature representing an urban block (Polygon)."""

    pass


class BlockFeatureCollection(FeatureCollection[BlockFeature]):
    """GeoJSON FeatureCollection of block polygons."""

    pass


class ScenarioBody(BaseModel):
    targets_by_zone: Optional[Dict[str, Dict[str, float]]] = Field(
        default={
            "la_target": {
                "residential": 20000,
                "business": 10000,
                "industrial": 0,
                "transport": 0,
                "special": 0,
                "agriculture": 5000,
                "recreation": 0
            },
            "floors_avg": {
                "residential": 12,
                "business": 7,
                "industrial": 5,
                "transport": 1,
                "special": 3,
                "agriculture": 3,
                "recreation": 1
            }
        },
        description="Sum of living area and mean of floors count for functional zone types",
        json_schema_extra={
            "examples": [
                {
                    "la_target": {"residential": 20000, "business": 6000, "industrial": 0},
                    "floors_avg": {"residential": 12, "business": 7, "industrial": 5},
                }
            ]
        }
    )

    params: Optional[Dict[str, Any]] = Field(
        default={
            "knn": 8,
            "e_thr": 0.8,
            "il_thr": 0.5,
            "sv1_thr": 0.5,
            "slots": 5000
        },
        description="Inference hyperparameters",
        json_schema_extra={
            "examples": [
                {
                    "knn": 8,
                    "e_thr": 0.8,
                    "il_thr": 0.5,
                    "sv1_thr": 0.5,
                    "slots": 5000
                }
            ]
        }
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

    targets_by_zone: Dict[str, Dict[str, float]] = Field(
        default=None,
        description=(
            "Target indicators by zone, e.g. "
            "{'residential': {'la': 20000, 'floors_avg': 12}}"
        ),
        json_schema_extra={
            "examples": [
                {
                    "la_target": {
                        "residential": 20000,
                        "business": 6000,
                        "industrial": 0,
                    },
                    "floors_avg": {
                        "residential": 12,
                        "business": 7,
                        "industrial": 5,
                    },
                }
            ]
        },
    )

    params: Dict[str, Any] = Field(
        default=None,
        description="Inference hyperparameters (as a dictionary)",
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

    @model_validator(mode="after")
    def _ensure_polygons(self):
        for feature in self.blocks.features:
            if getattr(feature, "geometry", None) is None or feature.geometry.type != "Polygon":
                raise ValueError(
                    "Each Feature in `blocks` must have geometry of type Polygon"
                )
        return self


PIPELINE_EXAMPLE = {
    "blocks": EXAMPLE_BLOCKS,
    "targets_by_zone": {
        "residential": {"la": 20000, "floors_avg": 12},
        "business": {"la": 6000, "floors_avg": 7},
        "industrial": {"la": 0, "floors_avg": 5},
    },
    "params": {
        "infer_knn": 8,
        "infer_e_thr": 0.8,
        "infer_il_thr": 0.5,
        "infer_sv1_thr": 0.5,
        "infer_slots": 5000,
    },
}


class BuildingFeatureCollection(BaseModel):
    type: str = Field(...)
    features: list = Field(..., description="List of building features, service optional (else - null)")

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
                                "floors_count": 7,
                                "service": [{"school": 350}]
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [
                                        [31.03664649794203, 59.920231764344585],
                                        [31.04196685516627, 59.920231764344585],
                                        [31.04196685516627, 59.922346520462526],
                                        [31.03664649794203, 59.922346520462526],
                                        [31.03664649794203, 59.920231764344585]
                                    ]
                                ]
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
    "PIPELINE_EXAMPLE",
    "ScenarioRequest",
    "BuildingFeatureCollection"
]
