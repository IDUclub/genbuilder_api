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


class ScenarioRequest(BaseModel):
    scenario_id: int = Field(
        ...,
        description="Scenario ID",
        json_schema_extra={"examples": [198]},
    )

    functional_zone_types: List[str] = Field(
        ...,
        description="List of target functional zone types",
        json_schema_extra={"examples": [["residential", "business"]]},
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


__all__ = [
    "BlockFeatureCollection",
    "TerritoryRequest",
    "PIPELINE_EXAMPLE",
    "ScenarioRequest",
]
