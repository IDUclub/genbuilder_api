from typing import Any, Dict, Optional, List, Literal
from pydantic import BaseModel, Field, model_validator
from geojson_pydantic import Feature, FeatureCollection, MultiPolygon, Polygon
from app.schema._blocks_example import blocks

class ZonePrompt(BaseModel):
    prompt: Optional[str] = Field(
        default=None,
        example="best quality, white background",
        description="Positive prompt for this zone"
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        example="cropped buildings, bad geometry, complex geometry",
        description="Negative prompt for this zone"
    )

class ZoneProperties(BaseModel):
    zone_id: int = Field(..., example=0, description="Уникальный идентификатор зоны")
    func_zone: str = Field(..., examples=["residential", "industrial", "transport"], description="Функциональная зона")
    storey_category: Optional[Literal["low_rise", "medium_rise", "high_rise"]] = Field(default=None, example="high_rise", description="Категория этажности")
    construction_period: Optional[Literal["historic", "soviet", "modern"]] = Field(default=None, example="soviet", description="Период строительства")
    population: Optional[Literal["density_1", "density_2", "density_3", "density_4", "density_5"]] = Field(default=None, example="density_1", description="Численность населения, выше значение - больше")
    objects: Optional[str] = Field(default=None, examples=["Park", "School", "School, Hospital"], description="Описание объектов в зоне")

class ZoneFeature(Feature[Polygon | MultiPolygon, ZoneProperties]):
    pass

class ZoneFeatureCollection(FeatureCollection[ZoneFeature]):
    pass

class GenerateBuildingsRequest(BaseModel):
    input_zones: ZoneFeatureCollection = Field(
        default=None,
        examples=[blocks],
        description="GeoJSON FeatureCollection of zones"
    )
    image_size: int = Field(
        default=1024,
        example=[512, 1024],
        description="Width/height of the square output image"
    )
    prompt_dict: Optional[Dict[int, ZonePrompt]] = Field(
        default=None,
        example={
            63:{
            "prompt": "best quality, white background",
            "negative_prompt": "cropped buildings, bad geometry, complex geometry"
            }, 
            65:{
            "prompt": "best quality, white background",
            "negative_prompt": "cropped buildings, bad geometry, complex geometry"
            }, 
            68:{
            "prompt": "best quality, white background",
            "negative_prompt": "cropped buildings, bad geometry, complex geometry"
            }},
        description="Mapping from zone keys to their prompt settings"
    )
    num_steps: int = Field(
        default=40,
        description="Number of diffusion/inpainting steps"
    )
    guidance_scale: float = Field(
        default=8.5,
        description="Classifier-free guidance scale")
    
    @model_validator(mode="after")
    def ensure_no_extra_prompt_keys(self):
        expected = {
            feature.properties.zone_id
            for feature in self.input_zones.features
        }
        prompt_ids = set(self.prompt_dict.keys())
        extra = prompt_ids - expected
        if extra:
            raise ValueError(
                f"Unexpected prompt_dict keys not in input_zones: {sorted(extra)}"
            )
        return self


class VectorizeBuildingsRequest(BaseModel):
    input_zones: ZoneFeatureCollection = Field(
        default=None,
        examples=[blocks],
        description="GeoJSON FeatureCollection of zones"
    )


class ProcessBuildingsRequest(BaseModel):
    buildings_geojson: Dict[str, Any]
