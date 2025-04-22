from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from io import BytesIO
import json
import numpy as np
import geopandas as gpd
from PIL import Image

from app.schema.dto import GenerateBuildingsRequest

from app.logic.generation import builder
from app.logic.vectorization import vectorizer
from app.logic.postprocessing import postprocessing
from app.logic.pipeline import genbuilder_pipe

generation_router = APIRouter()

@generation_router.post(
    "/generate_buildings",
    summary="Generate an inpainted image for each city block"
)
async def generate_buildings_route(
    request: GenerateBuildingsRequest
):
    """
    Accepts a GeoJSON FeatureCollection plus per‑zone prompts and inpainting settings,
    and returns a single PNG image with each zone inpainted.
    """
    gdf_zones = gpd.GeoDataFrame.from_features(request.input_zones)
    prompt_params = {
        zone_key: zp.dict()
        for zone_key, zp in request.prompt_dict.items()
    }
    img = await builder.generate_buildings(
        request.image_size,
        gdf_zones,
        prompt_params,
        request.num_steps,
        request.guidance_scale,
    )

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@generation_router.post(
    "/vectorize_buildings",
    summary="Vectorize buildings from generated image and input city blocks"
)
async def vectorize_buildings_route(
    geojson_file: UploadFile = File(..., media_type="application/geo+json"),
    image_file: UploadFile = File(..., media_type="image/png")
):
    """
    Принимает два файла: GeoJSON с зонами и изображение.
    Возвращает GeoJSON с векторизованными зданиями.
    """
    content = await geojson_file.read()
    stream = BytesIO(content)
    gdf_zones = gpd.read_file(stream)
    img_bytes = await image_file.read()
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    np_img = np.array(pil_img)
    gdf_buildings = await vectorizer.vectorize_buildings(gdf_zones, np_img)
    return json.loads(gdf_buildings.to_json())

@generation_router.post(
    "/process_buildings",
    summary="Post-process vectorized buildings",
)
async def process_buildings_route(
    geojson_file: UploadFile = File(..., media_type="application/geo+json")
):
    content = await geojson_file.read()
    stream = BytesIO(content)
    gdf_buildings = gpd.read_file(stream)
    gdf_processed = await postprocessing.process_buildings(gdf_buildings)
    return json.loads(gdf_processed.to_json())


@generation_router.post(
    "/pipeline",
    summary="Run full Genbuilder pipeline end-to-end",
)
async def pipeline_route(
    request: GenerateBuildingsRequest
):
    gdf_zones = gpd.GeoDataFrame.from_features(request.input_zones)
    gdf_result = await genbuilder_pipe.run(
        request.image_size,
        gdf_zones,
        request.prompt_dict,
        request.num_steps,
        request.guidance_scale
    )
    geojson = json.loads(gdf_result.to_json())
    return geojson
