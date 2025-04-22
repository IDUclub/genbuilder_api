from loguru import logger
from shapely.geometry import MultiPolygon
import numpy as np
import geopandas as gpd
from PIL import Image, ImageDraw
from app.api.genbuilder_gateway import genbuilder_inference
from app.schema.dto import ZonePrompt
from typing import Dict, Any


class Builder:
    async def transform_polygon_to_image_coords(
        self,
        polygon,
        minx,
        miny,
        maxx,
        maxy,
        width=512,
        height=512
    ):
        """
        Преобразует координаты полигона (в метрах) в координаты изображения размером width x height.
        """
        logger.debug(
            f"transform_polygon_to_image_coords called with bounds=({minx}, {miny}, {maxx}, {maxy}) "
            f"and image size=({width}, {height})"
        )
        if isinstance(polygon, MultiPolygon):
            polygon = list(polygon.geoms)[0]

        coords = []
        range_x = maxx - minx
        range_y = maxy - miny

        for x, y in polygon.exterior.coords:
            new_x = (x - minx) * (width / range_x)
            new_y = (maxy - y) * (height / range_y)
            coords.append((new_x, new_y))

        logger.debug(f"Computed {len(coords)} image coordinates for polygon")
        return coords

    async def create_polygon_mask(self, image_size, polygon_coords):
        """
        Создает бинарную маску (PIL.Image в режиме "L") с заполненной многоугольной областью.
        """
        logger.debug(
            f"create_polygon_mask called with image_size={image_size} "
            f"and polygon_coords length={len(polygon_coords)}"
        )
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygon_coords, fill=255)
        logger.debug("Polygon mask created")
        return mask

    async def create_prompt_zones(
        self,
        input_zones: gpd.GeoDataFrame,
        prompt_dict: dict,
        image_size: int
    ):
        """
        Преобразует GeoDataFrame кварталов в словарь зон с их пиксельными координатами 
        и соответствующими prompt/negative_prompt.
        """
        logger.info("Starting create_prompt_zones")
        gdf = input_zones.copy()
        minx, miny, maxx, maxy = gdf.total_bounds
        logger.debug(f"GeoDataFrame bounds: minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}")

        zones: Dict[str, Any] = {}
        for idx, row in gdf.iterrows():
            zone_id = row.get("zone_id", idx)
            key = str(zone_id)

            coords = await self.transform_polygon_to_image_coords(
                row["geometry"],
                minx, miny, maxx, maxy,
                width=image_size,
                height=image_size
            )
            attrs = [
                row.get("func_zone", ""),
                row.get("storey_category", ""),
                row.get("construction_period", ""),
                row.get("population", ""),
                row.get("objects", "")
            ]
            scale = row.get("scale", None)
            if scale not in (None, "", np.nan):
                attrs.append(f"scale_{scale}")

            parts = [str(a) for a in attrs if a not in (None, "", np.nan)]
            default_prompt = "overhead vector_map" + (
                ", " + ", ".join(parts) if parts else ""
            )
            default_negative = ""

            custom = None
            if isinstance(prompt_dict, dict):
                custom = prompt_dict.get(key) or prompt_dict.get(zone_id)

            prompt_text = (
                custom.prompt if isinstance(custom, ZonePrompt) and custom.prompt else default_prompt
            )
            negative_text = (
                custom.negative_prompt if isinstance(custom, ZonePrompt) and custom.negative_prompt is not None else default_negative
            )

            logger.debug(
                f"Zone {key}: prompt='{prompt_text}', "
                f"negative_prompt='{negative_text}', coords_count={len(coords)}"
            )

            zones[key] = {
                "coords": coords,
                "prompt": prompt_text,
                "negative_prompt": negative_text,
            }

        logger.info(f"Created {len(zones)} prompt zones")
        return zones

    async def generate_buildings(
        self,
        image_size,
        input_zones,
        prompt_dict,
        num_steps,
        guidance_scale
    ):
        logger.info("Starting generate_buildings")
        width, height = image_size, image_size
        base_image = Image.new("RGB", (width, height), "white")

        zones = await self.create_prompt_zones(input_zones, prompt_dict or {}, image_size)
        logger.info(f"Number of zones to process: {len(zones)}")

        for zone_key, data in zones.items():
            logger.info(f"Generating buildings for zone {zone_key}")
            coords = data["coords"]
            prompt = data["prompt"]
            negative_prompt = data["negative_prompt"]

            mask = await self.create_polygon_mask((width, height), coords)
            inpainted_image = await genbuilder_inference.generate(
                prompt,
                base_image,
                mask,
                negative_prompt,
                num_steps,
                guidance_scale
            )

            # Composite this zone onto base_image
            base_np = np.array(base_image)
            inpainted_np = np.array(inpainted_image.resize((width, height)))
            mask_np = np.array(mask) / 255.0

            composite_np = (
                inpainted_np * mask_np[..., None]
                + base_np * (1 - mask_np[..., None])
            ).astype(np.uint8)

            base_image = Image.fromarray(composite_np)
            logger.info(f"Zone {zone_key} composited")

        # After all zones have been rendered and composited, return the final image
        logger.info("All zones processed, returning final composite")
        return base_image



builder = Builder()
