import aiohttp
from loguru import logger
from shapely.geometry import shape
import geopandas as gpd

from iduconfig import Config
from app.exceptions.http_exception_wrapper import http_exception
from app.api.api_error_handler import APIHandler
from app.dependencies import config

class UrbanDBAPI:
    def __init__(self, config: Config):
        self.config = config
        self.url = config.get("UrbanDB_API")
        self.session = None
        self.handler = None

    async def init(self):
        self.session = aiohttp.ClientSession()
        self.handler = APIHandler()

    async def close(self):
        if self.session:
            await self.handler.close_session(self.session)
            self.session = None
            logger.info("UrbanDBAPI session closed.")

    async def get_territories_for_buildings(self, scenario_id: int):
        api_url = (
            f"{self.url}v1/scenarios/{scenario_id}/functional_zones?year=2024&source=OSM"
        )
        logger.info(f"Fetching functional zones from API: {api_url}")

        json_data = await self.handler.request("GET", api_url, session=self.session)
        features = json_data.get("features", [])
        if not features:
            logger.warning(f"No functional zones found for scenario {scenario_id}")
            return gpd.GeoDataFrame(columns=["name", "functional_zone_id", "geometry"], crs=4326)

        records = []
        for feature in features:
            props = feature.get("properties", {})
            geom = shape(feature["geometry"])
            record = {
                "zone": props.get("functional_zone_type", {}).get("name"),
                "functional_zone_id": props.get("functional_zone_id"),
                "geometry": geom,
            }
            records.append(record)

        zones = gpd.GeoDataFrame(records, geometry="geometry", crs=4326)
        logger.info(f"Zones for scenario {scenario_id} collected: {len(zones)} items.")
        return zones


urban_db_api = UrbanDBAPI(config)