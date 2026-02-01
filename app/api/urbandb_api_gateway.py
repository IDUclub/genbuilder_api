import aiohttp
import geopandas as gpd
import pandas as pd
from iduconfig import Config
from loguru import logger
from shapely.geometry import shape

from app.api.api_error_handler import APIHandler
from app.exceptions.http_exception_wrapper import http_exception


class UrbanDBAPI:
    def __init__(self, config: Config):
        self.config = config
        self.url = config.get("UrbanDB_API")
        self.handler = APIHandler()

    async def get_territories_for_buildings(self, scenario_id: int, year: int, source: str, token: str):
        api_url = f"{self.url.rstrip('/')}/api/v1/scenarios/{scenario_id}/functional_zones?year={year}&source={source}"
        logger.info(f"Fetching functional zones from API: {api_url}")
        headers = {'Authorization': f'Bearer {token}'}
        async with aiohttp.ClientSession() as session:
            json_data = await self.handler.request("GET", api_url, session=session, headers=headers, expect_json=True)

        features = json_data.get("features", [])
        if not features:
            raise http_exception(404, f"No functional zones found for scenario {scenario_id}")

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
        if not zones.empty:
            geoms = list(zones.geometry)
            to_drop: set[int] = set()

            for i in range(len(geoms)):
                if i in to_drop:
                    continue
                gi = geoms[i]
                if gi is None or gi.is_empty:
                    continue
                for j in range(i + 1, len(geoms)):
                    if j in to_drop:
                        continue
                    gj = geoms[j]
                    if gj is None or gj.is_empty:
                        continue
                    if gi.equals(gj):
                        to_drop.add(j)

            if to_drop:
                before = len(zones)
                zones = zones.drop(zones.index[list(to_drop)]).reset_index(drop=True)
                logger.info(f"Removed duplicated (equal) polygons: {before - len(zones)} items.")

        logger.info(f"Zones for scenario {scenario_id} collected: {len(zones)} items.")
        return zones
    
    async def get_territory_by_scenario(self, scenario_id: int, token: str):
        api_url = f"{self.url.rstrip('/')}/api/v1/scenarios/{scenario_id}"
        headers = {'Authorization': f'Bearer {token}'}
        logger.info(f"Fetching service normatives from API: {api_url}")
        async with aiohttp.ClientSession() as session:
            json_data = await self.handler.request("GET", api_url, session=session, headers=headers, expect_json=True)
        territory_id =  json_data.get("project", {}).get("region", {}).get("id", False)
        if not territory_id:
            raise http_exception(404, f"No territory id found for scenario {scenario_id}")
        logger.info(f"Territory id for scenario {scenario_id} collected.")
        return territory_id
    
    async def get_normatives_for_territory(self, territory_id: int, token: str):
        api_url = f"{self.url.rstrip('/')}/api/v1/territory/{territory_id}/normatives?last_only=true&include_child_territories=false&cities_only=false"
        logger.info(f"Fetching service normatives from API: {api_url}")
        headers = {'Authorization': f'Bearer {token}'}
        async with aiohttp.ClientSession() as session:
            json_data = await self.handler.request("GET", api_url, session=session, headers=headers, expect_json=True)
        service_normatives = []
        for service in json_data:
            service_data = service.get("service_type")
            service_id = service_data.get("id")
            service_name = service_data.get("name")
            service_capacity = service.get("services_capacity_per_1000_normative")
            record = {"service_id":service_id, "service_name":service_name, "service_capacity":service_capacity}
            service_normatives.append(record)
        service_normatives = pd.DataFrame(service_normatives)
        service_normatives.dropna(subset=['service_capacity'], inplace=True)
        logger.info(f"Normatives for territory {territory_id} collected.")
        return service_normatives

    async def get_scenario_functional_zones(self, scenario_id: int, source:str, year: int, token: str):
        api_url = f"{self.url.rstrip('/')}/api/v1/scenarios/{scenario_id}/functional_zones"
        logger.info(f"Fetching functional_zones from API: {api_url} for scenario {scenario_id}")
        headers = {'Authorization': f'Bearer {token}'}
        params = {
            "source": source,
            "year": year,
        }

        async with aiohttp.ClientSession() as session:
            response_json = await self.handler.request(
                method="GET",
                url=api_url,
                session=session,
                headers=headers,
                params=params,
                expect_json=True,
            )

        return response_json

    async def get_physical_objects(self, scenario_id: int, token: str, physical_object_type_id: int = None):
        api_url = f"{self.url.rstrip('/')}/api/v1/scenarios/{scenario_id}/physical_objects_with_geometry"
        logger.info(f"Fetching functional_zones from API: {api_url} for scenario {scenario_id}")
        headers = {'Authorization': f'Bearer {token}'}
        if physical_object_type_id:
            params = {
                "physical_object_type_id": physical_object_type_id,
            }
        else:
            params = None

        async with aiohttp.ClientSession() as session:
            response_json = await self.handler.request(
                method="GET",
                url=api_url,
                session=session,
                headers=headers,
                params=params,
                expect_json=True,
            )

        return response_json
