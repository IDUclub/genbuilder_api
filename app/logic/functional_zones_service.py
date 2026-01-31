from typing import Dict, List

from app.exceptions.http_exception_wrapper import http_exception
from app.schema.dto import BlockFeatureCollection


class FunctionalZonesService:
    """
    Service responsible for fetching functional zones from Urban DB
    and preparing BlockFeatureCollection objects for Genbuilder.
    """

    def __init__(self, urban_db_api):
        self.urban_db_api = urban_db_api

    async def prepare_blocks_by_zones(
        self,
        scenario_id: int,
        year: int,
        source: str,
        token: str,
        functional_zone_types: List[str],
        zones: list,
    ) -> Dict[int, BlockFeatureCollection]:
        """
        Fetch functional zones, filter them by type, validate requested ids
        and build BlockFeatureCollection per functional zone.
        """

        response_json = await self.urban_db_api.get_scenario_functional_zones(
            scenario_id=scenario_id,
            source=source,
            year=year,
            token=token,
        )

        features = response_json.get("features", [])
        if not features:
            raise http_exception(
                404,
                f"No functional zones found for scenario {scenario_id}",
            )

        filtered_features: Dict[int, dict] = {}
        for feature in features:
            props = feature.get("properties", {})
            zone_type = (props.get("functional_zone_type") or {}).get("name")
            zone_id = props.get("functional_zone_id")

            if zone_type in functional_zone_types and zone_id is not None:
                filtered_features[int(zone_id)] = feature

        requested_ids = [zone.functional_zone_id for zone in zones]
        missing_ids = [
            zone_id for zone_id in requested_ids if zone_id not in filtered_features
        ]
        if missing_ids:
            raise http_exception(
                404,
                "Functional zones not found for provided ids",
                input_data={"missing_ids": missing_ids},
            )

        blocks_by_zone: Dict[int, BlockFeatureCollection] = {}

        for zone in zones:
            feature = filtered_features[zone.functional_zone_id]
            props = feature.get("properties", {})
            zone_type = (props.get("functional_zone_type") or {}).get("name")

            block_feature = {
                "type": "Feature",
                "properties": {
                    **props,
                    "block_id": props.get("functional_zone_id"),
                    "zone": zone_type,
                },
                "geometry": feature.get("geometry"),
            }

            blocks = BlockFeatureCollection.model_validate(
                {"type": "FeatureCollection", "features": [block_feature]}
            )

            blocks_by_zone[int(zone.functional_zone_id)] = blocks

        return blocks_by_zone
