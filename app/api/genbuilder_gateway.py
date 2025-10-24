from __future__ import annotations

import ssl
from typing import Any, Dict, Optional

import aiohttp
from iduconfig import Config
from loguru import logger

from app.api.api_error_handler import APIHandler
from app.dependencies import config


class GenbuilderInference:
    def __init__(self, config: Config):
        self.url = config.get("GPU_URL")
        self.client_cert = config.get("GPU_CLIENT_CERTIFICATE")
        self.client_key = config.get("GPU_CLIENT_KEY")
        self.ca_cert = config.get("GPU_CERTIFICATE")
        self.session: Optional[aiohttp.ClientSession] = None
        self.handler: Optional[APIHandler] = None
        self.config = config

    async def init(self):
        ssl_ctx = ssl.create_default_context(cafile=self.ca_cert)
        ssl_ctx.load_cert_chain(certfile=self.client_cert, keyfile=self.client_key)
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        self.session = aiohttp.ClientSession(connector=connector)
        self.handler = APIHandler()

    async def close(self):
        if self.session:
            await self.handler.close_session(self.session)
            self.session = None
            logger.info("GenbuilderInference session closed.")

    async def generate_centroids(
        self,
        *,
        feature: Dict[str, Any],
        zone_label: str,
        infer_params: Optional[Dict[str, Any]] = None,
        la_target: float,
        floors_avg: float,
    ) -> Dict[str, Any]:
        if self.session is None or self.handler is None:
            await self.init()

        endpoint = self.url.rstrip("/") + "/centroids"

        if infer_params is None:
            infer_params = {
                "slots": 1,
                "knn": 1,
                "e_thr": 0.0,
                "il_thr": 0.0,
                "sv1_thr": 0.0,
            }

        payload = {
            "zone_label": zone_label,
            "feature": feature,
            "infer_params": infer_params,
            "la_target": la_target,
            "floors_avg": floors_avg,
        }

        response = await self.handler.request(
            method="POST",
            url=endpoint,
            session=self.session,
            json=payload,
            expect_json=True,
        )
        return response

    @staticmethod
    def to_feature_collection(resp: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "FeatureCollection", "features": resp.get("features", [])}


genbuilder_inference = GenbuilderInference(config)
