from loguru import logger
import pandas as pd
import aiohttp
import os
import ssl
from io import BytesIO
from PIL import Image
from iduconfig import Config
from app.api.api_error_handler import APIHandler
from app.dependencies import config

class GenbuilderInference:
    def __init__(self, config: Config):
        self.url = config.get("GPU_URL")
        self.client_cert = config.get("GPU_CLIENT_CERTIFICATE")
        self.client_key = config.get("GPU_CLIENT_KEY")
        self.ca_cert = config.get("GPU_CERTIFICATE")
        self.session = None
        self.handler = None
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

    @staticmethod
    def pil_to_buffer(img: Image.Image, fmt="PNG"):
        buf = BytesIO()
        img.save(buf, format=fmt)
        buf.seek(0)
        return buf
    
    async def generate(self,
                    prompt: str,
                    image: Image.Image,
                    mask: Image.Image,
                    negative_prompt: str = "",
                    num_steps: int = 20,
                    guidance_scale: float = 7.5
                    ):

        api_url = self.url
        logger.info(f"Sending inpainting request to API: {api_url}")

        form = aiohttp.FormData()
        form.add_field("prompt", prompt)
        form.add_field("negative_prompt", negative_prompt)
        form.add_field("num_steps", str(num_steps))
        form.add_field("guidance_scale", str(guidance_scale))

        img_buf = genbuilder_inference.pil_to_buffer(image, fmt="PNG")
        form.add_field(
            name="image",
            value=img_buf,
            filename="base.png",
            content_type="image/png",
        )

        mask_buf = genbuilder_inference.pil_to_buffer(mask, fmt="PNG")
        form.add_field(
            name="mask",
            value=mask_buf,
            filename="mask.png",
            content_type="image/png",
        )

        response = await self.handler.request(
            method="POST",
            url=api_url,
            session=self.session,
            data=form,
            expect_json=False
        )
        logger.info("Inpainting response received.")
        return Image.open(BytesIO(response))

genbuilder_inference = GenbuilderInference(config)