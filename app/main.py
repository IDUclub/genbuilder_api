from fastapi import FastAPI

from app.dependencies import config, setup_logger
from app.routers.generation_routers import generation_router
from app.routers.logs_routers import logs_router

setup_logger(config)

app = FastAPI(title="GenBuilder API")

app.include_router(logs_router)

app.include_router(generation_router)
