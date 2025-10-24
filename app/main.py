from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.dependencies import config, setup_logger
from app.routers.logs_routers import logs_router
from app.routers.generation_routers import generation_router
from app.api.genbuilder_gateway import genbuilder_inference
from app.api.urbandb_api_gateway import urban_db_api

setup_logger(config)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await genbuilder_inference.init()
    app.state.genbuilder_inference = genbuilder_inference

    await urban_db_api.init()
    app.state.urban_db_api = urban_db_api

    yield

    await app.state.genbuilder_inference.close()
    await app.state.urban_db_api.close()

app = FastAPI(title="GenBuilder API", lifespan=lifespan)

app.include_router(logs_router)

app.include_router(generation_router)