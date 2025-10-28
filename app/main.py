from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.dependencies import config, setup_logger, urban_db_api, genbuilder_inference_api
from app.routers.generation_routers import generation_router
from app.routers.logs_routers import logs_router

setup_logger(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await genbuilder_inference_api.init()
    app.state.genbuilder_inference = genbuilder_inference_api

    await urban_db_api.init()
    app.state.urban_db_api = urban_db_api

    yield

    await app.state.genbuilder_inference.close()
    await app.state.urban_db_api.close()


app = FastAPI(title="GenBuilder API", lifespan=lifespan)

app.include_router(logs_router)

app.include_router(generation_router)
