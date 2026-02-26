from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from app.dependencies import config, setup_logger
from app.routers.generation_routers import generation_router
from app.routers.logs_routers import logs_router
from app.init_entities import start_prometheus, shutdown_prometheus
from app.observability.metrics import setup_metrics
from app.common.middlewares.prometheus_handler import ObservabilityMiddleware
from app.common.middlewares.exception_handler import ExceptionHandlerMiddleware

setup_logger(config)
metrics = setup_metrics()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_prometheus()
    try:
        yield
    finally:
        await shutdown_prometheus()
app = FastAPI(title="GenBuilder API", version = "0.8")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(ExceptionHandlerMiddleware, metrics=metrics)
app.add_middleware(ObservabilityMiddleware, metrics=metrics)

@app.get("/", include_in_schema=False)
async def read_root():
    return RedirectResponse("/docs")

app.include_router(logs_router)

app.include_router(generation_router)
