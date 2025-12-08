from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.dependencies import config, setup_logger
from app.routers.generation_routers import generation_router
from app.routers.logs_routers import logs_router

setup_logger(config)

app = FastAPI(title="GenBuilder API", version = "0.8")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(logs_router)

app.include_router(generation_router)
