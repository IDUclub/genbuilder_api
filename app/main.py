from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from app.dependencies import config, setup_logger
from app.routers.generation_routers import generation_router
from app.routers.logs_routers import logs_router

setup_logger(config)

app = FastAPI(title="GenBuilder API", version = "0.1.1")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def read_root():
    return RedirectResponse("/docs")

app.include_router(logs_router)

app.include_router(generation_router)
