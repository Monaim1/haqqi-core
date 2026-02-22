from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI

from src.api.routes import router as api_router
from src.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    for name in ("httpx", "qdrant_client", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()


def run() -> None:
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
