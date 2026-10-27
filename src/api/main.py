"""FastAPI application factory and endpoints."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.inference import ModelLoader
from src.api.schemas import HealthResponse, PredictionResponse
from src.config import Config
from src.utils.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

model_loader = ModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load()
    if model_loader.is_loaded():
        logger.info(f"Model loaded from {model_loader.model_path}")
    else:
        logger.warning("Model file not found, /predict will return 503")
    app.state.model_loader = model_loader
    yield


app = FastAPI(
    title="Industrial Defect Detection API",
    description="Detect surface defects on MVTec AD images.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Liveness/readiness probe used by Docker and Kubernetes."""
    loader = getattr(app.state, "model_loader", None) or model_loader
    if loader is None:
        return {"status": "ok", "model_loaded": False}
    try:
        il = getattr(loader, "is_loaded")
        if callable(il):
            model_loaded = bool(il())
        else:
            model_loaded = bool(il)
    except Exception:
        model_loaded = False
    if not model_loaded:
        try:
            if getattr(loader, "_session", None) is not None or getattr(loader, "session", None) is not None or getattr(loader, "_model", None) is not None:
                model_loaded = True
        except Exception:
            pass
    return {"status": "ok", "model_loaded": bool(model_loaded)}


@app.get("/config", tags=["system"])
def read_config():
    """Expose non-secret configuration and alert thresholds."""
    config = Config.load()
    return {"params": dict(config.params), "alerting": dict(config.alerting)}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Classify an uploaded image as ``good`` or ``defect``."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image")

    payload = await file.read()
    loader = getattr(app.state, "model_loader", None) or model_loader
    if loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    loaded = False
    try:
        il = getattr(loader, "is_loaded")
        if callable(il):
            loaded = bool(il())
        else:
            loaded = bool(il)
        if not loaded and getattr(loader, "_session", None) is not None:
            loaded = True
        if not loaded and getattr(loader, "session", None) is not None:
            loaded = True
        if not loaded and getattr(loader, "_model", None) is not None:
            loaded = True
    except Exception:
        loaded = False
    if not loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = loader.predict(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return result
