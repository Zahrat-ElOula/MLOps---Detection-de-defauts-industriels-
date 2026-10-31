"""FastAPI application factory and endpoints."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.inference import ModelLoader
from src.api.schemas import HealthResponse, PredictionResponse
from src.config import Config, PROJECT_ROOT
from src.models.yolo_detector import YOLODetector
from src.utils.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Charger le classifieur
    loader = ModelLoader()
    loader.load()
    app.state.model_loader = loader
    if loader.is_loaded():
        logger.info(f"Model loaded from {loader.model_path}")
    else:
        logger.warning("Classifier not loaded")

    # Charger YOLO
    yolo_path = PROJECT_ROOT / "model_registry" / "yolo_best.onnx"
    if yolo_path.exists():
        try:
            detector = YOLODetector(yolo_path)
            app.state.yolo_detector = detector
            logger.info(f"YOLO detector loaded from {yolo_path}")
        except Exception as exc:
            logger.warning(f"Failed to load YOLO: {exc}")
            app.state.yolo_detector = None
    else:
        logger.warning(f"YOLO file not found: {yolo_path}")
        app.state.yolo_detector = None

    logger.info(f"YOLO loaded: {getattr(app.state, 'yolo_detector', None) is not None}")
    yield


app = FastAPI(
    title="Industrial Defect Detection API",
    description="Detect surface defects on MVTec AD images.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    loader = getattr(app.state, "model_loader", None)
    return {
        "status": "ok",
        "model_loaded": loader.is_loaded() if loader else False,
    }


@app.get("/config", tags=["system"])
def read_config():
    config = Config.load()
    return {"params": dict(config.params), "alerting": dict(config.alerting)}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image")

    payload = await file.read()
    loader = getattr(app.state, "model_loader", None)
    if loader is None or not loader.is_loaded():
        raise HTTPException(status_code=503, detail="Classifier not loaded")

    try:
        result = loader.predict(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    if result.get("label") == "defect" and float(result.get("confidence", 1.0)) < 0.5:
        raise HTTPException(status_code=422, detail="Catégorie non reconnue ou incertaine.")

    detector = getattr(app.state, "yolo_detector", None)
    if result.get("label") == "defect" and detector is not None:
        try:
            result["boxes"] = detector.detect(payload)
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
            result["boxes"] = []
    else:
        result["boxes"] = []

    logger.info(f"Returning result with {len(result.get('boxes', []))} boxes")
    return result


@app.post("/feedback", tags=["feedback"])
async def feedback(file: UploadFile = File(...), label: str = Form(...)):
    """Store a user-corrected label next to the image for continuous training."""
    feedback_dir = Path("data/feedback")
    feedback_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = feedback_dir / f"{timestamp}_{file.filename or 'image'}"

    payload = await file.read()
    image_path.write_bytes(payload)

    label_path = image_path.with_suffix(".txt")
    label_path.write_text(label, encoding="utf-8")

    logger.info(f"Feedback saved: {image_path} (label={label})")
    return {"status": "success", "message": "Feedback enregistré dans data/feedback/"}