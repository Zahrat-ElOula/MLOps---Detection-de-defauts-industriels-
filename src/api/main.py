"""FastAPI application factory and endpoints."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.github_sync import GitHubSync
from src.api.inference import ModelLoader, load_class_names
from src.api.schemas import HealthResponse, PredictionResponse
from src.config import Config, PROJECT_ROOT
from src.models.yolo_detector import YOLODetector
from src.utils.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"


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

    # Synchronisation GitHub : reprend les feedbacks en attente au démarrage,
    # puis réessaie périodiquement. Une erreur GitHub ne bloque pas l'API.
    github_sync: GitHubSync | None = None
    try:
        github_sync = GitHubSync.from_env(feedback_dir=FEEDBACK_DIR)
        app.state.github_sync = github_sync
        await github_sync.start()
    except Exception as exc:
        logger.warning(f"GitHub feedback sync unavailable: {type(exc).__name__}")
        app.state.github_sync = None

    logger.info(f"YOLO loaded: {getattr(app.state, 'yolo_detector', None) is not None}")
    try:
        yield
    finally:
        if github_sync is not None:
            await github_sync.stop()
            await github_sync.aclose()


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
    sync = getattr(app.state, "github_sync", None)
    if sync is None:
        github_sync = {
            "status": "disabled",
            "pending_feedback": 0,
            "last_success": None,
            "last_error": None,
        }
    else:
        try:
            github_sync = sync.status()
        except Exception as exc:
            logger.warning(f"GitHub sync status unavailable: {type(exc).__name__}")
            github_sync = {
                "status": "error",
                "pending_feedback": 0,
                "last_success": None,
                "last_error": "GitHub sync status unavailable",
            }
    return {
        "status": "ok",
        "model_loaded": loader.is_loaded() if loader else False,
        "github_sync": github_sync,
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

    logger.info(
        f"Returning label={result.get('label')} defect_type={result.get('defect_type')} "
        f"with {len(result.get('boxes', []))} boxes"
    )
    return result


@app.post("/feedback", tags=["feedback"])
async def feedback(file: UploadFile = File(...), label: str = Form(...)):
    """Store a user-corrected label next to the image for continuous training.

    Accepted labels are ``good``, every defect type from ``classes.json`` and
    the legacy binary ``defect`` (stored for traceability but ignored by the
    multi-class training preparation).
    """
    label = (label or "").strip()
    allowed = set(load_class_names()) | {"defect"}
    if label not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid label '{label}'. Allowed: {sorted(allowed)}",
        )

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    original_name = Path(file.filename or "image").name
    image_path = FEEDBACK_DIR / f"{timestamp}_{original_name}"

    payload = await file.read()
    image_path.write_bytes(payload)

    label_path = image_path.with_suffix(".txt")
    label_path.write_text(label, encoding="utf-8")

    logger.info(f"Feedback saved: {image_path} (label={label})")

    # Tentative d'envoi immédiate. En cas d'échec, le feedback reste dans
    # data/feedback/ et sera repris par la tâche de fond.
    sync = getattr(app.state, "github_sync", None)
    pushed = False
    if sync is not None and getattr(sync, "enabled", False):
        try:
            await sync.sync_pending()
            pushed = sync.is_pushed(image_path)
        except Exception as exc:
            logger.warning(f"GitHub feedback push failed: {type(exc).__name__}")

    if pushed:
        return {
            "status": "success",
            "message": "Feedback enregistré et envoyé à GitHub",
        }
    return {
        "status": "pending",
        "message": "Feedback enregistré, envoi en cours de nouvelle tentative",
    }
