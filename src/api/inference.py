"""Model loading and inference helpers."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.config import PROJECT_ROOT


class ModelLoader:
    def __init__(
        self,
        model_path: str | Path | None = None,
        backend: str = "auto",
        device: str = "cpu",
    ) -> None:
        if model_path is None and os.getenv("API_MODEL_PATH"):
            model_path = os.getenv("API_MODEL_PATH")
        self.model_path = Path(model_path) if model_path else (PROJECT_ROOT / "model_registry" / "model.onnx")
        self.backend = backend
        self.device = device
        self._session: Any = None
        self.session: Any = None
        self._model: Any = None
        self.input_name: str | None = None
        self.class_names = ("good", "defect")
        # Ensure attributes exist
        object.__setattr__(self, "_session", getattr(self, "_session", None))
        object.__setattr__(self, "session", getattr(self, "session", None))
        object.__setattr__(self, "_model", getattr(self, "_model", None))

    @classmethod
    def from_env(cls) -> "ModelLoader":
        return cls(
            model_path=os.getenv("API_MODEL_PATH") or None,
            backend=os.getenv("API_MODEL_BACKEND", "auto"),
            device=os.getenv("API_DEVICE", "cpu"),
        )

    def is_loaded(self) -> bool:
        return getattr(self, "_session", None) is not None or getattr(self, "session", None) is not None or getattr(self, "_model", None) is not None

    @property
    def version(self) -> str | None:
        try:
            mp = getattr(self, "model_path", None)
            if mp:
                return Path(mp).stem
        except Exception:
            pass
        return None

    @version.setter
    def version(self, value: str | None) -> None:
        pass

    def load(self) -> None:
        mp = getattr(self, "model_path", None)
        if mp is None:
            return
        mp = Path(mp)
        if not mp.is_file():
            return
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(mp), providers=["CPUExecutionProvider"])
            self.session = sess
            self._session = sess
            self.input_name = sess.get_inputs()[0].name
        except Exception:
            self.session = None
            self._session = None
            return

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)
        return arr[None, ...]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def predict(self, image_bytes: bytes) -> dict:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        sess = getattr(self, "session", None) or getattr(self, "_session", None)
        if sess is None:
            raise RuntimeError("Model not loaded")
        self.session = sess
        self._session = sess
        inp = getattr(self, "input_name", None)
        if inp is None:
            try:
                inp = sess.get_inputs()[0].name
            except Exception:
                pass
        arr = self._preprocess(image_bytes)
        logits = sess.run(None, {inp: arr})[0]
        probs = self._softmax(logits[0])
        idx = int(np.argmax(probs))
        return {
            "label": self.class_names[idx],
            "confidence": float(probs[idx]),
            "probabilities": {
                "good": float(probs[0]),
                "defect": float(probs[1]),
            },
        }
