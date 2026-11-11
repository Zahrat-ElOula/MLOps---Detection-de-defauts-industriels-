"""Model loading and inference helpers."""

from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.config import PROJECT_ROOT


def load_class_names(
    classes_path: str | Path | None = None,
    num_outputs: int | None = None,
) -> tuple[str, ...]:
    """Load the class list from ``classes.json``.

    When the saved list does not match the ONNX output dimension (for
    example an old binary model), fall back to ``("good", "defect")`` or
    generic names. This keeps old artefacts servable during migration.
    """
    candidates: list[Path] = []
    if classes_path is not None:
        candidates.append(Path(classes_path))
    candidates.append(PROJECT_ROOT / "model_registry" / "classes.json")

    for path in candidates:
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        values = raw.get("classes", raw) if isinstance(raw, dict) else raw
        names = tuple(str(name) for name in values)
        if num_outputs is None or len(names) == num_outputs:
            return names

    if num_outputs == 2:
        return ("good", "defect")
    if num_outputs and num_outputs > 0:
        return tuple(f"class_{index}" for index in range(num_outputs))
    return ("good", "defect")


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
        self.class_names: tuple[str, ...] = load_class_names()
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

            num_outputs: int | None = None
            try:
                output_shape = sess.get_outputs()[0].shape
                if output_shape and isinstance(output_shape[-1], int):
                    num_outputs = output_shape[-1]
            except Exception:
                pass
            self.class_names = load_class_names(mp.parent / "classes.json", num_outputs)
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
        probs = np.asarray(self._softmax(logits[0])).reshape(-1)

        names = tuple(self.class_names)
        if len(names) < len(probs):
            names = names + tuple(
                f"class_{index}" for index in range(len(names), len(probs))
            )
        names = names[: len(probs)]

        index = int(np.argmax(probs))
        predicted = names[index]
        is_good = predicted == "good"
        probabilities = {
            name: float(probs[position]) for position, name in enumerate(names)
        }
        return {
            "label": "good" if is_good else "defect",
            "defect_type": "none" if is_good else predicted,
            "confidence": float(probs[index]),
            "probabilities": probabilities,
            "classes": list(names),
        }
