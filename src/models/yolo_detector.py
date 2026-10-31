from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class YOLODetector:
    def __init__(self, onnx_path: Path | str) -> None:
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        self.session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def is_loaded(self) -> bool:
        return self.session is not None

    def _preprocess(self, image_bytes: bytes) -> tuple[np.ndarray, int, int]:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((640, 640))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        arr = arr[None, ...]
        return arr, orig_w, orig_h

    def detect(self, image_bytes: bytes) -> list[dict]:
        arr, orig_w, orig_h = self._preprocess(image_bytes)
        outputs = self.session.run(None, {self.input_name: arr})[0]  # shape (1, 5, 8400)
        predictions = outputs[0].T  # shape (8400, 5)

        conf_threshold = 0.25
        nms_threshold = 0.45

        mask = predictions[:, 4] > conf_threshold
        predictions = predictions[mask]
        if predictions.shape[0] == 0:
            return []

        boxes_xywh = []
        confidences = []
        for i in range(predictions.shape[0]):
            xc = float(predictions[i, 0])
            yc = float(predictions[i, 1])
            w = float(predictions[i, 2])
            h = float(predictions[i, 3])
            conf = float(predictions[i, 4])
            if conf < conf_threshold:
                continue
            x1 = xc - w / 2
            y1 = yc - h / 2
            boxes_xywh.append([int(x1), int(y1), int(w), int(h)])
            confidences.append(conf)

        if not boxes_xywh:
            return []

        indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences, conf_threshold, nms_threshold)
        results = []
        if len(indices) == 0:
            return []

        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0
        for i in indices.flatten():
            box = boxes_xywh[i]
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int((box[0] + box[2]) * scale_x)
            y2 = int((box[1] + box[3]) * scale_y)
            results.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": float(confidences[i]),
            })
        return results

