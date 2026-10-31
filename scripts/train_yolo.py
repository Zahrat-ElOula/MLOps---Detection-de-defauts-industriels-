#!/usr/bin/env python
from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    from ultralytics import YOLO

    project_root = Path(__file__).resolve().parents[1]
    data_yaml = project_root / "data" / "yolo_v2" / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml introuvable : {data_yaml}")
    print(f"Using data.yaml: {data_yaml}")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=20,
        imgsz=640,
        batch=16,
        name="defect_detector",
        project=str(project_root / "model_registry" / "yolo_runs"),
    )
    model.export(format="onnx", imgsz=640)

    runs_dir = project_root / "model_registry" / "yolo_runs" / "defect_detector"
    best_pt = runs_dir / "weights" / "best.pt"
    best_onnx = runs_dir / "weights" / "best.onnx"
    out_dir = project_root / "model_registry"
    if best_pt.exists():
        shutil.copy2(best_pt, out_dir / "yolo_best.pt")
    if best_onnx.exists():
        shutil.copy2(best_onnx, out_dir / "yolo_best.onnx")


if __name__ == "__main__":
    main()
