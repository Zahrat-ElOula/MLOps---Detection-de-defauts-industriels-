"""Export model_registry/model.pt to model_registry/model.onnx.

Usage:
    python scripts/export_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.config import Config
from src.models.build import build_model
from src.utils.export import export_to_onnx


def main() -> int:
    config = Config.load(params_path=PROJECT_ROOT / "configs" / "params.yaml")
    params = dict(config.params)

    model = build_model(params["model_name"], num_classes=2, pretrained=False)
    state = torch.load(PROJECT_ROOT / "model_registry" / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    onnx_path = PROJECT_ROOT / "model_registry" / "model.onnx"
    export_to_onnx(model, dummy, onnx_path)
    print(f"Exported: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    onnx_out = session.run(None, {"input": dummy.numpy()})[0]
    diff = float(np.abs(torch_out - onnx_out).max())
    print(f"Max diff torch vs onnx: {diff:.6e} (attendu < 1e-4)")
    assert diff < 1e-4, "ONNX export mismatch"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())