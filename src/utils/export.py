"""Model export utilities (ONNX, TorchScript)."""

from __future__ import annotations

from pathlib import Path


def export_to_onnx(model, dummy_input, path: Path) -> None:
    """Export a PyTorch model to ONNX format."""
    import torch

    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # Avoids "torch._dynamo.exc.Unsupported" errors with torch.compile()
    )


def export_to_torchscript(model, dummy_input, path: Path) -> None:
    """Export a PyTorch model to TorchScript via tracing."""
    import torch

    model.eval()
    traced = torch.jit.trace(model, dummy_input)
    traced.save(str(path))