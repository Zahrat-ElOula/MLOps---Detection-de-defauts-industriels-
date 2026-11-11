from typing import Any

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class PredictionResponse(BaseModel):
    label: str
    defect_type: str = "none"
    confidence: float
    probabilities: dict[str, float]
    classes: list[str] = Field(default_factory=list)
    boxes: list[BoundingBox] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    github_sync: dict[str, Any] = Field(default_factory=dict)
