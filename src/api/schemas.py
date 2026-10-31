from pydantic import BaseModel


class Probabilities(BaseModel):
    good: float
    defect: float


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Probabilities
    boxes: list[BoundingBox] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

