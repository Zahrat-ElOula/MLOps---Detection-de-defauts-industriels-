from pydantic import BaseModel


class Probabilities(BaseModel):
    good: float
    defect: float


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Probabilities


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
