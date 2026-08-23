from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    prediction: PredictionItem
    top_predictions: list[PredictionItem]
    low_confidence: bool
    threshold: float = Field(ge=0, le=1)


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_path: str
    detail: str | None = None
