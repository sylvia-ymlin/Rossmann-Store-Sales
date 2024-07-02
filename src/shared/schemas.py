from typing import Any, Dict, List

from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    Store: int = Field(gt=0)
    Date: str
    Promo: int = 0
    StateHoliday: str = "0"
    SchoolHoliday: int = 0
    ForecastDays: int = Field(default=1, ge=1, le=42)

class ExplanationItem(BaseModel):
    feature: str
    score: float
    formatted_val: str

class PredictionResponse(BaseModel):
    Store: int
    Date: str
    PredictedSales: float
    ModelVersion: str
    Explanation: List[ExplanationItem] = Field(default_factory=list)
    Forecast: List[Dict[str, Any]] = Field(default_factory=list)
    Status: str
