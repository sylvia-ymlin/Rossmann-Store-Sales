from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
import pandas as pd
import xgboost as xgb

from src.shared.config import settings
from src.shared.schemas import ExplanationItem, PredictionRequest, PredictionResponse
from src.training.features import apply_feature_pipeline, build_feature_matrix


class SupportsPredict(Protocol):
    def predict(self, X: object) -> np.ndarray: ...


class SupportsBooster(SupportsPredict, Protocol):
    def get_booster(self) -> xgb.Booster: ...


ModelLike = xgb.Booster | SupportsBooster


@dataclass
class PredictionArtifacts:
    response: PredictionResponse
    latency_ms: float


def predict_with_model(loaded_model: ModelLike, X: pd.DataFrame) -> np.ndarray:
    if isinstance(loaded_model, xgb.Booster):
        return loaded_model.predict(xgb.DMatrix(X))
    return loaded_model.predict(X)


def predict_contributions(loaded_model: ModelLike, X: pd.DataFrame) -> np.ndarray:
    if isinstance(loaded_model, xgb.Booster):
        return loaded_model.predict(xgb.DMatrix(X), pred_contribs=True)
    return loaded_model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)


def build_prediction_frame(
    request: PredictionRequest,
    store_meta: dict[str, object],
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    start_date = pd.to_datetime(request.Date)
    dates = [start_date + pd.Timedelta(days=i) for i in range(request.ForecastDays)]

    rows = []
    for current_date in dates:
        rows.append(
            {
                "Store": request.Store,
                "Date": current_date,
                "Promo": request.Promo,
                "StateHoliday": request.StateHoliday,
                "SchoolHoliday": request.SchoolHoliday,
                "Assortment": store_meta["Assortment"],
                "StoreType": store_meta["StoreType"],
                "CompetitionDistance": store_meta["CompetitionDistance"],
                "Promo2": store_meta["Promo2"],
                "Promo2SinceWeek": store_meta["Promo2SinceWeek"],
                "Promo2SinceYear": store_meta["Promo2SinceYear"],
                "Open": 1,
            }
        )

    return pd.DataFrame(rows), dates


def build_explanation_items(feature_cols: list[str], contribs: np.ndarray) -> list[ExplanationItem]:
    avg_contribs = contribs[:, :-1].mean(axis=0)
    top_impacts = sorted(
        zip(feature_cols, avg_contribs),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:5]
    return [
        ExplanationItem(feature=name, score=float(score), formatted_val=f"{score:+.3f}")
        for name, score in top_impacts
    ]


def build_forecast_rows(dates: list[pd.Timestamp], y_sales: np.ndarray) -> list[dict[str, object]]:
    forecast = []
    for current_date, sales in zip(dates, y_sales):
        forecast.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "sales": float(round(sales, 2)),
            }
        )
    return forecast


def generate_prediction_response(
    request: PredictionRequest,
    loaded_model: ModelLike,
    store_meta: dict[str, object],
    model_version: str,
) -> PredictionArtifacts:
    started_at = perf_counter()
    df, dates = build_prediction_frame(request, store_meta)
    df = apply_feature_pipeline(
        df,
        fourier_period=settings.pipeline.fourier_period,
        fourier_order=settings.pipeline.fourier_order,
    )

    feature_cols = settings.data.features
    X = build_feature_matrix(df, feature_cols)

    y_log = predict_with_model(loaded_model, X)
    y_sales = np.expm1(y_log)
    contribs = predict_contributions(loaded_model, X)

    response = PredictionResponse(
        Store=request.Store,
        Date=request.Date,
        PredictedSales=float(y_sales[0]),
        ModelVersion=model_version,
        Explanation=build_explanation_items(feature_cols, contribs),
        Forecast=build_forecast_rows(dates, y_sales),
        Status="success",
    )
    return PredictionArtifacts(
        response=response,
        latency_ms=(perf_counter() - started_at) * 1000,
    )
