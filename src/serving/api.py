from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import os
from pathlib import Path
import json
from time import perf_counter

from src.shared.config import DEFAULT_MODEL_METADATA_PATH, DEFAULT_MODEL_PATH, settings
from src.shared.schemas import PredictionRequest, PredictionResponse, ExplanationItem
from src.serving.monitoring import (
    DEFAULT_INFERENCE_LOG_PATH,
    append_jsonl_record,
    build_inference_log_entry,
)
from src.training.data_loader import load_store_data
from src.training.features import (
    apply_feature_pipeline,
    build_feature_matrix
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model: xgb.Booster | None = None
store_lookup: dict[int, dict[str, object]] = {}
model_version = "unknown"


def load_store_lookup() -> dict[int, dict[str, object]]:
    """Loads store-level metadata used to build prediction rows."""
    store_df = load_store_data(settings.data.store_path).copy()
    store_df["CompetitionDistance"] = store_df["CompetitionDistance"].fillna(100000)
    for col in ["Promo2", "Promo2SinceWeek", "Promo2SinceYear"]:
        store_df[col] = store_df[col].fillna(0).astype(int)
    for col in ["StoreType", "Assortment"]:
        store_df[col] = store_df[col].fillna("0").astype(str)
    return store_df.set_index("Store").to_dict(orient="index")


def load_runtime_assets() -> None:
    global model, model_version, store_lookup

    store_lookup = load_store_lookup()
    model_path = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if model_path.exists():
        model = xgb.Booster()
        model.load_model(str(model_path))
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.warning(f"Model not found at {model_path}. Predict endpoint will fail.")

    metadata_path = Path(os.environ.get("MODEL_METADATA_PATH", str(DEFAULT_MODEL_METADATA_PATH)))
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            model_version = json.load(f).get("model_version", "unknown")
    else:
        logger.warning("Model metadata not found at %s. Using unknown version.", metadata_path)
        model_version = "unknown"


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_runtime_assets()
    yield


def predict_with_model(loaded_model: object, X: pd.DataFrame) -> np.ndarray:
    if isinstance(loaded_model, xgb.Booster):
        return loaded_model.predict(xgb.DMatrix(X))
    return loaded_model.predict(X)


def predict_contributions(loaded_model: object, X: pd.DataFrame) -> np.ndarray:
    if isinstance(loaded_model, xgb.Booster):
        return loaded_model.predict(xgb.DMatrix(X), pred_contribs=True)
    if hasattr(loaded_model, "get_booster"):
        return loaded_model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    raise TypeError("Loaded model does not support feature contribution prediction")


app = FastAPI(
    title=settings.model.name,
    description=settings.model.description,
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None, "model_version": model_version}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if request.Store not in store_lookup:
        raise HTTPException(status_code=404, detail=f"Store {request.Store} not found in metadata")

    try:
        started_at = perf_counter()
        store_meta = store_lookup[request.Store]
        start_date = pd.to_datetime(request.Date)
        dates = [start_date + pd.Timedelta(days=i) for i in range(request.ForecastDays)]

        rows = []
        for d in dates:
            rows.append({
                "Store": request.Store,
                "Date": d,
                "Promo": request.Promo,
                "StateHoliday": request.StateHoliday,
                "SchoolHoliday": request.SchoolHoliday,
                "Assortment": store_meta["Assortment"],
                "StoreType": store_meta["StoreType"],
                "CompetitionDistance": store_meta["CompetitionDistance"],
                "Promo2": store_meta["Promo2"],
                "Promo2SinceWeek": store_meta["Promo2SinceWeek"],
                "Promo2SinceYear": store_meta["Promo2SinceYear"],
                "Open": 1
            })

        df = pd.DataFrame(rows)
        df = apply_feature_pipeline(
            df,
            fourier_period=settings.pipeline.fourier_period,
            fourier_order=settings.pipeline.fourier_order,
        )

        feature_cols = settings.data.features
        X = build_feature_matrix(df, feature_cols)

        y_log = predict_with_model(model, X)
        y_sales = np.expm1(y_log)

        contribs = predict_contributions(model, X)
        avg_contribs = contribs[:, :-1].mean(axis=0)

        explanation_items = []
        impact_map = sorted(zip(feature_cols, avg_contribs), key=lambda x: abs(x[1]), reverse=True)[:5]
        for name, score in impact_map:
            explanation_items.append(ExplanationItem(
                feature=name,
                score=float(score),
                formatted_val=f"{score:+.3f}"
            ))

        forecast_result = []
        for d, s in zip(dates, y_sales):
            date_str = d.strftime("%Y-%m-%d")
            sales_val = float(round(s, 2))
            forecast_result.append({
                "date": date_str,
                "sales": sales_val
            })

        latency_ms = (perf_counter() - started_at) * 1000
        append_jsonl_record(
            DEFAULT_INFERENCE_LOG_PATH,
            build_inference_log_entry(
                store=request.Store,
                start_date=request.Date,
                forecast_days=request.ForecastDays,
                promo=request.Promo,
                state_holiday=request.StateHoliday,
                school_holiday=request.SchoolHoliday,
                model_version=model_version,
                latency_ms=latency_ms,
            ),
        )
        logger.info(
            "prediction_complete store=%s horizon=%s model_version=%s latency_ms=%.3f",
            request.Store,
            request.ForecastDays,
            model_version,
            latency_ms,
        )

        return PredictionResponse(
            Store=request.Store,
            Date=request.Date,
            PredictedSales=float(y_sales[0]),
            ModelVersion=model_version,
            Explanation=explanation_items,
            Forecast=forecast_result,
            Status="success"
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def index():
    from web.frontend import get_frontend_html
    return get_frontend_html()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
