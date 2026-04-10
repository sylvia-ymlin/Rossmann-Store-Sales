from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import logging

from src.shared.config import settings
from src.shared.schemas import PredictionRequest, PredictionResponse
from src.serving.monitoring import (
    append_jsonl_record,
    build_inference_log_entry,
    get_inference_log_path,
)
from src.serving.predictor import (
    ModelLike,
    generate_prediction_response,
)
from src.serving.runtime import load_runtime_assets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model: ModelLike | None = None
store_lookup: dict[int, dict[str, object]] = {}
model_version = "unknown"


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, model_version, store_lookup
    assets = load_runtime_assets()
    model = assets.model
    store_lookup = assets.store_lookup
    model_version = assets.model_version
    yield


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
        store_meta = store_lookup[request.Store]
        artifacts = generate_prediction_response(request, model, store_meta, model_version)
        append_jsonl_record(
            get_inference_log_path(),
            build_inference_log_entry(
                store=request.Store,
                start_date=request.Date,
                forecast_days=request.ForecastDays,
                promo=request.Promo,
                state_holiday=request.StateHoliday,
                school_holiday=request.SchoolHoliday,
                model_version=model_version,
                latency_ms=artifacts.latency_ms,
            ),
        )
        logger.info(
            "prediction_complete store=%s horizon=%s model_version=%s latency_ms=%.3f",
            request.Store,
            request.ForecastDays,
            model_version,
            artifacts.latency_ms,
        )
        return artifacts.response

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
