from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime

from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import RossmannPipeline
from src.core import setup_logger
from sklearn.preprocessing import LabelEncoder

logger = setup_logger(__name__)

app = FastAPI(
    title="Rossmann Store Sales Prediction API",
    description="Real-time inference service for store sales forecasting.",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="fastapi_app/static"), name="static")

# Global variables for model and metadata
pipeline = None
store_metadata = None
feature_cols = None
label_encoders = {}

@app.on_event("startup")
def load_assets():
    global pipeline, store_metadata, feature_cols, label_encoders
    
    model_path = os.path.abspath("models/rossmann_production_model.pkl")
    store_path = os.path.abspath("data/raw/store.csv")
    train_sample_path = os.path.abspath("data/raw/train_schema.csv") # Used to init pipeline ingestor
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        raise RuntimeError("Production model missing.")
    
    # Initialize pipeline
    pipeline = RossmannPipeline(train_sample_path)
    with open(model_path, 'rb') as f:
        pipeline.model = pickle.load(f)
        
    # Load store metadata for lookups
    if os.path.exists(store_path):
        store_metadata = pd.read_csv(store_path)
        logger.info("Store metadata loaded for real-time lookups.")
    else:
        logger.error(f"Store metadata not found at {store_path}")
        
    # Define features (must match exactly what XGBoost expects)
    # We use the same list defined in training/submission scripts
    feature_cols = [
        'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
        'Year', 'Month', 'Day', 'IsWeekend', 'DayOfMonth',
        'CompetitionDistance', 'CompetitionOpenTime', 'StoreType', 'Assortment'
    ]
    # Add fourier/easter terms dynamically based on pipeline config
    # Since we know the config (order=5, period=365.25), we can hardcode or reflect
    for i in range(1, 6):
        feature_cols.extend([f'fourier_sin_{i}', f'fourier_cos_{i}'])
    feature_cols.append('easter_effect')
    feature_cols.append('days_to_easter')

class PredictionRequest(BaseModel):
    Store: int
    Date: str
    Promo: int = 0
    StateHoliday: str = "0"
    SchoolHoliday: int = 0

class PredictionResponse(BaseModel):
    Store: int
    Date: str
    PredictedSales: float
    Status: str

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("fastapi_app/static/index.html")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return {}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # 1. Prepare raw input dataframe
        input_data = pd.DataFrame([{
            'Store': request.Store,
            'Date': request.Date,
            'Promo': request.Promo,
            'StateHoliday': request.StateHoliday,
            'SchoolHoliday': request.SchoolHoliday,
            'Open': 1 # Assume open for individual prediction requests
        }])
        
        # 2. Enrich with Store Metadata
        if store_metadata is not None:
            input_data = input_data.merge(store_metadata, on='Store', how='left')
        
        # 3. Apply Feature Engineering
        # Use pipeline's built-in engineering chain
        processed_df = pipeline.run_feature_engineering(input_data)
        
        # 4. Handle Categorical Encoding (StoreType, Assortment)
        # We use a simple fit_transform here for demo, 
        # but in production these should be pre-fitted savers.
        le = LabelEncoder()
        for col in ['StoreType', 'Assortment']:
            if col in processed_df.columns:
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
        
        # 5. Inference
        X = processed_df[feature_cols].fillna(0)
        y_log = pipeline.model.predict(X)
        y_sales = np.expm1(y_log)[0]
        
        return PredictionResponse(
            Store=request.Store,
            Date=request.Date,
            PredictedSales=float(y_sales),
            Status="success"
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
