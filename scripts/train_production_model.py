import os
import sys
import pickle
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import RossmannPipeline
from src.core import setup_logger

logger = setup_logger(__name__)

def run_production_training():
    """
    Executes a formal production training run.
    """
    train_csv = os.path.abspath("data/raw/train.csv")
    if not os.path.exists(train_csv):
        logger.error(f"Raw data not found at {train_csv}. Please ensure data is present.")
        return

    logger.info("Initializing Production Training Pipeline...")
    pipeline = RossmannPipeline(train_csv)
    
    # 1. Ingest Full Dataset
    logger.info("Ingesting full dataset...")
    df_raw = pipeline.ingestor.ingest(train_csv)
    
    # 2. Feature Engineering
    logger.info("Running feature engineering...")
    df_feat = pipeline.run_feature_engineering(df_raw)
    
    # 3. Define Final Feature Set
    # Include Store ID and Store Metadata
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in ['StoreType', 'Assortment']:
        if col in df_feat.columns:
            df_feat[col] = le.fit_transform(df_feat[col].astype(str))

    feature_cols = [
        'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
        'Year', 'Month', 'Day', 'IsWeekend', 'DayOfMonth',
        'CompetitionDistance', 'CompetitionOpenTime', 'StoreType', 'Assortment'
    ] + [c for c in df_feat.columns if 'fourier' in c or 'easter' in c]
    
    # 4. Final Training (using all available data to create the 'Gold' model)
    X = df_feat[feature_cols].fillna(0)
    y = df_feat['target']
    
    logger.info(f"Training final model on {len(df_feat)} records with {len(feature_cols)} features...")
    pipeline.train(X, y)
    
    # 5. Export Model & Metadata
    os.makedirs('models', exist_ok=True)
    model_path = 'models/rossmann_production_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline.model, f)
    
    # Log the successful run
    pipeline.tracker.log_experiment(
        name="production_training_run",
        params={
            "feature_count": len(feature_cols),
            "data_size": len(df_feat),
            "model_type": str(type(pipeline.model))
        },
        metrics={"status": "success"}
    )
    
    logger.info("--- PRODUCTION TRAINING COMPLETE ---")
    logger.info(f"Model saved to: {model_path}")
    logger.info("Project is now ready for deployment/inference.")

if __name__ == "__main__":
    run_production_training()
