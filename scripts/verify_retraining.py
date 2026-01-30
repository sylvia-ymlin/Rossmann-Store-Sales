import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import RossmannPipeline
from src.core import setup_logger

logger = setup_logger(__name__)

def verify_retraining():
    """
    Verification script for the Drift-Triggered Auto-Retraining logic.
    """
    train_csv = os.path.abspath("data/raw/train.csv")
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found.")
        return

    logger.info("Starting Auto-Retraining Verification...")
    pipeline = RossmannPipeline(train_csv)
    
    # 1. Warm start: Train initial model on 2013 data
    logger.info("--- Phase 1: Initial Training ---")
    df_full = pipeline.ingestor.ingest(train_csv)
    df_full['Date'] = pd.to_datetime(df_full['Date'])
    
    # Force a small sample for fast verification
    warmup_df = df_full[df_full['Date'] < '2014-01-01'].head(100000)
    feat_warmup = pipeline.run_feature_engineering(warmup_df)
    
    feature_cols = [
        'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
        'Year', 'Month', 'Day', 'IsWeekend', 'DayOfMonth',
        'CompetitionDistance', 'CompetitionOpenTime'
    ] + [c for c in feat_warmup.columns if 'fourier' in c or 'easter' in c]
    
    pipeline.train(feat_warmup[feature_cols].fillna(0), feat_warmup['target'])
    
    # 2. Simulate Normal Streaming (No Drift)
    logger.info("--- Phase 2: Normal Streaming (No Drift) ---")
    pipeline.simulate_streaming(start_date='2014-01-01', end_date='2014-01-15', interval_days=7)
    
    # 3. Injected Drift & Auto-Retrain
    logger.info("--- Phase 3: Injected Drift & Auto-Retrain ---")
    drifted_csv = "data/raw/train_drifted.csv"
    
    # Original train.csv columns to avoid duplication on merge
    original_cols = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
    
    # We use df_full which was already ingested and merged, so we must slice it
    # to only the original columns before saving to simulate a raw file.
    train_cols_present = [c for c in original_cols if c in df_full.columns]
    df_drifted = df_full[train_cols_present].copy()
    
    # Mask: Force Promo=1 for 2014-02-01 onwards to trigger drift
    df_drifted.loc[df_drifted['Date'] >= '2014-02-01', 'Promo'] = 1 
    df_drifted.to_csv(drifted_csv, index=False)
    
    try:
        pipeline_drift = RossmannPipeline(os.path.abspath(drifted_csv))
        pipeline_drift.model = pipeline.model # Start with same model
        
        # This should trigger drift around 2014-02-01
        pipeline_drift.simulate_streaming(
            start_date='2014-01-15', 
            end_date='2014-02-15', 
            interval_days=7,
            drift_threshold=0.1 # Lower threshold for easy trigger
        )
    finally:
        if os.path.exists(drifted_csv):
            os.remove(drifted_csv)

    logger.info("Verification Complete.")

if __name__ == "__main__":
    verify_retraining()
