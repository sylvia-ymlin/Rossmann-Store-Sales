import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from src.core import setup_logger
from src.data import DataIngestorFactory
from src.features import (
    FeatureEngineer, DateTransformation, RossmannFeatureEngineering,
    FourierSeriesSeasonality, EasterFeature
)
from src.models import ModelEvaluator
from src.monitoring import ExperimentTracker, DriftDetector

logger = setup_logger(__name__)

class RossmannPipeline:
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.factory = DataIngestorFactory()
        self.ingestor = self.factory.get_data_ingestor("rossmann")
        self.model = None
        self.tracker = ExperimentTracker()
        self.drift_detector = DriftDetector()
        
    def run_feature_engineering(self, df):
        logger.info("Running consolidated feature engineering...")
        eng = FeatureEngineer(DateTransformation())
        df = eng.apply_feature_engineering(df)
        eng.set_strategy(RossmannFeatureEngineering())
        df = eng.apply_feature_engineering(df)
        eng.set_strategy(FourierSeriesSeasonality(period=365.25, order=5))
        df = eng.apply_feature_engineering(df)
        eng.set_strategy(EasterFeature())
        df = eng.apply_feature_engineering(df)
        
        if 'Sales' in df.columns:
            df = df[(df['Open'] != 0) & (df['Sales'] > 0)]
            df['target'] = np.log1p(df['Sales'])
        return df

    def train(self, X, y):
        logger.info("Training XGBoost Regressor...")
        self.model = XGBRegressor(
            n_estimators=1000, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        self.model.fit(X, y)
        return self.model

    def evaluate(self, X_test, y_test):
        if self.model is None: raise ValueError("Model not trained.")
        y_pred_log = self.model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)
        rmspe = ModelEvaluator.calculate_rmspe(y_true, y_pred)
        mae = ModelEvaluator.evaluate(self.model, X_test, y_test)["MAE"]
        return {"MAE": mae, "RMSPE": rmspe, "SMAPE": 0.0} # Placeholder for backward compatibility

    def simulate_streaming(self, start_date, end_date, interval_days=7, drift_threshold=0.2):
        logger.info(f"Simulating streaming from {start_date} to {end_date}.")
        full_df = self.ingestor.ingest(self.raw_data_path)
        full_df['Date'] = pd.to_datetime(full_df['Date'])
        
        current_date = pd.to_datetime(start_date)
        while current_date < pd.to_datetime(end_date):
            next_date = current_date + timedelta(days=interval_days)
            batch_df = full_df[(full_df['Date'] >= current_date) & (full_df['Date'] < next_date)]
            if len(batch_df) > 0:
                processed_batch = self.run_feature_engineering(batch_df)
                drift_detected, _ = self.drift_detector.check_drift(processed_batch, ['DayOfWeek', 'Promo'], threshold=drift_threshold)
                if drift_detected:
                    logger.warning(f"Drift detected at {current_date.date()}. Retraining requested.")
                    self.auto_retrain(full_df, current_date)
            current_date = next_date

    def auto_retrain(self, full_df, current_date):
        logger.info(f"Auto-retraining at {current_date.date()}...")
        train_df = full_df[full_df['Date'] < current_date].tail(100000)
        train_feat = self.run_feature_engineering(train_df)
        feature_cols = [
            'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
            'Year', 'Month', 'Day', 'IsWeekend', 'DayOfMonth',
            'CompetitionDistance'
        ] + [c for c in train_feat.columns if 'fourier' in c or 'easter' in c]
        X, y = train_feat[feature_cols].fillna(0), train_feat['target']
        self.train(X, y)
        logger.info("Retraining complete.")
