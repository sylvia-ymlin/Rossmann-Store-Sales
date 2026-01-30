from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from src.core import setup_logger

logger = setup_logger(__name__)

try:
    import shap
except ImportError:
    shap = None

# --- MODEL BUILDING ---

class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        pass

class XGBoostStrategy(ModelBuildingStrategy):
    def __init__(self, **params):
        self.params = params

    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        from xgboost import XGBRegressor
        logger.info("Building XGBoost model.")
        
        # Filtering logic for Rossmann
        valid_mask = (y_train > 0)
        if "Open" in X_train.columns:
            valid_mask = valid_mask & (X_train["Open"] == 1)
        
        X_filtered = X_train[valid_mask]
        y_log = np.log1p(y_train[valid_mask])
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(**self.params))
        ])
        pipeline.fit(X_filtered, y_log)
        return pipeline

# --- EVALUATION ---

class ModelEvaluator:
    @staticmethod
    def calculate_rmspe(y_true, y_pred):
        mask = y_true > 0
        return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask])**2)) * 100

    @staticmethod
    def evaluate(model, X_test, y_test):
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_true = y_test if not isinstance(y_test, pd.Series) else y_test.values
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmspe = ModelEvaluator.calculate_rmspe(y_true, y_pred)
        
        return {"MSE": mse, "MAE": mae, "RMSPE": rmspe}

# --- EXPLAINABILITY ---

class ModelExplainer:
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        if shap is None:
            logger.warning("SHAP not installed. Explainer will not function.")

    def plot_importance(self, X, save_path=None):
        if hasattr(self.model, 'named_steps'):
            importances = self.model.named_steps['model'].feature_importances_
        else:
            importances = self.model.feature_importances_
        
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        feat_imp.head(20).plot(kind='bar')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        return feat_imp
