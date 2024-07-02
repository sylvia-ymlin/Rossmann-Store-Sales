# ruff: noqa: E402
import numpy as np
import xgboost as xgb
import logging
import json
from pathlib import Path
import sys
import subprocess
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import DEFAULT_MODEL_METADATA_PATH, DEFAULT_MODEL_PATH, settings
from src.shared.mlflow_utils import start_run
from src.training.data_loader import load_raw_data, clean_data
from src.training.features import (
    apply_feature_pipeline,
    build_feature_matrix
)
from src.training.splits import holdout_masks

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_model_metadata(model_path: Path) -> dict[str, str]:
    """Builds lightweight metadata for the generated model artifact."""
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        logger.info("Unable to read git hash; using 'unknown' in model metadata.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version = f"{timestamp}-{git_hash}"
    return {
        "model_version": version,
        "created_at_utc": timestamp,
        "git_short_hash": git_hash,
        "model_path": str(model_path),
    }

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes RMSPE in percentage points."""
    safe_true = np.clip(y_true, a_min=1.0, a_max=None)
    return float(np.sqrt(np.mean(np.square((y_true - y_pred) / safe_true))) * 100)

def run_training(
    model_path: str = str(DEFAULT_MODEL_PATH),
    metrics_path: str = "reports/metrics/training_summary.json",
    metadata_path: str = str(DEFAULT_MODEL_METADATA_PATH),
) -> dict:
    """Runs training, records validation metadata, and saves the final model."""
    logger.info("Starting Rossmann training pipeline")
    
    # 1. Load and Clean Data
    df = load_raw_data(settings.data.train_path, settings.data.store_path)
    df = clean_data(df)
    
    # 2. Feature Engineering
    logger.info("Applying feature engineering...")
    df = apply_feature_pipeline(
        df,
        fourier_period=settings.pipeline.fourier_period,
        fourier_order=settings.pipeline.fourier_order,
    )
    
    # 3. Final Feature Matrix Construction
    feature_cols = settings.data.features
    X = build_feature_matrix(df, feature_cols)
    
    # Target transformation (Log)
    y = np.log1p(df[settings.data.target])

    # 4. Simple time-based validation
    params = settings.model_params.get("xgboost", {})
    metrics = {
        "num_rows": int(len(df)),
        "num_features": int(len(feature_cols)),
        "validation_days": 42,
        "model_params": params,
    }
    validation_mask, validation_start, validation_end = holdout_masks(df["Date"], validation_days=42)
    metrics["validation_start_date"] = validation_start.strftime("%Y-%m-%d")
    metrics["validation_end_date"] = validation_end.strftime("%Y-%m-%d")

    if validation_mask.any() and (~validation_mask).any():
        train_model = xgb.XGBRegressor(**params)
        train_model.fit(X.loc[~validation_mask], y.loc[~validation_mask])
        y_train_actual = np.expm1(y.loc[~validation_mask].to_numpy())
        y_train_pred = np.expm1(train_model.predict(X.loc[~validation_mask]))
        y_valid = np.expm1(y.loc[validation_mask].to_numpy())
        y_pred = np.expm1(train_model.predict(X.loc[validation_mask]))
        metrics["train_rmspe"] = round(rmspe(y_train_actual, y_train_pred), 4)
        metrics["validation_rmspe"] = round(rmspe(y_valid, y_pred), 4)
        metrics["validation_rows"] = int(validation_mask.sum())
        logger.info("Validation RMSPE: %.4f%%", metrics["validation_rmspe"])

    # 5. Train final model on all available data and save it
    final_model = xgb.XGBRegressor(**params)
    logger.info("Fitting final XGBoost model...")
    final_model.fit(X, y)

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(output_path))

    summary_path = Path(metrics_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    metadata_output_path = Path(metadata_path)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    model_metadata = build_model_metadata(output_path)
    with metadata_output_path.open("w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    run_name = f"xgb_holdout_{metrics['validation_start_date']}_{metrics['validation_end_date']}"
    with start_run(run_name, experiment_name="rossmann-training") as run:
        if run is not None:
            import mlflow

            mlflow.log_params(params)
            mlflow.log_param("num_rows", metrics["num_rows"])
            mlflow.log_param("num_features", metrics["num_features"])
            mlflow.log_param("validation_days", metrics["validation_days"])
            mlflow.log_param("validation_start_date", metrics["validation_start_date"])
            mlflow.log_param("validation_end_date", metrics["validation_end_date"])
            if "train_rmspe" in metrics:
                mlflow.log_metric("train_rmspe", metrics["train_rmspe"])
            if "validation_rmspe" in metrics:
                mlflow.log_metric("validation_rmspe", metrics["validation_rmspe"])
            mlflow.log_artifact(str(output_path))
            mlflow.log_artifact(str(summary_path))
            mlflow.log_artifact(str(metadata_output_path))

    logger.info("Model saved to %s", output_path)
    logger.info("Training summary written to %s", summary_path)
    logger.info("Model metadata written to %s", metadata_output_path)
    logger.info("Training pipeline completed successfully.")
    return metrics

if __name__ == "__main__":
    run_training()
