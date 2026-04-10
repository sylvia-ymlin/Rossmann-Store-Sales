# ruff: noqa: E402
import argparse
import numpy as np
import xgboost as xgb
import logging
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    metrics_path: str = "metrics/training_summary.json",
    metadata_path: str = str(DEFAULT_MODEL_METADATA_PATH),
    *,
    track_experiments: bool = True,
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
    with start_run(run_name, experiment_name="rossmann-training") if track_experiments else nullcontext() as run:
        if track_experiments and run is not None:
            import mlflow

            mlflow.log_params(params)
            mlflow.log_param("num_rows", metrics["num_rows"])
            mlflow.log_param("num_features", metrics["num_features"])
            mlflow.log_param("validation_days", metrics["validation_days"])
            mlflow.log_param("validation_start_date", metrics["validation_start_date"])
            mlflow.log_param("validation_end_date", metrics["validation_end_date"])
            train_rmspe = metrics.get("train_rmspe")
            if isinstance(train_rmspe, (int, float)):
                mlflow.log_metric("train_rmspe", float(train_rmspe))
            validation_rmspe = metrics.get("validation_rmspe")
            if isinstance(validation_rmspe, (int, float)):
                mlflow.log_metric("validation_rmspe", float(validation_rmspe))
            mlflow.log_artifact(str(output_path))
            mlflow.log_artifact(str(summary_path))
            mlflow.log_artifact(str(metadata_output_path))

    logger.info("Model saved to %s", output_path)
    logger.info("Training summary written to %s", summary_path)
    logger.info("Model metadata written to %s", metadata_output_path)
    logger.info("Training pipeline completed successfully.")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Rossmann XGBoost model.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--metrics-path", default="metrics/training_summary.json")
    parser.add_argument("--metadata-path", default=str(DEFAULT_MODEL_METADATA_PATH))
    parser.add_argument("--no-track", action="store_true", help="Disable MLflow experiment tracking.")
    return parser.parse_args(argv)


def nullcontext():
    from contextlib import nullcontext as _nullcontext

    return _nullcontext(None)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    return run_training(
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        metadata_path=args.metadata_path,
        track_experiments=not args.no_track,
    )


if __name__ == "__main__":
    main()
