# ruff: noqa: E402
import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.shared.config import settings
from src.shared.mlflow_utils import start_run
from src.training.data_loader import clean_data, load_raw_data
from src.training.features import apply_feature_pipeline, build_feature_matrix
from src.training.splits import holdout_masks, rolling_date_windows

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ORIGINAL_CONFIG = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "objective": "reg:squarederror",
}


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    safe_true = np.clip(y_true, a_min=1.0, a_max=None)
    return float(np.sqrt(np.mean(np.square((y_true - y_pred) / safe_true))) * 100)


def prepare_dataset() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.Series]:
    logger.info("Loading Rossmann data for evaluation")
    df = load_raw_data(settings.data.train_path, settings.data.store_path)
    df = clean_data(df)
    df = apply_feature_pipeline(
        df,
        fourier_period=settings.pipeline.fourier_period,
        fourier_order=settings.pipeline.fourier_order,
    )
    X = build_feature_matrix(df, settings.data.features)
    y = df[settings.data.target].to_numpy()
    dates = pd.to_datetime(df["Date"]).dt.normalize()
    return df, X, y, dates


def baseline_predict(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> np.ndarray:
    global_mean = float(train_df["Sales"].mean())
    store_dow_mean = train_df.groupby(["Store", "DayOfWeek"])["Sales"].mean()
    store_mean = train_df.groupby("Store")["Sales"].mean()
    valid_keys = list(zip(valid_df["Store"], valid_df["DayOfWeek"]))

    return np.array(
        [store_dow_mean.get(key, store_mean.get(key[0], global_mean)) for key in valid_keys],
        dtype=float,
    )


def fit_and_score(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    params: dict[str, Any],
) -> dict[str, float]:
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, np.log1p(y_train))
    train_pred = np.expm1(model.predict(X_train))
    valid_pred = np.expm1(model.predict(X_valid))
    return {
        "train_rmspe": round(rmspe(y_train, train_pred), 4),
        "valid_rmspe": round(rmspe(y_valid, valid_pred), 4),
    }


def holdout_evaluation(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    dates: pd.Series,
    validation_days: int,
) -> dict[str, Any]:
    valid_mask, validation_start, validation_end = holdout_masks(dates, validation_days)
    train_mask = ~valid_mask

    train_df = df.loc[train_mask]
    valid_df = df.loc[valid_mask]
    y_valid = y[valid_mask]

    baseline_score = rmspe(y_valid, baseline_predict(train_df, valid_df))
    original_score = fit_and_score(
        X.loc[train_mask],
        X.loc[valid_mask],
        y[train_mask],
        y_valid,
        ORIGINAL_CONFIG,
    )
    tuned_score = fit_and_score(
        X.loc[train_mask],
        X.loc[valid_mask],
        y[train_mask],
        y_valid,
        settings.model_params["xgboost"],
    )

    return {
        "validation_days": validation_days,
        "validation_start_date": validation_start.strftime("%Y-%m-%d"),
        "validation_end_date": validation_end.strftime("%Y-%m-%d"),
        "rows_train": int(train_mask.sum()),
        "rows_valid": int(valid_mask.sum()),
        "baseline_rmspe": round(baseline_score, 4),
        "pre_tuning_model": original_score,
        "tuned_model": tuned_score,
        "tuned_improvement_vs_pre_tuning": round(
            original_score["valid_rmspe"] - tuned_score["valid_rmspe"], 4
        ),
        "tuned_improvement_vs_baseline": round(
            baseline_score - tuned_score["valid_rmspe"], 4
        ),
    }


def rolling_backtest(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    dates: pd.Series,
    validation_days: int,
    windows: int,
) -> list[dict[str, Any]]:
    date_windows = rolling_date_windows(dates, validation_days, windows)

    results: list[dict[str, Any]] = []
    for index, window_dates in enumerate(date_windows, start=1):
        train_mask = dates < window_dates.min()
        valid_mask = dates.isin(window_dates)

        train_df = df.loc[train_mask]
        valid_df = df.loc[valid_mask]
        y_valid = y[valid_mask]

        baseline_score = rmspe(y_valid, baseline_predict(train_df, valid_df))
        tuned_score = fit_and_score(
            X.loc[train_mask],
            X.loc[valid_mask],
            y[train_mask],
            y_valid,
            settings.model_params["xgboost"],
        )

        results.append(
            {
                "window": index,
                "validation_start_date": window_dates.min().strftime("%Y-%m-%d"),
                "validation_end_date": window_dates.max().strftime("%Y-%m-%d"),
                "rows_train": int(train_mask.sum()),
                "rows_valid": int(valid_mask.sum()),
                "baseline_rmspe": round(baseline_score, 4),
                "tuned_valid_rmspe": tuned_score["valid_rmspe"],
                "improvement_vs_baseline": round(baseline_score - tuned_score["valid_rmspe"], 4),
            }
        )

    return results


def build_summary(holdout: dict[str, Any], backtest: list[dict[str, Any]]) -> dict[str, Any]:
    backtest_scores = [window["tuned_valid_rmspe"] for window in backtest]
    baseline_scores = [window["baseline_rmspe"] for window in backtest]
    return {
        "dataset_rows_after_cleaning": holdout["rows_train"] + holdout["rows_valid"],
        "holdout": holdout,
        "rolling_backtest": backtest,
        "rolling_backtest_summary": {
            "windows": len(backtest),
            "average_tuned_rmspe": round(float(np.mean(backtest_scores)), 4),
            "average_baseline_rmspe": round(float(np.mean(baseline_scores)), 4),
            "average_improvement_vs_baseline": round(
                float(np.mean(np.array(baseline_scores) - np.array(backtest_scores))), 4
            ),
        },
        "selected_model_params": settings.model_params["xgboost"],
    }


def run_evaluation(
    output_path: str = "metrics/model_evaluation.json",
    *,
    track_experiments: bool = True,
) -> Path:
    df, X, y, dates = prepare_dataset()
    holdout = holdout_evaluation(df, X, y, dates, validation_days=42)
    backtest = rolling_backtest(df, X, y, dates, validation_days=42, windows=3)
    summary = build_summary(holdout, backtest)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    run_name = f"xgb_backtest_{holdout['validation_start_date']}_{holdout['validation_end_date']}"
    with start_run(run_name, experiment_name="rossmann-evaluation") if track_experiments else nullcontext() as run:
        if track_experiments and run is not None:
            import mlflow

            mlflow.log_param("validation_days", holdout["validation_days"])
            mlflow.log_param("backtest_windows", summary["rolling_backtest_summary"]["windows"])
            mlflow.log_metric("holdout_baseline_rmspe", holdout["baseline_rmspe"])
            mlflow.log_metric("holdout_tuned_rmspe", holdout["tuned_model"]["valid_rmspe"])
            mlflow.log_metric(
                "average_backtest_rmspe",
                summary["rolling_backtest_summary"]["average_tuned_rmspe"],
            )
            mlflow.log_metric(
                "average_backtest_improvement_vs_baseline",
                summary["rolling_backtest_summary"]["average_improvement_vs_baseline"],
            )
            mlflow.log_artifact(str(output))

    logger.info("Evaluation summary written to %s", output)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run holdout and rolling evaluation for Rossmann.")
    parser.add_argument("--output-path", default="metrics/model_evaluation.json")
    parser.add_argument("--no-track", action="store_true", help="Disable MLflow experiment tracking.")
    return parser.parse_args(argv)


def nullcontext():
    from contextlib import nullcontext as _nullcontext

    return _nullcontext(None)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    return run_evaluation(output_path=args.output_path, track_experiments=not args.no_track)


if __name__ == "__main__":
    main()
