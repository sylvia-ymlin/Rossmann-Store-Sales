# ruff: noqa: E402
import argparse
import json
from pathlib import Path

import pandas as pd

from src.serving.monitoring import get_inference_log_path
from src.shared.config import settings
from src.training.data_loader import clean_data, load_raw_data, load_store_data
from src.training.features import apply_feature_pipeline, build_feature_matrix


def load_recent_inference_rows(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame()

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return pd.DataFrame()

    request_df = pd.DataFrame(records)
    store_df = load_store_data(settings.data.store_path)
    request_df = request_df.rename(
        columns={
            "store": "Store",
            "start_date": "Date",
            "promo": "Promo",
            "state_holiday": "StateHoliday",
            "school_holiday": "SchoolHoliday",
        }
    )
    merged = request_df.merge(store_df, on="Store", how="left")
    merged["Open"] = 1
    for col in ["Promo2", "Promo2SinceWeek", "Promo2SinceYear"]:
        merged[col] = merged[col].fillna(0).astype(int)
    return merged


def build_drift_report() -> dict[str, object]:
    train_df = load_raw_data(settings.data.train_path, settings.data.store_path)
    train_df = clean_data(train_df)
    train_df = apply_feature_pipeline(
        train_df,
        fourier_period=settings.pipeline.fourier_period,
        fourier_order=settings.pipeline.fourier_order,
    )
    train_features = build_feature_matrix(train_df, settings.data.features)

    inference_log_path = get_inference_log_path()
    inference_df = load_recent_inference_rows(inference_log_path)
    if inference_df.empty:
        return {
            "status": "no_inference_logs",
            "log_path": str(inference_log_path),
        }

    inference_df = apply_feature_pipeline(
        inference_df,
        fourier_period=settings.pipeline.fourier_period,
        fourier_order=settings.pipeline.fourier_order,
    )
    inference_features = build_feature_matrix(inference_df, settings.data.features)

    drift_rows = []
    for col in inference_features.columns:
        train_mean = float(train_features[col].mean())
        inference_mean = float(inference_features[col].mean())
        train_std = float(train_features[col].std(ddof=0))
        drift_score = abs(inference_mean - train_mean) / max(train_std, 1e-6)
        drift_rows.append(
            {
                "feature": col,
                "train_mean": round(train_mean, 4),
                "inference_mean": round(inference_mean, 4),
                "train_std": round(train_std, 4),
                "normalized_mean_shift": round(float(drift_score), 4),
            }
        )

    drift_rows.sort(key=lambda row: row["normalized_mean_shift"], reverse=True)
    return {
        "status": "ok",
        "log_path": str(inference_log_path),
        "num_inference_events": int(len(inference_df)),
        "top_drift_features": drift_rows[:10],
    }


def write_drift_report(output_path: str = "metrics/drift_report.json") -> Path:
    report = build_drift_report()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Drift report written to {output}")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline drift report from inference logs.")
    parser.add_argument("--output-path", default="metrics/drift_report.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    return write_drift_report(output_path=args.output_path)


if __name__ == "__main__":
    main()
