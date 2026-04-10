import json
from datetime import datetime, timezone
import os
from pathlib import Path

from src.shared.config import PROJECT_ROOT

DEFAULT_INFERENCE_LOG_PATH = PROJECT_ROOT / "logs" / "inference_requests.jsonl"


def get_inference_log_path() -> Path:
    override = os.environ.get("INFERENCE_LOG_PATH")
    return Path(override) if override else DEFAULT_INFERENCE_LOG_PATH


def build_inference_log_entry(
    *,
    store: int,
    start_date: str,
    forecast_days: int,
    promo: int,
    state_holiday: str,
    school_holiday: int,
    model_version: str,
    latency_ms: float,
) -> dict[str, object]:
    """Builds a single structured inference log event."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "store": store,
        "start_date": start_date,
        "forecast_days": forecast_days,
        "promo": promo,
        "state_holiday": state_holiday,
        "school_holiday": school_holiday,
        "model_version": model_version,
        "latency_ms": round(latency_ms, 3),
    }


def append_jsonl_record(path: Path, payload: dict[str, object]) -> None:
    """Appends a JSON line to the configured log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
