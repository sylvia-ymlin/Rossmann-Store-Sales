from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path

import xgboost as xgb

from src.shared.config import DEFAULT_MODEL_METADATA_PATH, DEFAULT_MODEL_PATH, settings
from src.training.data_loader import load_store_data

logger = logging.getLogger(__name__)


@dataclass
class RuntimeAssets:
    model: xgb.Booster | None
    store_lookup: dict[int, dict[str, object]]
    model_version: str


def load_store_lookup() -> dict[int, dict[str, object]]:
    """Loads store-level metadata used to build prediction rows."""
    store_df = load_store_data(settings.data.store_path).copy()
    store_df["CompetitionDistance"] = store_df["CompetitionDistance"].fillna(100000)
    for col in ["Promo2", "Promo2SinceWeek", "Promo2SinceYear"]:
        store_df[col] = store_df[col].fillna(0).astype(int)
    for col in ["StoreType", "Assortment"]:
        store_df[col] = store_df[col].fillna("0").astype(str)
    return store_df.set_index("Store").to_dict(orient="index")


def load_runtime_assets() -> RuntimeAssets:
    store_lookup = load_store_lookup()

    model: xgb.Booster | None = None
    model_path = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if model_path.exists():
        model = xgb.Booster()
        model.load_model(str(model_path))
        logger.info("Model loaded from %s", model_path)
    else:
        logger.warning("Model not found at %s. Predict endpoint will fail.", model_path)

    metadata_path = Path(os.environ.get("MODEL_METADATA_PATH", str(DEFAULT_MODEL_METADATA_PATH)))
    model_version = "unknown"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            model_version = json.load(f).get("model_version", "unknown")
    else:
        logger.warning("Model metadata not found at %s. Using unknown version.", metadata_path)

    return RuntimeAssets(
        model=model,
        store_lookup=store_lookup,
        model_version=model_version,
    )
