from pathlib import Path
from typing import Any, Dict, List
import logging

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict

class ModelConfig(BaseModel):
    name: str
    description: str
    tags: List[str]

class DataConfig(BaseModel):
    features: List[str]
    target: str
    train_path: str
    store_path: str

class PipelineConfig(BaseModel):
    fourier_period: float = 365.25
    fourier_order: int = 5

class Config(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: ModelConfig
    data: DataConfig
    pipeline: PipelineConfig
    model_params: Dict[str, Dict[str, Any]]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "rossmann_model.json"
DEFAULT_MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

def load_config(config_path: str = "config.yaml") -> Config:
    """Loads and validates the configuration from a YAML file."""
    candidate_paths = [Path(config_path), PROJECT_ROOT / config_path]

    actual_path = next((path for path in candidate_paths if path.exists()), None)
    if actual_path is None:
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    with actual_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    return Config(**raw_config)

# Global configuration instance
try:
    settings = load_config()
except Exception as e:
    logging.error(f"Failed to load configuration: {e}")
    settings = None  # type: ignore
