"""
Centralized configuration and logging for the Rossmann Store Sales Predictor.
"""

import os
import sys
import logging
from typing import List, Optional
from pydantic_settings import BaseSettings
from pythonjsonlogger import jsonlogger

class Settings(BaseSettings):
    # API Settings
    APP_NAME: str = "Rossmann Store Sales Predictor"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Model Settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/rossmann_production_model.pkl")
    FALLBACK_MODEL_PATH: str = "models/model.pkl"
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "true").lower() == "true"
    
    # Data Settings
    DATA_DIR: str = "data"
    EXTRACTED_DATA_DIR: str = "extracted_data"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # MLOps
    MLFLOW_TRACKING_URI: Optional[str] = os.getenv("MLFLOW_TRACKING_URI")
    ZENML_STACK_NAME: str = os.getenv("ZENML_STACK_NAME", "default")

    class Config:
        case_sensitive = True
        env_file = ".env"

# Global settings instance
settings = Settings()

def setup_logger(name: str = "rossmann_predictor") -> logging.Logger:
    """
    Configure a structured JSON logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
