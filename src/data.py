"""
Data ingestion and processing modules for Rossmann Store Sales.
"""

import os
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.core import setup_logger

logger = setup_logger(__name__)

# --- INGESTION ---

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        pass

class RossmannDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        logger.info(f"Ingesting Rossmann sales data from {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        data_dir = os.path.dirname(file_path)
        store_path = os.path.join(data_dir, "store.csv")

        if os.path.exists(store_path):
            logger.info(f"Merging with store metadata from {store_path}")
            store_df = pd.read_csv(store_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = pd.merge(df, store_df, on='Store', how='left')
        else:
            logger.warning(f"Store metadata not found. Proceeding with sales data only.")
        return df

class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(dataset_name: str) -> DataIngestor:
        if "rossmann" in dataset_name.lower():
            return RossmannDataIngestor()
        raise ValueError(f"No ingestor available for dataset: {dataset_name}")

# --- PROCESSING / CLEANING ---

class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method: str = "mean", fill_value: any = None):
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df_cleaned = df.copy()
        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].mean())
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        return df_cleaned

# --- OUTLIER DETECTION ---

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        return (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

# --- SPLITTING ---

class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str):
        pass

class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
