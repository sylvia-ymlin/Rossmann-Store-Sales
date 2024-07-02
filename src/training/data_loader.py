import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_raw_data(train_path: str, store_path: str) -> pd.DataFrame:
    """Loads and merges the training and store datasets."""
    logger.info(f"Loading data from {train_path} and {store_path}")
    
    train_df = pd.read_csv(train_path, low_memory=False)
    store_df = pd.read_csv(store_path)
    
    # Merge datasets
    df = pd.merge(train_df, store_df, on="Store", how="left")
    logger.info(f"Data merged. Shape: {df.shape}")
    
    return df


def load_store_data(store_path: str) -> pd.DataFrame:
    """Loads store metadata used by both training and serving."""
    logger.info("Loading store metadata from %s", store_path)
    return pd.read_csv(store_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs basic data cleaning."""
    logger.info("Cleaning data...")
    df = df.copy()

    # Fill missing CompetitionDistance with a large value
    if "CompetitionDistance" in df.columns:
        df["CompetitionDistance"] = df["CompetitionDistance"].fillna(100000)

    # Convert StateHoliday to numeric
    if "StateHoliday" in df.columns:
        df["StateHoliday"] = df["StateHoliday"].astype(str).map({
            "0": 0, "a": 1, "b": 2, "c": 3
        }).fillna(0).astype(int)

    # Fill binary promo indicators
    for col in ["Promo2", "Promo2SinceWeek", "Promo2SinceYear"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Filter out closed stores or zero sales for training
    if "Sales" in df.columns:
        df = df[df["Open"] != 0]
        df = df[df["Sales"] > 0]
        logger.info(f"Filtered rows with zero sales/closed shops. New shape: {df.shape}")

    return df
