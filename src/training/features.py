import pandas as pd
import numpy as np
import logging
from typing import Iterable

logger = logging.getLogger(__name__)


def encode_state_holiday(values: pd.Series) -> pd.Series:
    """Normalizes Rossmann StateHoliday values to the numeric encoding used by training."""
    normalized = values.astype(str).str.strip().str.lower().map(
        {"0": 0, "a": 1, "b": 2, "c": 3}
    )
    numeric_fallback = pd.to_numeric(values, errors="coerce")
    return normalized.fillna(numeric_fallback).fillna(0).astype(int)

def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts basic calendar features from the Date column."""
    df = df.copy()
    date_col = "Date" if "Date" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    
    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Day"] = df[date_col].dt.day
    df["DayOfWeek"] = df[date_col].dt.dayofweek + 1
    df["IsWeekend"] = (df[date_col].dt.dayofweek >= 5).astype(int)
    df["DayOfMonth"] = df[date_col].dt.day
    return df

def apply_fourier_seasonality(df: pd.DataFrame, period: float = 365.25, order: int = 5) -> pd.DataFrame:
    """Applies Fourier terms to capture annual seasonality."""
    df = df.copy()
    date_col = "Date" if "Date" in df.columns else "date"
    
    # Calculate days since a reference point
    times = pd.to_datetime(df[date_col]).values.view(np.int64) / 10**9 / (60 * 60 * 24)
    
    for i in range(1, order + 1):
        df[f"fourier_sin_{i}"] = np.sin(2 * np.pi * i * times / period)
        df[f"fourier_cos_{i}"] = np.cos(2 * np.pi * i * times / period)
    return df

def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Easter countdown and holiday effect windows."""
    df = df.copy()
    date_col = "Date" if "Date" in df.columns else "date"
    dates = pd.to_datetime(df[date_col])
    
    # Known Easter dates for the dataset period
    easter_dates = {
        2013: "2013-03-31", 2014: "2014-04-20", 2015: "2015-04-05", 2016: "2016-03-27"
    }
    
    df["days_to_easter"] = 999
    for year, date_str in easter_dates.items():
        mask = dates.dt.year == year
        if any(mask):
            df.loc[mask, "days_to_easter"] = (dates[mask] - pd.to_datetime(date_str)).dt.days
            
    df["easter_effect"] = ((df["days_to_easter"] >= -7) & (df["days_to_easter"] <= 7)).astype(int)
    return df

def apply_rossmann_store_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies store-specific transformations (Competition, Assortment)."""
    df = df.copy()

    if "StateHoliday" in df.columns:
        df["StateHoliday"] = encode_state_holiday(df["StateHoliday"])
    
    # StoreType/Assortment encoding
    if "StoreType" in df.columns:
        df["StoreType"] = df["StoreType"].astype(str).map({"a": 1, "b": 2, "c": 3, "d": 4}).fillna(0)
    
    if "Assortment" in df.columns:
        df["Assortment"] = df["Assortment"].astype(str).map({"a": 1, "b": 2, "c": 3}).fillna(0)
        
    # Log Competiton Distance
    if "CompetitionDistance" in df.columns:
        df["LogCompetitionDistance"] = np.log1p(df["CompetitionDistance"])
        
    return df

def apply_feature_pipeline(
    df: pd.DataFrame,
    *,
    fourier_period: float = 365.25,
    fourier_order: int = 5,
) -> pd.DataFrame:
    """Applies the full feature engineering sequence used by training and serving."""
    df = extract_date_features(df)
    df = apply_fourier_seasonality(df, period=fourier_period, order=fourier_order)
    df = add_holiday_features(df)
    return apply_rossmann_store_features(df)

def build_feature_matrix(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    """Constructs the final feature matrix with strict ordering and clipping."""
    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in df.columns:
            val = df[col]
            # Dataset ends in 2015; clip year to prevent extrapolation on unseen year values
            if col == "Year":
                val = val.clip(upper=2015)
            X[col] = val
        else:
            X[col] = 0
            
    # Ensure numeric and handle any remaining NaNs
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X
