import numpy as np
import pandas as pd


def validation_window_start(max_date: pd.Timestamp, validation_days: int) -> pd.Timestamp:
    """Returns the inclusive start date for a fixed-length validation window."""
    if validation_days < 1:
        raise ValueError("validation_days must be at least 1")
    return pd.Timestamp(max_date).normalize() - pd.Timedelta(days=validation_days - 1)


def holdout_masks(dates: pd.Series, validation_days: int) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp]:
    """Builds an inclusive holdout mask covering exactly validation_days calendar days."""
    normalized_dates = pd.to_datetime(dates).dt.normalize()
    end_date = normalized_dates.max()
    start_date = validation_window_start(end_date, validation_days)
    valid_mask = normalized_dates >= start_date
    return valid_mask, start_date, end_date


def rolling_date_windows(dates: pd.Series, validation_days: int, windows: int) -> list[pd.DatetimeIndex]:
    """Splits the last validation_days * windows unique dates into contiguous windows."""
    if windows < 1:
        raise ValueError("windows must be at least 1")

    unique_dates = pd.Index(sorted(pd.to_datetime(dates).dt.normalize().unique()))
    backtest_dates = unique_dates[-validation_days * windows :]
    chunks = np.array_split(backtest_dates.to_numpy(), windows)
    return [pd.DatetimeIndex(pd.to_datetime(chunk)) for chunk in chunks if len(chunk) > 0]
