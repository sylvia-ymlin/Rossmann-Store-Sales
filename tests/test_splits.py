import pandas as pd

from src.training.splits import holdout_masks, rolling_date_windows


def test_holdout_mask_covers_exact_number_of_days():
    dates = pd.Series(pd.date_range("2015-01-01", periods=90, freq="D"))
    valid_mask, start_date, end_date = holdout_masks(dates, validation_days=42)

    assert valid_mask.sum() == 42
    assert start_date.strftime("%Y-%m-%d") == "2015-02-18"
    assert end_date.strftime("%Y-%m-%d") == "2015-03-31"


def test_rolling_windows_are_contiguous_and_fixed_length():
    dates = pd.Series(pd.date_range("2015-01-01", periods=126, freq="D"))
    windows = rolling_date_windows(dates, validation_days=42, windows=3)

    assert len(windows) == 3
    assert all(len(window) == 42 for window in windows)
    assert windows[0].max() + pd.Timedelta(days=1) == windows[1].min()
    assert windows[1].max() + pd.Timedelta(days=1) == windows[2].min()
