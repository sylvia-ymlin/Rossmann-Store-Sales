import pandas as pd

from src.training.features import apply_feature_pipeline


def _training_style_row() -> pd.DataFrame:
    """Many columns, like a row from the merged train+store DataFrame."""
    return pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2015-07-31"],
            "Promo": [1],
            "StateHoliday": ["0"],
            "SchoolHoliday": [0],
            "StoreType": ["a"],
            "Assortment": ["c"],
            "CompetitionDistance": [1200.0],
            "Promo2": [0],
            "Promo2SinceWeek": [0],
            "Promo2SinceYear": [0],
            "Customers": [500],
            "Open": [1],
            "Sales": [5000],
        }
    )


def _serving_style_row() -> pd.DataFrame:
    """Minimal columns, like a row built from PredictionRequest in api.py."""
    return pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2015-07-31"],
            "Promo": [1],
            "StateHoliday": ["0"],
            "SchoolHoliday": [0],
            "StoreType": ["a"],
            "Assortment": ["c"],
            "CompetitionDistance": [1200.0],
            "Open": [1],
        }
    )


EXPECTED_ENGINEERED_COLS = {
    "Year", "Month", "Day", "DayOfWeek", "IsWeekend", "DayOfMonth",
    "fourier_sin_1", "fourier_cos_1",
    "days_to_easter", "easter_effect",
    "LogCompetitionDistance",
}


def test_pipeline_produces_same_engineered_columns():
    """Training-style and serving-style inputs must produce the same engineered columns."""
    train_out = apply_feature_pipeline(_training_style_row())
    serve_out = apply_feature_pipeline(_serving_style_row())

    train_engineered = set(train_out.columns) & EXPECTED_ENGINEERED_COLS
    serve_engineered = set(serve_out.columns) & EXPECTED_ENGINEERED_COLS

    assert train_engineered == serve_engineered, (
        f"Column mismatch: {train_engineered.symmetric_difference(serve_engineered)}"
    )


def test_pipeline_produces_same_values():
    """For shared columns, identical inputs must produce identical feature values."""
    train_out = apply_feature_pipeline(_training_style_row())
    serve_out = apply_feature_pipeline(_serving_style_row())

    shared = sorted(set(train_out.columns) & set(serve_out.columns) & EXPECTED_ENGINEERED_COLS)

    for col in shared:
        t_val = train_out[col].iloc[0]
        s_val = serve_out[col].iloc[0]
        assert t_val == s_val, f"Value mismatch for {col}: training={t_val}, serving={s_val}"
