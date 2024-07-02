import pandas as pd

from src.training.features import apply_feature_pipeline, build_feature_matrix


def test_feature_pipeline_builds_expected_columns():
    df = pd.DataFrame(
        {
            "Store": [1],
            "Date": ["2015-07-31"],
            "Promo": [1],
            "StateHoliday": [0],
            "SchoolHoliday": [0],
            "StoreType": ["a"],
            "Assortment": ["c"],
            "CompetitionDistance": [1200.0],
        }
    )

    transformed = apply_feature_pipeline(df, fourier_order=2)

    assert {"Year", "Month", "DayOfWeek", "fourier_sin_1", "days_to_easter", "LogCompetitionDistance"} <= set(
        transformed.columns
    )

    X = build_feature_matrix(transformed, ["Store", "Year", "fourier_sin_1", "missing_feature"])
    assert list(X.columns) == ["Store", "Year", "fourier_sin_1", "missing_feature"]
    assert X.loc[0, "missing_feature"] == 0
