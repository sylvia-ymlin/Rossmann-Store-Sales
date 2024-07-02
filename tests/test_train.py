import numpy as np
import pandas as pd
import xgboost as xgb

from src.training import train as train_module


class DummyRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        self.columns_ = list(X.columns)
        return self

    def predict(self, X):
        return np.log1p(np.full(len(X), 200.0))


def sample_training_frame() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=90, freq="D")
    return pd.DataFrame(
        {
            "Store": 1,
            "DayOfWeek": dates.dayofweek + 1,
            "Date": dates,
            "Sales": np.linspace(1000, 1500, len(dates)),
            "Customers": 500,
            "Open": 1,
            "Promo": 0,
            "StateHoliday": "0",
            "SchoolHoliday": 0,
            "StoreType": "a",
            "Assortment": "a",
            "CompetitionDistance": 1000.0,
            "Promo2": 0,
            "Promo2SinceWeek": 0,
            "Promo2SinceYear": 0,
        }
    )


def test_run_training_saves_model_and_reports_metric(monkeypatch, tmp_path):
    monkeypatch.setattr(train_module, "load_raw_data", lambda *args, **kwargs: sample_training_frame())

    model_path = tmp_path / "model.json"
    metadata_path = tmp_path / "model_metadata.json"
    metrics = train_module.run_training(str(model_path), metadata_path=str(metadata_path))

    assert model_path.exists()
    assert metadata_path.exists()
    assert "validation_rmspe" in metrics
    assert metrics["validation_rows"] == 42

    loaded = xgb.Booster()
    loaded.load_model(str(model_path))
    assert loaded.num_boosted_rounds() > 0
