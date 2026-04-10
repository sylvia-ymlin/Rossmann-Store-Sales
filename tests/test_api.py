import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.serving import api


class DummyBooster:
    def predict(self, matrix, pred_contribs=False):
        column_count = matrix.num_col()
        return np.array([[0.1] * (column_count + 1)])


class DummyModel:
    def predict(self, X):
        return np.log1p(np.full(len(X), 100.0))

    def get_booster(self):
        return DummyBooster()


def test_predict_endpoint_uses_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setenv("INFERENCE_LOG_PATH", str(tmp_path / "inference.jsonl"))
    client = TestClient(api.app)
    monkeypatch.setattr(api, "model", DummyModel())
    monkeypatch.setattr(api, "model_version", "test-version")
    monkeypatch.setattr(
        api,
        "store_lookup",
        {
            1: {
                "StoreType": "a",
                "Assortment": "c",
                "CompetitionDistance": 1200.0,
                "Promo2": 0,
                "Promo2SinceWeek": 0,
                "Promo2SinceYear": 0,
            }
        },
    )

    response = client.post(
        "/predict",
        json={
            "Store": 1,
            "Date": "2015-07-31",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["Store"] == 1
    assert payload["Status"] == "success"
    assert payload["PredictedSales"] == pytest.approx(100.0)
    assert payload["ModelVersion"] == "test-version"
    assert len(payload["Forecast"]) == 1
    assert len(payload["Explanation"]) == 5
    assert "score" in payload["Explanation"][0]


def test_predict_endpoint_returns_404_for_unknown_store(monkeypatch):
    monkeypatch.setenv("INFERENCE_LOG_PATH", "logs/test_inference.jsonl")
    client = TestClient(api.app)
    monkeypatch.setattr(api, "model", DummyModel())
    monkeypatch.setattr(api, "store_lookup", {})

    response = client.post(
        "/predict",
        json={
            "Store": 9999,
            "Date": "2015-07-31",
        },
    )

    assert response.status_code == 404


def test_health_endpoint_is_stateless():
    client = TestClient(api.app)
    api.model_version = "test-version"
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_version"] == "test-version"
