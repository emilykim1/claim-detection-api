import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_predictor():
    """Avoid loading a real model during unit tests."""
    with patch("api.main.predictor") as mock:
        mock.model = MagicMock()  # truthy = "loaded"
        mock.predict.return_value = {
            "is_claim": True,
            "confidence": 0.97,
            "claim_probability": 0.97,
        }
        yield mock


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_claim():
    r = client.post("/predict", json={"sentence": "The Earth is round."})
    assert r.status_code == 200
    data = r.json()
    assert data["is_claim"] is True
    assert 0.0 <= data["confidence"] <= 1.0
    assert "latency_ms" in data


def test_predict_empty_sentence():
    r = client.post("/predict", json={"sentence": ""})
    assert r.status_code == 422  # validation error


def test_predict_too_long():
    r = client.post("/predict", json={"sentence": "x" * 1001})
    assert r.status_code == 422


def test_batch_predict():
    r = client.post("/predict/batch", json=["The sky is blue.", "Hello!"])
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_batch_too_large():
    r = client.post("/predict/batch", json=["x"] * 33)
    assert r.status_code == 400


def test_missing_sentence_field():
    r = client.post("/predict", json={})
    assert r.status_code == 422