import os

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    not os.path.exists("models/model.joblib"), reason="model not trained yet")

SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
          "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
          "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}


def client():
    from src.app import app
    return TestClient(app)


def test_health():
    assert client().get("/health").json()["status"] == "ok"


def test_predict_shape():
    r = client().post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1


def test_bad_input_rejected():
    assert client().post("/predict", json={"age": "x"}).status_code == 422


def test_metrics_exposed():
    client().post("/predict", json=SAMPLE)
    assert "predict_requests_total" in client().get("/metrics").text


def test_model_info_and_health():
    c = client()
    assert c.get("/health").json()["model_loaded"] is True
    info = c.get("/model/info").json()
    assert info["model"]
    assert "metrics" in info
