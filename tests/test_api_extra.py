import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import src.app as app_mod
from src.data import build_preprocessor, split_xy
from tests.conftest import synthetic

SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
          "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
          "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}


@pytest.fixture
def api(tmp_path, monkeypatch):
    """API pointed at a tiny model in a temp folder, so no training is needed."""
    X, y = split_xy(synthetic())
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "model_meta.json"
    joblib.dump(pipe, model_path)
    meta_path.write_text(json.dumps({"model": "logistic_regression",
                                     "metrics": {"cv_roc_auc": 0.9}}))
    monkeypatch.setattr(app_mod, "MODEL_PATH", model_path)
    monkeypatch.setattr(app_mod, "META_PATH", meta_path)
    monkeypatch.setitem(app_mod._cache, "mtime", None)
    monkeypatch.setitem(app_mod._cache, "model", None)
    return TestClient(app_mod.app)


def test_predict_fields_are_consistent(api):
    body = api.post("/predict", json=SAMPLE).json()
    p = body["probability_disease"]
    assert body["prediction"] == int(p >= 0.5)
    assert body["confidence"] == round(max(p, 1 - p), 4)
    assert 0 <= p <= 1


def test_predict_works_without_ca_and_thal(api):
    row = {k: v for k, v in SAMPLE.items() if k not in ("ca", "thal")}
    assert api.post("/predict", json=row).status_code == 200


def test_predict_rejects_missing_required_field(api):
    row = {k: v for k, v in SAMPLE.items() if k != "age"}
    r = api.post("/predict", json=row)
    assert r.status_code == 422
    assert "age" in r.text


def test_predict_rejects_wrong_type_and_empty_body(api):
    assert api.post("/predict", json=dict(SAMPLE, chol="high")).status_code == 422
    assert api.post("/predict", json={}).status_code == 422


def test_predict_rejects_get(api):
    assert api.get("/predict").status_code == 405


def test_predict_is_repeatable_for_same_input(api):
    a = api.post("/predict", json=SAMPLE).json()
    b = api.post("/predict", json=SAMPLE).json()
    assert a == b


def test_metrics_counts_ok_requests(api):
    def count():
        for line in api.get("/metrics").text.splitlines():
            if line.startswith('predict_requests_total{outcome="ok"}'):
                return float(line.split()[-1])
        return 0.0
    before = count()
    api.post("/predict", json=SAMPLE)
    api.post("/predict", json=SAMPLE)
    assert count() == before + 2


def test_metrics_has_latency_histogram(api):
    api.post("/predict", json=SAMPLE)
    text = api.get("/metrics").text
    assert "predict_latency_seconds_bucket" in text
    assert "predict_latency_seconds_count" in text


def test_422_requests_are_not_counted_as_predictions(api):
    def total():
        return sum(float(line.split()[-1]) for line in
                   api.get("/metrics").text.splitlines()
                   if line.startswith("predict_requests_total{"))
    before = total()
    api.post("/predict", json={})
    assert total() == before


def test_health_and_model_info(api):
    h = api.get("/health").json()
    assert h == {"status": "ok", "model_loaded": True}
    info = api.get("/model/info").json()
    assert info["model"] == "logistic_regression"
    assert info["metrics"]["cv_roc_auc"] == 0.9


def test_model_info_is_empty_when_meta_file_missing(api):
    app_mod.META_PATH.unlink()
    assert api.get("/model/info").json() == {}


def test_missing_model_gives_503_not_a_crash(api, tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "MODEL_PATH", tmp_path / "nope.joblib")
    assert api.get("/health").json()["model_loaded"] is False
    assert api.post("/predict", json=SAMPLE).status_code == 503
    assert api.get("/model/info").status_code == 503


def test_model_reloads_when_file_changes(api):
    api.post("/predict", json=SAMPLE)
    first = app_mod._cache["model"]
    X, y = split_xy(synthetic(seed=3))
    new = Pipeline([("prep", build_preprocessor()),
                    ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
    joblib.dump(new, app_mod.MODEL_PATH)
    import os
    st = app_mod.MODEL_PATH.stat()
    os.utime(app_mod.MODEL_PATH, (st.st_atime, st.st_mtime + 5))
    api.post("/predict", json=SAMPLE)
    assert app_mod._cache["model"] is not first
