"""FastAPI service: /predict, /health, /metrics."""
import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel

from src.data import CAT, NUM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("heart-api")

MODEL_PATH = Path("models/model.joblib")
META_PATH = Path("models/model_meta.json")
_cache = {"model": None, "mtime": None}


def get_model():
    """Load the model; reload if train.py wrote a newer file; 503 if missing."""
    if not MODEL_PATH.exists():
        raise HTTPException(503, "No trained model. Run python -m src.train first.")
    mtime = MODEL_PATH.stat().st_mtime
    if _cache["mtime"] != mtime:
        _cache["model"], _cache["mtime"] = joblib.load(MODEL_PATH), mtime
    return _cache["model"]


app = FastAPI(title="Heart Disease Risk API")

REQS = Counter("predict_requests_total", "Predict calls", ["outcome"])
LAT = Histogram("predict_latency_seconds", "Predict latency")


class Patient(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float | None = None
    thal: float | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/model/info")
def model_info():
    get_model()
    return json.loads(META_PATH.read_text()) if META_PATH.exists() else {}


@app.post("/predict")
def predict(p: Patient):
    start = time.time()
    try:
        row = pd.DataFrame([p.model_dump()]).astype(float)[NUM + CAT]
        proba = float(get_model().predict_proba(row)[0, 1])
        out = {"prediction": int(proba >= 0.5), "confidence": round(
            max(proba, 1 - proba), 4), "probability_disease": round(proba, 4)}
        REQS.labels("ok").inc()
        log.info("predict input=%s output=%s", p.model_dump(), out)
        return out
    except HTTPException:
        raise
    except Exception:
        REQS.labels("error").inc()
        log.exception("predict failed")
        raise
    finally:
        LAT.observe(time.time() - start)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
