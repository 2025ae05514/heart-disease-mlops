"""Predict from a JSON file with the saved model (no API needed).

Usage: python scripts/predict.py samples/high_risk.json
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import CAT, NUM  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "samples/high_risk.json"
patient = json.loads(Path(path).read_text())
row = pd.DataFrame([patient]).reindex(columns=NUM + CAT).astype(float)
model = joblib.load("models/model.joblib")
p = float(model.predict_proba(row)[0, 1])
print(json.dumps({"input_file": path, "prediction": int(p >= 0.5),
                  "probability_disease": round(p, 4)}, indent=2))
