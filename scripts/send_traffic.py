"""Send sample requests to /predict so Prometheus and Grafana have data.

Usage: python scripts/send_traffic.py [--url http://localhost:8000] [--n 100]
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://localhost:8000")
p.add_argument("--n", type=int, default=100)
p.add_argument("--bad", type=int, default=5, help="invalid requests to also send")
a = p.parse_args()

BASE = {"age": 55, "sex": 1, "cp": 3, "trestbps": 130, "chol": 240, "fbs": 0,
        "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 2,
        "ca": 0, "thal": 3}


def post(body):
    req = urllib.request.Request(a.url + "/predict", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=10).status
    except urllib.error.HTTPError as e:
        return e.code


codes = {}
for i in range(a.n):
    row = dict(BASE, age=random.randint(30, 77), chol=random.randint(150, 350),
               thalach=random.randint(90, 200), oldpeak=round(random.uniform(0, 4), 1))
    if i < a.bad:
        row["age"] = "x"  # fails validation (422)
    c = post(row)
    codes[c] = codes.get(c, 0) + 1
    time.sleep(0.05)
print("status codes:", codes)
