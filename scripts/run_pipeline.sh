#!/usr/bin/env bash
# Runs the whole local flow in order and stops at the first error.
# Usage (from the project folder, venv active): bash scripts/run_pipeline.sh
set -euo pipefail

echo "== 1/6 data";      python -m src.data
echo "== 2/6 eda";       python -m src.eda
echo "== 3/6 train";     python -m src.train
echo "== 4/6 lint";      flake8 .
echo "== 5/6 tests";     python -m pytest -q --cov=src
echo "== 6/6 sample prediction"
python scripts/predict.py samples/high_risk.json
echo "Done. Next: mlflow ui --backend-store-uri sqlite:///mlflow.db"
