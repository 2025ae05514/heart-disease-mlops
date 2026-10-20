# Heart Disease Risk - End-to-End MLOps

Binary classifier on the UCI Heart Disease all four states data, tracked with MLflow, tested and built by GitHub Actions, served by FastAPI in Docker, deployed on Kubernetes, monitored with Prometheus metrics.

## Setup
```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m src.data              # downloads all 4 processed.* UCI files, merges, cleans -> data/processed/heart_clean.csv
python -m src.eda               # figures -> reports/figures
python -m src.train             # trains LR + RF, logs to MLflow, saves models/model.joblib
pytest -v --cov=src && flake8 .
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
uvicorn src.app:app --port 8000
```
## Scripts
`bash scripts/run_pipeline.sh` runs data, eda, train, lint, tests and a sample prediction in order. `python scripts/predict.py samples/low_risk.json` predicts from a file without the API.

## Notebooks
Optional, for demos: `pip install notebook`, then open notebooks/ and run 01_eda, 02_training_mlflow, 03_inference in order. Each notebook switches to the project folder and gets what it needs itself: the data if missing (01, 02, 03) and the saved model if missing (03). No `python -m src.data` first.

## Docker
```
docker build -t heart-api:latest .
docker run -p 8000:8000 heart-api:latest
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @samples/high_risk.json   # also samples/low_risk.json, samples/invalid.json (422)
```
## Kubernetes (Minikube / Docker Desktop)
```
kubectl apply -f k8s/
kubectl get pods,svc
minikube tunnel        # Minikube only, gives the LoadBalancer an IP
```
## Monitoring
`/metrics` exposes `predict_requests_total` and `predict_latency_seconds`. `docker compose up` starts the API, Prometheus (:9090) and Grafana (:3000).
Request logs are printed by the API (one line per prediction).
Grafana loads the "Heart Disease API" dashboard on start (login admin/admin). Fill it with data: `python scripts/send_traffic.py --url http://localhost:8000 --n 100`.
Extra endpoints: `/health` (also says if a model is loaded), `/model/info` (best model, metrics, training time).

## Layout
src/ code, notebooks/ demo notebooks, tests/ pytest, k8s/ manifests, .github/workflows/ci.yml, monitoring/, reports/figures, screenshots/
