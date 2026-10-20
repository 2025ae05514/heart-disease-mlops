"""Train, tune and track models with MLflow; save the best pipeline."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import mlflow 
import mlflow.sklearn 
import pandas as pd 
import yaml 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.base import clone 
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, 
                             accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, 
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline 

from src.data import CLEAN, build_preprocessor, split_xy 

CONFIG = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "configs" / "train.yaml").read_text())
SEED = CONFIG["seed"]
MODELS = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000), CONFIG["logistic_regression"]),
    "random_forest": (
        RandomForestClassifier(random_state=SEED), CONFIG["random_forest"]),
}


def loso_auc(model, df):
    """Leave-one-source-out ROC-AUC: train on 3 sources, test on the 4th."""
    X, y = split_xy(df)
    out = {}
    for s in sorted(df.source.unique()):
        te = (df.source == s).values
        m = clone(model).fit(X[~te], y[~te])
        out[s] = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
    return out


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(CONFIG["experiment"])
    df = pd.read_csv(CLEAN)
    X, y = split_xy(df)
    strat = df["source"] + df["target"].astype(str)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=CONFIG["test_size"], stratify=strat, random_state=SEED)
    cv = StratifiedKFold(CONFIG["cv_folds"], shuffle=True, random_state=SEED)
    scoring = ["accuracy", "precision", "recall", "roc_auc"]
    best_auc, best_pipe, best_name, best_metrics = -1, None, None, None

    for name, (clf, grid) in MODELS.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        with mlflow.start_run(run_name=name):
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc").fit(Xtr, ytr)
            model = gs.best_estimator_
            cvres = cross_validate(model, Xtr, ytr, cv=cv, scoring=scoring)
            pred = model.predict(Xte)
            proba = model.predict_proba(Xte)[:, 1]
            metrics = {
                **{f"cv_{m}": cvres[f"test_{m}"].mean() for m in scoring},
                "test_accuracy": accuracy_score(yte, pred),
                "test_precision": precision_score(yte, pred),
                "test_recall": recall_score(yte, pred),
                "test_roc_auc": roc_auc_score(yte, proba),
            }
            src_te = df.loc[Xte.index, "source"]
            for s in sorted(src_te.unique()):
                k = (src_te == s).values
                if yte[k].nunique() == 2:
                    metrics[f"test_roc_auc_{s}"] = roc_auc_score(
                        yte[k], proba[k])
            for s, v in loso_auc(model, df).items():
                metrics[f"loso_roc_auc_{s}"] = v
            metrics["loso_roc_auc_mean"] = sum(
                v for k, v in metrics.items()
                if k.startswith("loso_roc_auc_") and k != "loso_roc_auc_mean"
            ) / df.source.nunique()
            mlflow.log_params({k.replace("clf__", ""): v
                               for k, v in gs.best_params_.items()})
            mlflow.log_param("model", name)
            mlflow.log_params({k: CONFIG[k] for k in ("seed", "test_size", "cv_folds")})
            mlflow.log_metrics(metrics)
            for kind, disp in [("roc", RocCurveDisplay),
                               ("confusion", ConfusionMatrixDisplay)]:
                fig, ax = plt.subplots()
                if kind == "roc":
                    disp.from_predictions(yte, proba, ax=ax)
                else:
                    disp.from_predictions(yte, pred, ax=ax)
                mlflow.log_figure(fig, f"{kind}_{name}.png")
                plt.close(fig)
            mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
            print(name, {k: round(float(v), 3) for k, v in metrics.items()})
            if metrics["cv_roc_auc"] > best_auc:
                best_auc = metrics["cv_roc_auc"]
                best_pipe, best_name = model, name
                best_metrics = {k: round(float(v), 4) for k, v in metrics.items()}

    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_pipe, "models/model.joblib")
    meta = {"model": best_name,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": int(len(df)), "features": list(X.columns),
            "metrics": best_metrics}
    Path("models/model_meta.json").write_text(json.dumps(meta, indent=2))
    print("best model:", best_name, "cv_roc_auc:", round(best_auc, 3))


if __name__ == "__main__":
    main()
