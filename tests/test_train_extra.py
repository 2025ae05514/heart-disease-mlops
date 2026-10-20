import json

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import src.train as train
from src.data import CAT, NUM, build_preprocessor, split_xy
from tests.conftest import synthetic


def _pipe():
    return Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=500))])


def test_loso_gives_one_auc_per_source_between_0_and_1():
    df = synthetic()
    out = train.loso_auc(_pipe(), df)
    assert set(out) == {"a", "b"}
    assert all(0 <= v <= 1 for v in out.values())


def test_loso_does_not_train_on_the_held_out_source():
    df = synthetic()
    df.loc[df.source == "b", "target"] = 1   # source b has one class only
    with pytest.raises(ValueError):          # AUC is undefined on one class
        train.loso_auc(_pipe(), df)


def test_models_are_the_two_required_and_grid_keys_match_pipeline():
    assert set(train.MODELS) == {"logistic_regression", "random_forest"}
    for clf, grid in train.MODELS.values():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        assert all(k in pipe.get_params() for k in grid)


def test_train_main_end_to_end_on_small_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/processed").mkdir(parents=True)
    synthetic(160).to_csv(train.CLEAN, index=False)
    monkeypatch.setattr(train, "MODELS", {
        "logistic_regression": (LogisticRegression(max_iter=500),
                                {"clf__C": [1]})})
    train.main()
    model = joblib.load("models/model.joblib")
    X, _ = split_xy(synthetic(160))
    proba = model.predict_proba(X.head(3))
    assert proba.shape == (3, 2)
    meta = json.load(open("models/model_meta.json"))
    assert meta["model"] == "logistic_regression"
    assert meta["rows"] == 160
    assert meta["features"] == NUM + CAT
    assert 0 <= meta["metrics"]["cv_roc_auc"] <= 1
    assert "loso_roc_auc_mean" in meta["metrics"]
    assert (tmp_path / "mlflow.db").exists()
    assert pd.read_csv(train.CLEAN).shape[0] == 160


def test_config_has_the_settings_train_needs():
    c = train.CONFIG
    assert c["seed"] == train.SEED
    assert 0 < c["test_size"] < 1
    assert c["cv_folds"] >= 2
    assert set(train.MODELS) <= set(c)
    for _, grid in train.MODELS.values():
        assert all(isinstance(v, list) and v for v in grid.values())
