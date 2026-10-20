from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import build_preprocessor, split_xy
from tests.test_data import make_df


def test_pipeline_trains_and_predicts_probabilities(tmp_path):
    df = make_df(tmp_path)
    X, y = split_xy(df)
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=200))]).fit(X, y)
    p = pipe.predict_proba(X)
    assert p.shape == (4, 2)
    assert ((p >= 0) & (p <= 1)).all()


def test_unseen_category_does_not_crash(tmp_path):
    df = make_df(tmp_path)
    X, y = split_xy(df)
    pipe = Pipeline([("prep", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=200))]).fit(X, y)
    X2 = X.iloc[[0]].copy()
    X2["cp"] = 9.0
    assert pipe.predict(X2).shape == (1,)
