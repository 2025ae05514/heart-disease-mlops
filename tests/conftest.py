import numpy as np
import pandas as pd

from src.data import CAT, NUM


def synthetic(n=120, seed=0):
    """Small fake merged table: 2 sources, both classes, a few missing values."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(100, 180, n),
        "chol": rng.integers(150, 350, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(90, 200, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.uniform(0, 4, n).round(1),
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    }).astype(float)
    # label depends on two features plus noise, so a model can learn a little
    score = (df.cp >= 3) * 1.0 + (df.thalach < 140) * 1.0 + rng.normal(0, 0.7, n)
    df["target"] = (score > 1).astype(int)
    df["source"] = ["a", "b"] * (n // 2)
    df.loc[[1, 5, 9], "ca"] = np.nan
    df.loc[[2, 6], "chol"] = np.nan
    assert set(NUM + CAT) <= set(df.columns)
    return df
