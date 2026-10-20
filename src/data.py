"""Data download, cleaning and the preprocessing pipeline."""
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
RAW_DIR = Path("data/raw")
CLEAN = Path("data/processed/heart_clean.csv")
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target"]
NUM = ["age", "trestbps", "chol", "thalach", "oldpeak"]
DOMAINS = {  # codes documented in heart-disease.names
    "sex": {0, 1}, "cp": {1, 2, 3, 4}, "fbs": {0, 1}, "restecg": {0, 1, 2},
    "exang": {0, 1}, "slope": {1, 2, 3}, "ca": {0, 1, 2, 3}, "thal": {3, 6, 7}}
DOCUMENTED_ROWS = {"cleveland": 303, "hungarian": 294, "switzerland": 123,
                   "va": 200}
CAT = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def download(dest_dir=RAW_DIR):
    """Download the UCI zip and keep every processed.*.data file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    names = [n for n in z.namelist()
             if n.startswith("processed.") and n.endswith(".data")]
    for n in names:
        (dest_dir / n).write_bytes(z.read(n))
    return sorted(names)


def source_name(path):
    name = Path(path).name
    if name.startswith("processed."):
        return name.split(".")[1]
    return Path(path).stem


def clean(raw=RAW_DIR):
    """Merge all processed.* files into one frame with a source column."""
    raw = Path(raw)
    files = sorted(raw.glob("processed.*.data")) if raw.is_dir() else [raw]
    frames = []
    for f in files:
        part = pd.read_csv(f, header=None, names=COLS,
                           na_values=["?", "-9", "-9.0"])
        part["source"] = source_name(f)
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    # 0 is not a real value for these two columns (it marks "missing"
    # in the Switzerland and VA files)
    for col in ["chol", "trestbps"]:
        df[col] = df[col].replace(0, np.nan)
    df["target"] = (df["target"] > 0).astype(int)
    return df.reset_index(drop=True)


def validate_domains(df):
    """Set values outside the documented codes to NaN. Returns (df, n_bad)."""
    df = df.copy()
    bad = 0
    for col, ok in DOMAINS.items():
        mask = df[col].notna() & ~df[col].isin(ok)
        bad += int(mask.sum())
        df.loc[mask, col] = np.nan
    return df, bad


def quality_report(df):
    """Plain dict describing the merged data (shown in logs and saved as JSON)."""
    feats = df.drop(columns="source")
    miss = feats.isna().sum()
    return {
        "rows": int(len(df)),
        "rows_per_source": df.source.value_counts().to_dict(),
        "positives_per_source": df.groupby("source").target.sum().astype(int).to_dict(),
        "dtypes": {c: str(t) for c, t in feats.dtypes.items()},
        "missing": {c: {"count": int(n), "pct": round(100 * n / len(df), 1)}
                    for c, n in miss.items() if n},
        "numeric_summary": feats[NUM].describe().round(2).to_dict(),
    }


def build_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=-1)),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])


def split_xy(df):
    return df[NUM + CAT], df["target"]


def prepare():
    print("downloaded:", download())
    df = clean()
    counts = df.source.value_counts().to_dict()
    assert counts == DOCUMENTED_ROWS, f"row counts differ from .names: {counts}"
    df, bad = validate_domains(df)
    print("values outside documented codes set to NaN:", bad)
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)
    print("rows:", len(df), "missing:", int(df.drop(columns="source").isna().sum().sum()),
          "positives:", int(df.target.sum()))
    report = quality_report(df)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(json.dumps(report, indent=2))
    for col, m in report["missing"].items():
        print(f"missing {col}: {m['count']} ({m['pct']}%)")
    print(df.groupby("source").agg(rows=("target", "size"), positives=("target", "sum")))


if __name__ == "__main__":
    prepare()
