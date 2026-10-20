import numpy as np

from src.data import (CAT, COLS, NUM, build_preprocessor, clean, split_xy,
                      validate_domains)

SAMPLE = """63,1,1,145,233,1,2,150,0,2.3,3,0,6,0
67,1,4,160,286,0,2,108,1,1.5,2,3,3,2
41,0,2,130,204,0,2,172,0,1.4,1,0,3,0
56,1,2,120,236,0,0,178,0,0.8,1,?,3,0
"""


def make_df(tmp_path):
    f = tmp_path / "raw.data"
    f.write_text(SAMPLE)
    return clean(f)


def test_clean_binary_target_and_columns(tmp_path):
    df = make_df(tmp_path)
    assert list(df.columns) == COLS + ["source"]
    assert set(df.target.unique()) <= {0, 1}
    assert df.target.tolist() == [0, 1, 0, 0]


def test_missing_marker_becomes_nan(tmp_path):
    df = make_df(tmp_path)
    assert df["ca"].isna().sum() == 1


def test_preprocessor_handles_missing_and_scales(tmp_path):
    X, _ = split_xy(make_df(tmp_path))
    out = build_preprocessor().fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert not np.isnan(out).any()
    assert abs(out[:, : len(NUM)].mean()) < 1e-6
    assert out.shape[0] == 4


def test_feature_lists_cover_all_inputs():
    assert set(NUM + CAT) == set(COLS) - {"target"}


def test_merge_all_processed_files_with_source(tmp_path):
    (tmp_path / "processed.a.data").write_text(SAMPLE)
    (tmp_path / "processed.b.data").write_text(
        "50,1,3,0,0,0,0,140,0,1.0,2,?,?,3\n")
    df = clean(tmp_path)
    assert len(df) == 5
    assert set(df.source) == {"a", "b"}
    b = df[df.source == "b"].iloc[0]
    assert b.target == 1
    assert np.isnan(b.chol) and np.isnan(b.trestbps)


def test_duplicates_are_kept_and_minus9_is_missing(tmp_path):
    row = "60,1,4,130,250,0,0,150,0,1.0,2,0,3,1\n"
    (tmp_path / "processed.a.data").write_text(row + row + "61,1,4,-9,250,0,0,150,0,1.0,2,0,3,0\n")
    df = clean(tmp_path)
    assert len(df) == 3
    assert df.trestbps.isna().sum() == 1


def test_validate_domains_blanks_invalid_codes(tmp_path):
    df = make_df(tmp_path)
    df.loc[0, "cp"] = 9
    df.loc[1, "thal"] = 5
    df.loc[2, "sex"] = 2
    out, bad = validate_domains(df)
    assert bad == 3
    assert out.loc[0, "cp"] != out.loc[0, "cp"]
    assert out.loc[3, "cp"] == df.loc[3, "cp"]
