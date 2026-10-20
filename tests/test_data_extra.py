import numpy as np
import pytest

from src.data import (CAT, COLS, DOCUMENTED_ROWS, DOMAINS, NUM,
                      build_preprocessor, quality_report, source_name,
                      split_xy, validate_domains)
from tests.conftest import synthetic


def test_documented_rows_add_up_to_920():
    assert sum(DOCUMENTED_ROWS.values()) == 920


def test_every_domain_column_is_a_categorical_feature():
    assert set(DOMAINS) == set(CAT)


@pytest.mark.parametrize("name,expected", [
    ("processed.cleveland.data", "cleveland"),
    ("processed.va.data", "va"),
    ("other.csv", "other"),
])
def test_source_name(name, expected):
    assert source_name(name) == expected


def test_validate_domains_keeps_valid_values_and_existing_nan():
    df = synthetic()
    out, bad = validate_domains(df)
    assert bad == 0
    assert out["ca"].isna().sum() == df["ca"].isna().sum()
    assert out[CAT].fillna(-1).equals(df[CAT].fillna(-1))


def test_validate_domains_does_not_change_the_input():
    df = synthetic()
    df.loc[0, "slope"] = 7
    validate_domains(df)
    assert df.loc[0, "slope"] == 7


def test_validate_domains_checks_each_column_against_its_own_codes():
    df = synthetic()
    df.loc[0, "thal"] = 3.0   # valid for thal
    df.loc[1, "ca"] = 3.0     # valid for ca
    df.loc[2, "ca"] = 4.0     # invalid for ca
    out, bad = validate_domains(df)
    assert bad == 1
    assert np.isnan(out.loc[2, "ca"])
    assert out.loc[1, "ca"] == 3.0


def test_split_xy_returns_model_columns_in_fixed_order():
    X, y = split_xy(synthetic())
    assert list(X.columns) == NUM + CAT
    assert len(X) == len(y)
    assert "target" not in X.columns and "source" not in X.columns


def test_quality_report_counts():
    df = synthetic()
    rep = quality_report(df)
    assert rep["rows"] == len(df)
    assert sum(rep["rows_per_source"].values()) == len(df)
    assert rep["missing"]["ca"]["count"] == 3
    assert rep["missing"]["chol"]["count"] == 2
    assert "age" not in rep["missing"]
    assert rep["missing"]["ca"]["pct"] == round(100 * 3 / len(df), 1)
    assert set(rep["numeric_summary"]) == set(NUM)


def test_quality_report_positives_match_target():
    df = synthetic()
    rep = quality_report(df)
    assert sum(rep["positives_per_source"].values()) == int(df.target.sum())


def test_preprocessor_median_fills_numeric_gaps():
    df = synthetic()
    X, _ = split_xy(df)
    prep = build_preprocessor().fit(X)
    imputer = prep.named_transformers_["num"].named_steps["imp"]
    chol_median = df["chol"].median()
    assert imputer.statistics_[NUM.index("chol")] == chol_median


def test_missing_category_gets_its_own_column():
    X, _ = split_xy(synthetic())
    prep = build_preprocessor().fit(X)
    names = prep.get_feature_names_out()
    assert "cat__ca_-1.0" in names


def test_preprocessor_output_is_fully_numeric_and_same_row_count():
    X, _ = split_xy(synthetic())
    out = build_preprocessor().fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert out.shape[0] == len(X)
    assert np.isfinite(out).all()


def test_preprocessor_needs_all_columns():
    X, _ = split_xy(synthetic())
    prep = build_preprocessor().fit(X)
    with pytest.raises(ValueError):
        prep.transform(X.drop(columns="chol"))


def test_cols_has_14_columns_ending_in_target():
    assert len(COLS) == 14 and COLS[-1] == "target"
