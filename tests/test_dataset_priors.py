import pandas as pd

DATA_PATH = "census-bureau.data"
COLUMNS_PATH = "census-bureau.columns"


def load_raw(data_path=DATA_PATH, columns_path=COLUMNS_PATH):
    cols = [line.strip() for line in open(columns_path) if line.strip()]
    df = pd.read_csv(data_path, header=None, names=cols)
    return df


def test_raw_shape_and_schema():
    df = load_raw()
    assert df.shape[0] == 199_523, "Unexpected row count; parsing may be broken."
    assert df.shape[1] == 42, "Unexpected column count; header alignment may be broken."
    assert df.columns[-2] == "year"
    assert df.columns[-1] == "label"
    assert "weight" in df.columns


def test_label_values_and_base_rate():
    df = load_raw()
    labels = set(df["label"].astype(str).str.strip().unique())
    assert labels == {"- 50000.", "50000+."}, f"Unexpected labels: {labels}"

    y = (df["label"].astype(str).str.strip() == "50000+.").astype(int)
    raw_pos_rate = y.mean()
    assert 0.055 <= raw_pos_rate <= 0.070, (
        f"Positive rate {raw_pos_rate:.4f} is outside the expected band; check label parsing."
    )

    w = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
    weighted_pos_rate = (w * y).sum() / w.sum()
    assert 0.058 <= weighted_pos_rate <= 0.070, (
        f"Weighted positive rate {weighted_pos_rate:.4f} is outside the expected band."
    )


def test_year_distribution():
    df = load_raw()
    year_counts = df["year"].value_counts().to_dict()
    assert set(year_counts) == {94, 95}, f"Unexpected survey years: {year_counts}"
    share_94 = year_counts[94] / len(df)
    assert 0.45 <= share_94 <= 0.55, "94/95 year balance drifted unexpectedly."


def test_structural_unknown_rates_are_preserved():
    df = load_raw()
    unknown_cols = [
        "migration code-change in msa",
        "migration code-change in reg",
        "migration code-move within reg",
        "migration prev res in sunbelt",
    ]
    for col in unknown_cols:
        q_rate = (df[col].astype(str).str.strip() == "?").mean()
        assert 0.45 <= q_rate <= 0.55, (
            f"Column '{col}' has '?' rate {q_rate:.4f}; unknown values may have been altered or dropped."
        )


def test_sparse_financial_features_are_not_misparsed():
    df = load_raw()
    gains_nz = (pd.to_numeric(df["capital gains"], errors="coerce").fillna(0) > 0).mean()
    losses_nz = (pd.to_numeric(df["capital losses"], errors="coerce").fillna(0) > 0).mean()
    divs_nz = (pd.to_numeric(df["dividends from stocks"], errors="coerce").fillna(0) > 0).mean()

    assert 0.02 <= gains_nz <= 0.06, "Capital gains sparsity looks wrong."
    assert 0.01 <= losses_nz <= 0.04, "Capital losses sparsity looks wrong."
    assert 0.07 <= divs_nz <= 0.15, "Dividend sparsity looks wrong."
