import numpy as np
import pandas as pd

try:
    from project_api import load_raw_data, make_splits
except ImportError as e:
    raise ImportError(
        "Edit this test to import your own project functions, or implement project_api.py."
    ) from e


def _positive_rate(df):
    return (df["label"].astype(str).str.strip() == "50000+.").mean()


def _weighted_positive_rate(df):
    y = (df["label"].astype(str).str.strip() == "50000+.").astype(int)
    w = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
    return (w * y).sum() / w.sum()


def test_split_preserves_population_characteristics():
    df = load_raw_data("census-bureau.data", "census-bureau.columns").copy()
    df["_row_id"] = np.arange(len(df))

    splits = make_splits(df, seed=42)
    assert {"train", "valid", "test"}.issubset(splits.keys())

    train_df = splits["train"]
    valid_df = splits["valid"]
    test_df = splits["test"]

    # No overlap
    train_ids = set(train_df["_row_id"])
    valid_ids = set(valid_df["_row_id"])
    test_ids = set(test_df["_row_id"])
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)

    # Combined coverage
    assert len(train_ids | valid_ids | test_ids) == len(df)

    # Base-rate preservation
    overall_rate = _positive_rate(df)
    for name, split_df in splits.items():
        rate = _positive_rate(split_df)
        assert abs(rate - overall_rate) <= 0.01, (
            f"{name} positive rate drift is too large: {rate:.4f} vs {overall_rate:.4f}"
        )

    # Weighted base-rate preservation
    overall_w_rate = _weighted_positive_rate(df)
    for name, split_df in splits.items():
        rate = _weighted_positive_rate(split_df)
        assert abs(rate - overall_w_rate) <= 0.01, (
            f"{name} weighted positive rate drift is too large: {rate:.4f} vs {overall_w_rate:.4f}"
        )

    # Year preservation matters because the raw data mixes 1994 and 1995.
    overall_year = df["year"].value_counts(normalize=True).sort_index()
    for name, split_df in splits.items():
        year_share = split_df["year"].value_counts(normalize=True).sort_index()
        aligned = overall_year.align(year_share, fill_value=0)
        max_drift = np.max(np.abs(aligned[0].values - aligned[1].values))
        assert max_drift <= 0.03, f"{name} year distribution drift is too large ({max_drift:.4f})."
