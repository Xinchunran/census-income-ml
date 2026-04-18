from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from model.baseline.dataio.dataset import TARGET_COL, clean_label_to_binary


def _build_stratification_key(df: pd.DataFrame) -> pd.Series:
    y = clean_label_to_binary(df[TARGET_COL]).astype(str)
    year = df["year"].astype(str).str.strip()
    return y + "_" + year


def _build_groups(df: pd.DataFrame) -> np.ndarray:
    # If a persistent row id exists, use it as group id to avoid accidental leakage.
    if "_row_id" in df.columns:
        return df["_row_id"].to_numpy()
    return np.arange(len(df), dtype=int)


def make_splits(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Create train/valid/test using StratifiedGroupKFold with no row drops."""
    work = df.copy()
    y_key = _build_stratification_key(work)
    groups = _build_groups(work)

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    outer_train_idx, test_idx = next(outer.split(work, y_key, groups))

    train_valid = work.iloc[outer_train_idx].copy()
    y_tv = y_key.iloc[outer_train_idx]
    groups_tv = groups[outer_train_idx]

    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    tr_rel, va_rel = next(inner.split(train_valid, y_tv, groups_tv))

    train_df = train_valid.iloc[tr_rel].copy()
    valid_df = train_valid.iloc[va_rel].copy()
    test_df = work.iloc[test_idx].copy()
    return {"train": train_df, "valid": valid_df, "test": test_df}
