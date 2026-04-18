from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_COL = "label"
WEIGHT_COL = "weight"
POSITIVE_LABEL = "50000+."


def load_raw_data(data_path: str | Path, columns_path: str | Path) -> pd.DataFrame:
    """Load raw census table with schema from the columns file."""
    data_path = Path(data_path)
    columns_path = Path(columns_path)
    columns = [line.strip() for line in columns_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.read_csv(data_path, header=None, names=columns)


def clean_label_to_binary(series: pd.Series) -> pd.Series:
    labels = series.astype(str).str.strip()
    return (labels == POSITIVE_LABEL).astype(int)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {TARGET_COL, WEIGHT_COL, "_row_id"}]
