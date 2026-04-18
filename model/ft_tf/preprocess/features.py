from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from model.ft_tf.dataio.dataset import TARGET_COL, WEIGHT_COL, get_feature_columns


@dataclass
class FTPreprocessor:
    numeric_cols: list[str]
    categorical_cols: list[str]
    numeric_fill_values: dict[str, float]
    categorical_maps: dict[str, dict[str, int]]

    @property
    def cat_cardinalities(self) -> list[int]:
        return [len(self.categorical_maps[col]) + 1 for col in self.categorical_cols]

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
        X = df[get_feature_columns(df)].copy()
        x_num = None
        x_cat = None

        if self.numeric_cols:
            num_frame = pd.DataFrame(index=X.index)
            for col in self.numeric_cols:
                num_frame[col] = pd.to_numeric(X[col], errors="coerce").fillna(self.numeric_fill_values[col])
            x_num = num_frame.to_numpy(dtype=np.float32)

        if self.categorical_cols:
            cat_frame = pd.DataFrame(index=X.index)
            for col in self.categorical_cols:
                mapper = self.categorical_maps[col]
                values = X[col].astype(str).fillna("__NA__")
                cat_frame[col] = values.map(mapper).fillna(0).astype(np.int64)
            x_cat = cat_frame.to_numpy(dtype=np.int64)

        return x_num, x_cat

    def to_state(self) -> dict[str, Any]:
        return {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "numeric_fill_values": self.numeric_fill_values,
            "categorical_maps": self.categorical_maps,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "FTPreprocessor":
        return cls(
            numeric_cols=list(state["numeric_cols"]),
            categorical_cols=list(state["categorical_cols"]),
            numeric_fill_values={k: float(v) for k, v in state["numeric_fill_values"].items()},
            categorical_maps={k: {kk: int(vv) for kk, vv in m.items()} for k, m in state["categorical_maps"].items()},
        )


def infer_column_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_df = df.drop(columns=[c for c in [TARGET_COL, WEIGHT_COL, "_row_id"] if c in df.columns], errors="ignore")
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in feature_df.columns:
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            numeric_cols.append(col)
            continue
        numeric_ratio = pd.to_numeric(feature_df[col], errors="coerce").notna().mean()
        if numeric_ratio > 0.98:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols


def build_preprocessor(train_df: pd.DataFrame) -> FTPreprocessor:
    numeric_cols, categorical_cols = infer_column_types(train_df)
    X = train_df[get_feature_columns(train_df)].copy()

    numeric_fill_values: dict[str, float] = {}
    for col in numeric_cols:
        numeric_fill_values[col] = float(pd.to_numeric(X[col], errors="coerce").median())

    categorical_maps: dict[str, dict[str, int]] = {}
    for col in categorical_cols:
        values = X[col].astype(str).fillna("__NA__").value_counts().index.tolist()
        # reserve 0 for unknown values at inference
        categorical_maps[col] = {v: i + 1 for i, v in enumerate(values)}

    return FTPreprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_fill_values=numeric_fill_values,
        categorical_maps=categorical_maps,
    )
