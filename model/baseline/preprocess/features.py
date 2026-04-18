from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from model.baseline.dataio.dataset import TARGET_COL, WEIGHT_COL


@dataclass
class CensusPreprocessor:
    numeric_cols: list[str]
    categorical_cols: list[str]
    transformer: ColumnTransformer

    def transform(self, df: pd.DataFrame):
        feature_df = df.drop(columns=[c for c in [TARGET_COL, WEIGHT_COL, "_row_id"] if c in df.columns], errors="ignore")
        return self.transformer.transform(feature_df)


def _safe_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


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


def build_preprocessor(train_df: pd.DataFrame) -> CensusPreprocessor:
    numeric_cols, categorical_cols = infer_column_types(train_df)

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _safe_one_hot_encoder()),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    feature_df = train_df.drop(columns=[c for c in [TARGET_COL, WEIGHT_COL, "_row_id"] if c in train_df.columns], errors="ignore")
    transformer.fit(feature_df)
    return CensusPreprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols, transformer=transformer)
