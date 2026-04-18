from __future__ import annotations

import numpy as np
import pandas as pd

from model.baseline.dataio import dataset as dataset_io
from model.baseline.models.classification import predict_logits as baseline_predict_logits
from model.baseline.models.classification import run_group_kfold_baseline
from model.baseline.models.segmentation import assign_segments as baseline_assign_segments
from model.baseline.models.segmentation import fit_segmenter as baseline_fit_segmenter
from model.baseline.preprocess.features import build_preprocessor as baseline_build_preprocessor
from model.baseline.preprocess.split import make_splits as baseline_make_splits


class BaselineClassifier:
    def __init__(self, artifact: dict):
        self.artifact = artifact


def load_raw_data(data_path: str, columns_path: str) -> pd.DataFrame:
    return dataset_io.load_raw_data(data_path, columns_path)


def make_splits(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    return baseline_make_splits(df, seed=seed)


def build_preprocessor(train_df: pd.DataFrame):
    return baseline_build_preprocessor(train_df)


def fit_classifier(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None = None,
    sample_weight_col: str = dataset_io.WEIGHT_COL,
):
    _ = sample_weight_col
    train_valid_df = train_df.copy() if valid_df is None else pd.concat([train_df, valid_df], axis=0).copy()
    _, _, artifact = run_group_kfold_baseline(train_valid_df, seed=42, n_splits=3)
    return BaselineClassifier(artifact)


def predict_logits(model: BaselineClassifier, df: pd.DataFrame) -> np.ndarray:
    return baseline_predict_logits(model.artifact, df)


def fit_segmenter(df: pd.DataFrame, n_clusters: int | None = None):
    return baseline_fit_segmenter(df, n_clusters=6 if n_clusters is None else n_clusters)


def assign_segments(segmenter, df: pd.DataFrame) -> np.ndarray:
    return baseline_assign_segments(segmenter, df)
