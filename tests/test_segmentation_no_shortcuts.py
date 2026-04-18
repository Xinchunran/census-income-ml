"""Tests to catch common shortcut implementations.

These tests expect a Python module path in SEGMENT_PIPELINE_MODULE that exposes:
- build_segmentation_features(df: pd.DataFrame) -> pd.DataFrame
- fit_segmenter(train_df: pd.DataFrame, random_state: int = 42)
- assign_segments(model, df: pd.DataFrame) -> pd.Series | np.ndarray

They are intentionally strict about leakage and business usability.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_PATH = Path("/mnt/data/census-bureau.data")
COLS_PATH = Path("/mnt/data/census-bureau.columns")


@pytest.fixture(scope="session")
def pipeline_module():
    module_name = os.environ.get("SEGMENT_PIPELINE_MODULE")
    if not module_name:
        pytest.skip("Set SEGMENT_PIPELINE_MODULE to your segmentation pipeline module.")
    return importlib.import_module(module_name)


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    if not DATA_PATH.exists() or not COLS_PATH.exists():
        pytest.skip("Raw dataset files are not available in expected paths.")

    cols = [line.strip() for line in COLS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.read_csv(DATA_PATH, header=None, names=cols)
    return df


def test_feature_builder_excludes_label_weight_and_predictions(pipeline_module, raw_df: pd.DataFrame) -> None:
    if not hasattr(pipeline_module, "build_segmentation_features"):
        pytest.skip("build_segmentation_features not implemented.")
    X = pipeline_module.build_segmentation_features(raw_df.copy())
    forbidden = {
        "income",
        "label",
        "target",
        "probability",
        "prediction",
        "score",
        "logit",
        "weight",
        "instance weight",
    }
    lowered = {str(c).strip().lower() for c in X.columns}
    overlap = sorted(forbidden & lowered)
    assert not overlap, f"Segmentation features appear to include forbidden leakage columns: {overlap}"


def test_fit_does_not_require_label_column(pipeline_module, raw_df: pd.DataFrame) -> None:
    if not hasattr(pipeline_module, "fit_segmenter"):
        pytest.skip("fit_segmenter not implemented.")

    df = raw_df.copy()
    for candidate in ["income", "label", "target"]:
        if candidate in df.columns:
            df = df.drop(columns=[candidate])
    try:
        model = pipeline_module.fit_segmenter(df, random_state=42)
    except Exception as exc:
        pytest.fail(f"fit_segmenter should not depend on label columns. Error: {exc}")
    assert model is not None


def test_assignments_not_single_cluster(pipeline_module, raw_df: pd.DataFrame) -> None:
    if not hasattr(pipeline_module, "fit_segmenter") or not hasattr(pipeline_module, "assign_segments"):
        pytest.skip("fit_segmenter / assign_segments not implemented.")
    df = raw_df.sample(n=min(20000, len(raw_df)), random_state=42)
    model = pipeline_module.fit_segmenter(df, random_state=42)
    seg = pipeline_module.assign_segments(model, df)
    seg = pd.Series(seg)
    assert seg.nunique(dropna=True) >= 3, "A single-cluster or two-cluster solution is too coarse for this task."


def test_no_tiny_clusters_by_row_count(pipeline_module, raw_df: pd.DataFrame) -> None:
    if not hasattr(pipeline_module, "fit_segmenter") or not hasattr(pipeline_module, "assign_segments"):
        pytest.skip("fit_segmenter / assign_segments not implemented.")
    df = raw_df.sample(n=min(20000, len(raw_df)), random_state=42)
    model = pipeline_module.fit_segmenter(df, random_state=42)
    seg = pd.Series(pipeline_module.assign_segments(model, df), index=df.index)
    shares = seg.value_counts(normalize=True)
    assert shares.min() >= 0.01, f"At least one cluster is too tiny by row share: min={shares.min():.4f}"


def test_seed_stability_reasonable(pipeline_module, raw_df: pd.DataFrame) -> None:
    if not hasattr(pipeline_module, "fit_segmenter") or not hasattr(pipeline_module, "assign_segments"):
        pytest.skip("fit_segmenter / assign_segments not implemented.")

    df = raw_df.sample(n=min(10000, len(raw_df)), random_state=123).reset_index(drop=True)
    model_a = pipeline_module.fit_segmenter(df, random_state=11)
    model_b = pipeline_module.fit_segmenter(df, random_state=22)
    seg_a = pd.Series(pipeline_module.assign_segments(model_a, df))
    seg_b = pd.Series(pipeline_module.assign_segments(model_b, df))

    # Compare pairwise co-clustering agreement, invariant to label permutation.
    sample_idx = np.random.default_rng(42).choice(len(df), size=min(2000, len(df)), replace=False)
    a = seg_a.iloc[sample_idx].to_numpy()
    b = seg_b.iloc[sample_idx].to_numpy()
    agree = []
    for i in range(len(sample_idx) - 1):
        same_a = a[i + 1 :] == a[i]
        same_b = b[i + 1 :] == b[i]
        agree.extend((same_a == same_b).astype(int).tolist())
    pairwise_agreement = float(np.mean(agree)) if agree else 1.0
    assert pairwise_agreement >= 0.75, f"Cluster solution seems unstable across seeds: pairwise agreement={pairwise_agreement:.3f}"
