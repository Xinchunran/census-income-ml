from __future__ import annotations

from pathlib import Path

import pandas as pd

from segment.common import SegmenterArtifact, assign_segments_from_features, fit_segmenter_from_features
from segment.features_v1 import build_segmentation_features_v1
from segment.pipeline_v1 import SCHEME_NAME_V1, run_segmentation_v1


def build_segmentation_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_segmentation_features_v1(df)


def fit_segmenter(df: pd.DataFrame, random_state: int = 42, n_clusters: int | None = None) -> SegmenterArtifact:
    return fit_segmenter_from_features(
        train_df=df,
        feature_builder=build_segmentation_features_v1,
        scheme_name=SCHEME_NAME_V1,
        random_state=random_state,
        fixed_n_clusters=n_clusters,
    )


def assign_segments(model: SegmenterArtifact, df: pd.DataFrame):
    return assign_segments_from_features(model, df, build_segmentation_features_v1)


def run_segmentation_pipeline(project_root: str | Path, output_dir: str | Path, random_state: int = 42):
    output_dir = Path(output_dir)
    return run_segmentation_v1(
        project_root=project_root,
        output_root=output_dir.parent,
        random_state=random_state,
        output_subdir=output_dir.name,
    )
