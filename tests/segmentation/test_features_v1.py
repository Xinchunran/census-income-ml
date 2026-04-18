from __future__ import annotations

from segment.features_v1 import build_segmentation_features_v1


def test_v1_feature_builder_excludes_label_and_prediction_fields(tiny_segmentation_df):
    X = build_segmentation_features_v1(tiny_segmentation_df)
    assert "label" not in X.columns
    assert "income_score" not in X.columns
    assert "age" in X.columns
    assert "education" in X.columns or "education" in [c.replace("_", " ") for c in X.columns]
