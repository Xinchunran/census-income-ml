from __future__ import annotations

from segment.features_v2 import build_segmentation_features_v2


def test_v2_feature_builder_adds_score_features_only(tiny_segmentation_df):
    df = tiny_segmentation_df.copy()
    df["income_score"] = 0.5
    X = build_segmentation_features_v2(df, score_column="income_score")

    assert "income_score" in X.columns
    assert "income_score_band" in X.columns
    assert "income_score_margin" in X.columns
    assert "label" not in X.columns
