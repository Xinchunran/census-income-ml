from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from segment.pipeline_v2 import run_segmentation_v2


def test_run_segmentation_v2_smoke(monkeypatch, tmp_path, tiny_segmentation_df, tiny_score_df):
    df = tiny_segmentation_df.copy()

    def fake_load_raw_data(*args, **kwargs):
        return df.copy()

    def fake_make_splits(df_in, seed=42):
        return {
            "train": df_in.iloc[:28].copy(),
            "valid": df_in.iloc[28:34].copy(),
            "test": df_in.iloc[34:].copy(),
        }

    score_path = tmp_path / "scores.csv"
    tiny_score_df.to_csv(score_path, index=False)

    monkeypatch.setattr("segment.pipeline_v2.load_raw_data", fake_load_raw_data)
    monkeypatch.setattr("segment.pipeline_v2.make_splits", fake_make_splits)

    run_segmentation_v2(
        project_root=".",
        output_root=tmp_path,
        score_path=score_path,
        random_state=42,
        score_column="income_score",
    )

    scheme_dir = Path(tmp_path) / "v2_score_augmented_classifier_informed"
    assert scheme_dir.exists()
    assert (scheme_dir / "segment_assignments.csv").exists()
    assert (scheme_dir / "segment_profiles.csv").exists()
    assert (scheme_dir / "segment_diagnostics.json").exists()

    assignments = pd.read_csv(scheme_dir / "segment_assignments.csv")
    assert "scheme_name" in assignments.columns
    assert set(assignments["scheme_name"].unique()) == {"v2_score_augmented_classifier_informed"}
    assert "income_score" in assignments.columns

    diagnostics = json.loads((scheme_dir / "segment_diagnostics.json").read_text())
    assert diagnostics["scheme_name"] == "v2_score_augmented_classifier_informed"
    assert diagnostics["score_column_used"] == "income_score"
