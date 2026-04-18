from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from segment.pipeline_v1 import run_segmentation_v1


def test_run_segmentation_v1_smoke(monkeypatch, tmp_path, tiny_segmentation_df):
    df = tiny_segmentation_df.copy()

    def fake_load_raw_data(*args, **kwargs):
        return df.copy()

    def fake_make_splits(df_in, seed=42):
        return {
            "train": df_in.iloc[:28].copy(),
            "valid": df_in.iloc[28:34].copy(),
            "test": df_in.iloc[34:].copy(),
        }

    monkeypatch.setattr("segment.pipeline_v1.load_raw_data", fake_load_raw_data)
    monkeypatch.setattr("segment.pipeline_v1.make_splits", fake_make_splits)

    run_segmentation_v1(project_root=".", output_root=tmp_path, random_state=42)

    scheme_dir = Path(tmp_path) / "v1_raw_feature_unsupervised"
    assert scheme_dir.exists()
    assert (scheme_dir / "segment_assignments.csv").exists()
    assert (scheme_dir / "segment_profiles.csv").exists()
    assert (scheme_dir / "segment_diagnostics.json").exists()

    assignments = pd.read_csv(scheme_dir / "segment_assignments.csv")
    assert "scheme_name" in assignments.columns
    assert set(assignments["scheme_name"].unique()) == {"v1_raw_feature_unsupervised"}

    diagnostics = json.loads((scheme_dir / "segment_diagnostics.json").read_text())
    assert diagnostics["scheme_name"] == "v1_raw_feature_unsupervised"
