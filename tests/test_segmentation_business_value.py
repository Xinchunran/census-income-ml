"""Business-facing segmentation validation.

Expected file contract:
- segment_assignments.csv contains `segment_id` and optional `income_label`, `weight`
- segment_profiles.csv contains weighted summaries

These tests ensure the output is not just mathematically clustered but also business-usable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def output_dir() -> Path:
    value = os.environ.get("SEGMENT_OUTPUT_DIR")
    if not value:
        pytest.skip("Set SEGMENT_OUTPUT_DIR to the segmentation artifact directory.")
    path = Path(value)
    if not path.exists():
        pytest.fail(f"SEGMENT_OUTPUT_DIR does not exist: {path}")
    return path


@pytest.fixture(scope="session")
def assignments(output_dir: Path) -> pd.DataFrame:
    return pd.read_csv(output_dir / "segment_assignments.csv")


@pytest.fixture(scope="session")
def profiles(output_dir: Path) -> pd.DataFrame:
    return pd.read_csv(output_dir / "segment_profiles.csv")


def test_segments_have_business_names_or_descriptions(output_dir: Path, profiles: pd.DataFrame) -> None:
    has_name_col = "segment_name" in profiles.columns and profiles["segment_name"].notna().all()
    actions_text = (output_dir / "segment_marketing_actions.md").read_text(encoding="utf-8")
    has_actions_for_all = profiles["segment_id"].astype(str).apply(lambda x: x in actions_text).all()
    assert has_name_col or has_actions_for_all, "Each segment needs a business-readable name or a written description in the action memo."


def test_segments_show_outcome_separation_if_label_available(assignments: pd.DataFrame) -> None:
    label_cols = [c for c in assignments.columns if c.lower() in {"income_label", "target", "label", "true_label"}]
    if not label_cols:
        pytest.skip("No label column present in segment assignments; skipping post hoc outcome separation test.")

    label_col = label_cols[0]
    weight_col = "weight" if "weight" in assignments.columns else None

    def wmean(g: pd.DataFrame) -> float:
        y = pd.to_numeric(g[label_col])
        if weight_col:
            w = pd.to_numeric(g[weight_col])
            return float(np.average(y, weights=w))
        return float(y.mean())

    rates = assignments.groupby("segment_id", observed=True).apply(wmean)
    spread = float(rates.max() - rates.min())
    assert spread >= 0.03, f"Segments barely differ on outcome rate; spread={spread:.4f}. This may not be useful for marketing prioritization."


def test_weighted_segment_shares_not_too_concentrated(profiles: pd.DataFrame) -> None:
    if "segment_share_weighted" not in profiles.columns:
        pytest.skip("segment_share_weighted not found in segment_profiles.csv")
    max_share = float(profiles["segment_share_weighted"].max())
    assert max_share <= 0.80, f"One segment dominates too much of the weighted population: max share={max_share:.3f}"


def test_profiles_include_multiple_distinguishing_dimensions(profiles: pd.DataFrame) -> None:
    candidate_cols = [
        "top_age_band",
        "top_education",
        "top_marital_status",
        "top_work_class",
        "top_occupation",
        "weeks_worked_mean",
        "income_rate_weighted",
    ]
    present = [c for c in candidate_cols if c in profiles.columns]
    assert len(present) >= 4, f"segment_profiles.csv should profile segments on at least 4 dimensions, found {present}"
