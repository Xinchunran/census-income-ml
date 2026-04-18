"""Contract tests for required segmentation deliverables.

Expected project outputs:
- segment_assignments.csv
- segment_profiles.csv
- segment_diagnostics.json
- segment_marketing_actions.md

Set SEGMENT_OUTPUT_DIR to the directory that contains these files.
Example:
    SEGMENT_OUTPUT_DIR=artifacts/segments pytest -q segmentation_tests
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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
    path = output_dir / "segment_assignments.csv"
    if not path.exists():
        pytest.fail(f"Missing required file: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def profiles(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "segment_profiles.csv"
    if not path.exists():
        pytest.fail(f"Missing required file: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def diagnostics(output_dir: Path) -> dict:
    path = output_dir / "segment_diagnostics.json"
    if not path.exists():
        pytest.fail(f"Missing required file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_required_assignment_columns(assignments: pd.DataFrame) -> None:
    required = {"row_id", "segment_id"}
    missing = required - set(assignments.columns)
    assert not missing, f"segment_assignments.csv missing columns: {sorted(missing)}"


def test_one_segment_per_row(assignments: pd.DataFrame) -> None:
    assert assignments["row_id"].is_unique, "Each row_id should appear exactly once."
    assert assignments["segment_id"].notna().all(), "Every row must receive a segment_id."


def test_segment_count_reasonable(assignments: pd.DataFrame) -> None:
    k = assignments["segment_id"].nunique(dropna=True)
    assert 3 <= k <= 10, f"Expected 3-10 segments for business usability, got {k}."


def test_profiles_cover_same_segments(assignments: pd.DataFrame, profiles: pd.DataFrame) -> None:
    a = set(assignments["segment_id"].dropna().unique())
    p = set(profiles["segment_id"].dropna().unique())
    assert a == p, f"Segments in assignments and profiles differ. assignments={sorted(a)} profiles={sorted(p)}"


def test_profile_minimum_columns(profiles: pd.DataFrame) -> None:
    required = {
        "segment_id",
        "segment_size_n",
        "segment_weight_sum",
        "segment_share_weighted",
        "income_rate_weighted",
    }
    missing = required - set(profiles.columns)
    assert not missing, f"segment_profiles.csv missing columns: {sorted(missing)}"


def test_profile_weight_share_sums_to_one(profiles: pd.DataFrame) -> None:
    total = profiles["segment_share_weighted"].sum()
    assert abs(total - 1.0) < 0.02, f"Weighted segment shares should sum to ~1.0, got {total:.4f}"


def test_no_empty_segments(assignments: pd.DataFrame) -> None:
    counts = assignments.groupby("segment_id").size()
    assert (counts > 0).all(), "No segment may be empty."


def test_marketing_actions_file_present(output_dir: Path) -> None:
    path = output_dir / "segment_marketing_actions.md"
    if not path.exists():
        pytest.fail(f"Missing required file: {path}")
    text = path.read_text(encoding="utf-8").strip()
    assert len(text) > 200, "segment_marketing_actions.md is too short to be useful."


def test_diagnostics_has_core_fields(diagnostics: dict) -> None:
    required = {"n_clusters", "cluster_size_distribution"}
    missing = required - set(diagnostics.keys())
    assert not missing, f"segment_diagnostics.json missing fields: {sorted(missing)}"
