from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from segment.validation import validate_score_table


def load_score_table(score_path: str | Path, score_column: str = "income_score") -> pd.DataFrame:
    score_df = pd.read_csv(score_path)
    validate_score_table(score_df, score_column=score_column)
    return score_df


def merge_scores_into_split(
    split_df: pd.DataFrame,
    score_df: pd.DataFrame,
    split_name: str,
    score_column: str = "income_score",
) -> pd.DataFrame:
    split_scores = score_df[score_df["split"].astype(str) == str(split_name)].copy()
    merged = split_df.merge(
        split_scores[["row_id", score_column]],
        left_on="_row_id",
        right_on="row_id",
        how="left",
        validate="one_to_one",
    )
    missing = int(merged[score_column].isna().sum())
    if missing > 0:
        raise ValueError(f"Split `{split_name}` has missing score after merge: {missing} rows")
    return merged


def save_segmentation_outputs(
    output_dir: Path,
    scheme_name: str,
    assignments: pd.DataFrame,
    profiles: pd.DataFrame,
    diagnostics: dict[str, Any],
    actions_md: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    a = assignments.copy()
    p = profiles.copy()
    a["scheme_name"] = scheme_name
    p["scheme_name"] = scheme_name

    (output_dir / "segment_assignments.csv").write_text(a.to_csv(index=False), encoding="utf-8")
    (output_dir / "segment_profiles.csv").write_text(p.to_csv(index=False), encoding="utf-8")
    (output_dir / "segment_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "segment_marketing_actions.md").write_text(actions_md, encoding="utf-8")
