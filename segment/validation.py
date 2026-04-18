from __future__ import annotations

import pandas as pd


def validate_scheme_name(scheme_name: str) -> None:
    allowed = {
        "v1_raw_feature_unsupervised",
        "v2_score_augmented_classifier_informed",
    }
    if scheme_name not in allowed:
        raise ValueError(f"Unknown scheme_name: {scheme_name}")


def validate_score_table(score_df: pd.DataFrame, score_column: str = "income_score") -> None:
    required = {"row_id", "split", score_column}
    missing = required - set(score_df.columns)
    if missing:
        raise ValueError(f"Score table missing required columns: {sorted(missing)}")

    if score_df["row_id"].duplicated().any():
        dup_n = int(score_df["row_id"].duplicated().sum())
        raise ValueError(f"Score table has duplicated row_id values: {dup_n}")

    score = pd.to_numeric(score_df[score_column], errors="coerce")
    if score.isna().any():
        raise ValueError(f"Score column {score_column} contains non-numeric or missing values")

    if ((score < 0) | (score > 1)).any():
        raise ValueError(f"Score column {score_column} must stay within [0, 1]")
