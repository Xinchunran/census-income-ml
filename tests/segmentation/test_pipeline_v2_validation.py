from __future__ import annotations

import pandas as pd
import pytest

from segment.io import merge_scores_into_split
from segment.validation import validate_score_table


def test_validate_score_table_rejects_missing_column():
    score_df = pd.DataFrame({"row_id": [1, 2], "split": ["train", "test"]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_score_table(score_df, score_column="income_score")


def test_validate_score_table_rejects_duplicate_row_ids():
    score_df = pd.DataFrame(
        {
            "row_id": [1, 1],
            "split": ["train", "train"],
            "income_score": [0.3, 0.4],
        }
    )
    with pytest.raises(ValueError, match="duplicated row_id"):
        validate_score_table(score_df, score_column="income_score")


def test_validate_score_table_rejects_out_of_range_scores():
    score_df = pd.DataFrame(
        {
            "row_id": [1, 2],
            "split": ["train", "test"],
            "income_score": [1.2, -0.1],
        }
    )
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        validate_score_table(score_df, score_column="income_score")


def test_merge_scores_into_split_fails_on_missing_scores():
    split_df = pd.DataFrame({"_row_id": [1, 2, 3]})
    score_df = pd.DataFrame(
        {
            "row_id": [1, 2],
            "split": ["train", "train"],
            "income_score": [0.1, 0.9],
        }
    )

    with pytest.raises(ValueError, match="missing score"):
        merge_scores_into_split(split_df, score_df, split_name="train", score_column="income_score")
