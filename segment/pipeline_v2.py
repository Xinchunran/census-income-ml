from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from model.baseline.dataio.dataset import WEIGHT_COL, clean_label_to_binary, load_raw_data
from model.baseline.preprocess.split import make_splits
from segment.common import assign_segments_from_features, build_segment_outputs, fit_segmenter_from_features
from segment.features_v2 import build_segmentation_features_v2
from segment.io import load_score_table, merge_scores_into_split, save_segmentation_outputs
from segment.validation import validate_scheme_name

SCHEME_NAME_V2 = "v2_score_augmented_classifier_informed"


def run_segmentation_v2(
    project_root: str | Path,
    output_root: str | Path,
    score_path: str | Path,
    random_state: int = 42,
    score_column: str = "income_score",
    output_subdir: str | None = None,
) -> dict[str, Any]:
    scheme_name = SCHEME_NAME_V2
    validate_scheme_name(scheme_name)
    scheme_type = "classifier_informed_score_augmented"

    project_root = Path(project_root)
    output_root = Path(output_root)
    output_dir = output_root / (output_subdir if output_subdir else scheme_name)

    df = load_raw_data(project_root / "census-bureau.data", project_root / "census-bureau.columns")
    df["_row_id"] = np.arange(len(df))
    splits = make_splits(df, seed=random_state)
    score_df = load_score_table(score_path, score_column=score_column)

    train_df = merge_scores_into_split(splits["train"].copy(), score_df, split_name="train", score_column=score_column)
    valid_df = merge_scores_into_split(splits["valid"].copy(), score_df, split_name="valid", score_column=score_column)
    test_df = merge_scores_into_split(splits["test"].copy(), score_df, split_name="test", score_column=score_column)

    builder: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: build_segmentation_features_v2(x, score_column=score_column)
    model = fit_segmenter_from_features(
        train_df=train_df,
        feature_builder=builder,
        scheme_name=scheme_name,
        random_state=random_state,
    )
    assignments_parts: list[pd.DataFrame] = []
    for split_name, part in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        seg = assign_segments_from_features(model, part, builder)
        y = clean_label_to_binary(part["label"]).to_numpy()
        assignments_parts.append(
            pd.DataFrame(
                {
                    "row_id": part["_row_id"].to_numpy(),
                    "segment_id": seg,
                    "segment_name": [f"Segment {int(v)}" for v in seg],
                    "split": split_name,
                    "weight": part[WEIGHT_COL].to_numpy(),
                    "year": part["year"].to_numpy(),
                    "income_label": y,
                    score_column: pd.to_numeric(part[score_column], errors="coerce").to_numpy(),
                    "score_column_used": score_column,
                    "scheme_name": scheme_name,
                }
            )
        )
    assignments = pd.concat(assignments_parts, axis=0, ignore_index=True)

    base_df = df.copy()
    score_series = assignments.set_index("row_id")[score_column]
    base_df[score_column] = base_df["_row_id"].map(score_series)
    profiles, diagnostics_extra, actions_md = build_segment_outputs(base_df, assignments, scheme_name=scheme_name)
    diagnostics = {
        **model.diagnostics,
        **diagnostics_extra,
        "scheme_name": scheme_name,
        "scheme_type": scheme_type,
        "score_column_used": score_column,
        "score_source_path": str(Path(score_path)),
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
    }
    save_segmentation_outputs(output_dir, scheme_name, assignments, profiles, diagnostics, actions_md)
    return diagnostics
