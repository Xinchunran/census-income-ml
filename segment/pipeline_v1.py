from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model.baseline.dataio.dataset import WEIGHT_COL, clean_label_to_binary, load_raw_data
from model.baseline.preprocess.split import make_splits
from segment.common import assign_segments_from_features, build_segment_outputs, fit_segmenter_from_features
from segment.features_v1 import build_segmentation_features_v1
from segment.io import save_segmentation_outputs
from segment.validation import validate_scheme_name

SCHEME_NAME_V1 = "v1_raw_feature_unsupervised"


def run_segmentation_v1(
    project_root: str | Path,
    output_root: str | Path,
    random_state: int = 42,
    output_subdir: str | None = None,
) -> dict[str, Any]:
    scheme_name = SCHEME_NAME_V1
    validate_scheme_name(scheme_name)
    scheme_type = "raw_feature_unsupervised"

    project_root = Path(project_root)
    output_root = Path(output_root)
    output_dir = output_root / (output_subdir if output_subdir else scheme_name)

    df = load_raw_data(project_root / "census-bureau.data", project_root / "census-bureau.columns")
    df["_row_id"] = np.arange(len(df))
    splits = make_splits(df, seed=random_state)

    train_df = splits["train"].copy()
    valid_df = splits["valid"].copy()
    test_df = splits["test"].copy()

    model = fit_segmenter_from_features(
        train_df=train_df,
        feature_builder=build_segmentation_features_v1,
        scheme_name=scheme_name,
        random_state=random_state,
    )
    assignments_parts: list[pd.DataFrame] = []
    for split_name, part in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        seg = assign_segments_from_features(model, part, build_segmentation_features_v1)
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
                    "scheme_name": scheme_name,
                }
            )
        )
    assignments = pd.concat(assignments_parts, axis=0, ignore_index=True)

    profiles, diagnostics_extra, actions_md = build_segment_outputs(df, assignments, scheme_name=scheme_name)
    diagnostics = {
        **model.diagnostics,
        **diagnostics_extra,
        "scheme_name": scheme_name,
        "scheme_type": scheme_type,
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
    }
    save_segmentation_outputs(output_dir, scheme_name, assignments, profiles, diagnostics, actions_md)
    return diagnostics
