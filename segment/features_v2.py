from __future__ import annotations

import pandas as pd

from segment.common import _prepare
from segment.features_v1 import build_segmentation_features_v1


def build_segmentation_features_v2(df: pd.DataFrame, score_column: str = "income_score") -> pd.DataFrame:
    """Build score-augmented segmentation inputs (classifier-informed)."""
    work = _prepare(df)
    out = build_segmentation_features_v1(work)

    score = pd.to_numeric(work[score_column], errors="coerce")
    out["income_score"] = score
    out["income_score_band"] = pd.cut(
        score,
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
    ).astype(str)
    out["income_score_margin"] = (score - 0.5).abs()
    return out
