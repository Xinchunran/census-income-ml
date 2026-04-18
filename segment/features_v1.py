from __future__ import annotations

import pandas as pd

from segment.common import _age_band, _prepare, _weeks_band


def build_segmentation_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    """Build raw-feature segmentation inputs (model-independent baseline)."""
    work = _prepare(df)

    out = pd.DataFrame(index=work.index)
    out["age"] = pd.to_numeric(work["age"], errors="coerce")
    out["weeks_worked_in_year"] = pd.to_numeric(work["weeks worked in year"], errors="coerce")
    out["num_persons_worked_for_employer"] = pd.to_numeric(work["num persons worked for employer"], errors="coerce")
    out["veterans_benefits"] = pd.to_numeric(work["veterans benefits"], errors="coerce")

    out["age_band"] = _age_band(work["age"])
    out["weeks_worked_band"] = _weeks_band(work["weeks worked in year"])
    out["capital_gains_positive"] = (pd.to_numeric(work["capital gains"], errors="coerce").fillna(0) > 0).astype(int)
    out["capital_losses_positive"] = (pd.to_numeric(work["capital losses"], errors="coerce").fillna(0) > 0).astype(int)
    out["dividends_positive"] = (pd.to_numeric(work["dividends from stocks"], errors="coerce").fillna(0) > 0).astype(int)

    cat_cols = [
        "sex",
        "education",
        "marital stat",
        "class of worker",
        "major occupation code",
        "major industry code",
        "detailed household summary in household",
        "full or part time employment stat",
        "citizenship",
        "race",
    ]
    for col in cat_cols:
        out[col.replace(" ", "_")] = work[col].astype(str).str.strip()

    return out
