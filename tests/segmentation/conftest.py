from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tiny_segmentation_df() -> pd.DataFrame:
    rows = 40
    return pd.DataFrame(
        {
            "age": [25 + (i % 30) for i in range(rows)],
            "weeks worked in year": [52 - (i % 20) for i in range(rows)],
            "num persons worked for employer": [1 + (i % 10) for i in range(rows)],
            "veterans benefits": [0 if i % 5 else 1 for i in range(rows)],
            "capital gains": [1000 if i % 7 == 0 else 0 for i in range(rows)],
            "capital losses": [200 if i % 11 == 0 else 0 for i in range(rows)],
            "dividends from stocks": [50 if i % 9 == 0 else 0 for i in range(rows)],
            "sex": ["Male" if i % 2 else "Female" for i in range(rows)],
            "education": ["Bachelors degree(BA AB BS)" if i % 3 == 0 else "HS graduate" for i in range(rows)],
            "marital stat": ["Married" if i % 4 == 0 else "Never married" for i in range(rows)],
            "class of worker": ["Private" for _ in range(rows)],
            "major occupation code": ["Professional specialty" if i % 3 == 0 else "Sales" for i in range(rows)],
            "major industry code": ["Finance" if i % 3 == 0 else "Retail" for i in range(rows)],
            "detailed household summary in household": ["Householder" for _ in range(rows)],
            "full or part time employment stat": ["Full-time schedules" if i % 6 else "Part-time schedules" for i in range(rows)],
            "citizenship": ["Native- Born in the United States" for _ in range(rows)],
            "race": ["White" if i % 3 else "Black" for i in range(rows)],
            "label": ["50000+." if i % 5 == 0 else " - 50000." for i in range(rows)],
            "weight": [1.0 + (i % 3) for i in range(rows)],
            "year": [1994 + (i % 2) for i in range(rows)],
            "_row_id": list(range(rows)),
        }
    )


@pytest.fixture
def tiny_score_df() -> pd.DataFrame:
    rows = 40
    splits = ["train"] * 28 + ["valid"] * 6 + ["test"] * 6
    return pd.DataFrame(
        {
            "row_id": list(range(rows)),
            "split": splits,
            "income_score": [0.1 + 0.02 * (i % 20) for i in range(rows)],
        }
    )
