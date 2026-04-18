import numpy as np
import pandas as pd

try:
    from project_api import load_raw_data, make_splits, build_preprocessor
except ImportError as e:
    raise ImportError(
        "Edit this test to import your own project functions, or implement project_api.py."
    ) from e


FORBIDDEN_FEATURES = {"label", "weight"}


def test_preprocessor_excludes_target_and_weight():
    df = load_raw_data("census-bureau.data", "census-bureau.columns")
    splits = make_splits(df, seed=42)
    train_df = splits["train"]
    valid_df = splits["valid"]

    preprocessor = build_preprocessor(train_df)

    X_train = preprocessor.transform(train_df)
    X_valid = preprocessor.transform(valid_df)

    assert X_train.shape[0] == len(train_df)
    assert X_valid.shape[0] == len(valid_df)

    # This catches silent row drops during transform.
    assert X_train.shape[0] > 0 and X_valid.shape[0] > 0


def test_unknown_categories_do_not_crash_transform():
    df = load_raw_data("census-bureau.data", "census-bureau.columns")
    splits = make_splits(df, seed=42)
    train_df = splits["train"].copy()
    valid_df = splits["valid"].copy()

    preprocessor = build_preprocessor(train_df)

    # Inject an unseen category into validation to make sure the transform is robust.
    cat_col = "class of worker"
    valid_df.loc[valid_df.index[:5], cat_col] = "__UNSEEN_TEST_CATEGORY__"
    X_valid = preprocessor.transform(valid_df)
    assert X_valid.shape[0] == len(valid_df)


def test_structural_missing_is_not_row_drop_shortcut():
    df = load_raw_data("census-bureau.data", "census-bureau.columns")
    raw_rows = len(df)
    splits = make_splits(df, seed=42)
    combined_rows = sum(len(v) for v in splits.values())
    assert combined_rows == raw_rows, (
        "Rows were dropped before/while splitting; this is risky because '?' is structural in several columns."
    )
