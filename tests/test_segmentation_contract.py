import numpy as np

try:
    from project_api import load_raw_data, make_splits, fit_segmenter, assign_segments
except ImportError as e:
    raise ImportError(
        "Edit this test to import your own project functions, or implement project_api.py."
    ) from e


def test_segmenter_assigns_every_row_and_avoids_empty_clusters():
    df = load_raw_data("census-bureau.data", "census-bureau.columns")
    splits = make_splits(df, seed=42)
    train_df = splits["train"]
    valid_df = splits["valid"]

    segmenter = fit_segmenter(train_df, n_clusters=6)
    train_seg = np.asarray(assign_segments(segmenter, train_df))
    valid_seg = np.asarray(assign_segments(segmenter, valid_df))

    assert train_seg.shape[0] == len(train_df)
    assert valid_seg.shape[0] == len(valid_df)

    train_unique = np.unique(train_seg)
    valid_unique = np.unique(valid_seg)

    assert len(train_unique) >= 3, "Too few active clusters; segmentation is likely underfit."
    assert len(train_unique) <= 12, "Too many active clusters; segmentation may not be actionable."
    assert len(valid_unique) >= 2, "Validation assignment collapsed to too few clusters."

    counts = np.bincount(train_seg.astype(int)) if np.issubdtype(train_seg.dtype, np.integer) else None
    if counts is not None:
        assert (counts > 0).all(), "At least one cluster is empty on training data."
