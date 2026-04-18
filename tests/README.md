# Test Suite for the Census Take-Home Project

These tests are designed to prevent overly aggressive preprocessing shortcuts and to enforce a clean modeling contract.

## Recommended project API

Create a `project_api.py` file (or adapt the imports in the tests) with the following functions:

```python
load_raw_data(data_path: str, columns_path: str) -> pandas.DataFrame
make_splits(df: pandas.DataFrame, seed: int = 42) -> dict[str, pandas.DataFrame]
build_preprocessor(train_df: pandas.DataFrame)
fit_classifier(train_df: pandas.DataFrame, valid_df: pandas.DataFrame | None = None, sample_weight_col: str = "weight")
predict_logits(model, df: pandas.DataFrame) -> numpy.ndarray
fit_segmenter(df: pandas.DataFrame, n_clusters: int | None = None)
assign_segments(segmenter, df: pandas.DataFrame) -> numpy.ndarray
```

The tests assume:
- `weight` is **not** used as a model feature.
- `label` is the binary target.
- raw unknown markers such as `?` are preserved or mapped explicitly to an `Unknown`/`Missing` category, rather than silently dropped.
- classification output is **logit-first**.
- clustering assigns a valid segment to every row.

## How to run

```bash
pip install pytest pandas numpy scikit-learn
pytest -q tests/
```

If you use a different module layout, edit the imports near the top of each test file.
