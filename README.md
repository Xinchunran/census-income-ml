# Census Income Take-Home Project

This repository contains:
- A supervised **income classification** pipeline (`<=50K` vs `>50K`).
- An unsupervised **customer segmentation** pipeline for marketing actions.

## 1. Project Structure

```text
.
├── census-bureau.data
├── census-bureau.columns
├── model/
│   ├── baseline/      # tree-based models: CatBoost / XGBoost / LightGBM
│   └── ft_tf/         # FT-Transformer model
├── segment/           # segmentation pipeline + outputs
├── tests/             # contract tests
└── project_api.py
```

## 2. Environment Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- If you train FT-Transformer on CPU, pass `--device cpu`.
- If CUDA is unavailable, FT scripts automatically work with CPU when configured.

## 3. Data Processing Pipeline

### 3.1 Raw Data Loading

- `census-bureau.columns` is read as schema.
- `census-bureau.data` is loaded with those column names.
- Target column is `label`.
- Sample weight column is `weight`.

### 3.2 Label and Weight Handling

- Binary target mapping: `label == "50000+."` -> `1`, otherwise `0`.
- `weight` is used as **sample weight** in training/evaluation.
- `weight` is **not used as a model feature**.

### 3.3 Split Strategy (No Leakage)

- A persistent `_row_id` is added before splitting.
- Data split uses `StratifiedGroupKFold` and a stratification key: `label + "_" + year`.
- Output split sizes (from current artifacts):
  - train: `119,713`
  - valid: `39,905`
  - test: `39,905`

### 3.4 Baseline Feature Processing

- Column typing:
  - numeric if dtype is numeric, or if numeric parse ratio > `0.98`
  - otherwise categorical
- Numeric pipeline: median imputation.
- Categorical pipeline: most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`).
- CatBoost path keeps categorical columns as raw strings for native categorical handling.

### 3.5 FT-Transformer Feature Processing

- Same column typing logic (`numeric` vs `categorical`).
- Numeric columns: median fill and cast to float32.
- Categorical columns: value-to-index mapping per column.
- Index `0` is reserved for unseen/unknown categories at inference.

### 3.6 Segmentation Feature Processing

- V1 segmentation excludes supervised outputs entirely.
- V2 segmentation reuses the V1 feature space and adds a model-derived income propensity score as an auxiliary feature.
- Engineered fields include:
  - numeric: age, weeks worked, veterans benefits, etc.
  - bands: `age_band`, `weeks_worked_band`
  - binary flags: positive capital gains/losses/dividends
  - selected categorical profile columns
- Pipeline:
  - numeric: median imputation + standard scaling
  - categorical: most-frequent imputation + one-hot encoding
- Clustering uses `MiniBatchKMeans`; best `k` searched from `3` to `8`.

## 4. How to Run

All commands below are run from project root.

### 4.1 Train Baseline Models (CatBoost/XGBoost/LightGBM Search)

```bash
python -m model.baseline.train.run_baseline \
  --data census-bureau.data \
  --columns census-bureau.columns \
  --output-root . \
  --seed 42 \
  --n-splits 5
```

Optional (to save suffixed files, e.g. `*_grid.*`):

```bash
python -m model.baseline.train.run_baseline \
  --data census-bureau.data \
  --columns census-bureau.columns \
  --output-root . \
  --seed 42 \
  --n-splits 5 \
  --file-suffix grid
```

Output folder:
- `model/baseline/inference/params/`
- `model/baseline/results/`

### 4.2 Tune Baseline Classification Threshold

```bash
python -m model.baseline.train.tune_threshold \
  --root-dir . \
  --artifact-path model/baseline/inference/params/lightgbm_artifact.pkl \
  --seed 42
```

Outputs:
- `model/baseline/inference/params/<model>_threshold.json`
- `model/baseline/results/threshold_report.json`

### 4.3 Baseline Batch Inference

```bash
python -m model.baseline.inference.predict \
  --params-pkl model/baseline/inference/params/lightgbm_artifact.pkl \
  --data census-bureau.data \
  --columns census-bureau.columns \
  --output model/baseline/results/inference_predictions.csv
```

### 4.4 Train FT-Transformer

```bash
python -m model.ft_tf.train.run_ft_tf \
  --data census-bureau.data \
  --columns census-bureau.columns \
  --output-root . \
  --seed 42 \
  --n-splits 5 \
  --max-epochs 6 \
  --batch-size 512 \
  --device cuda
```

CPU-only example:

```bash
python -m model.ft_tf.train.run_ft_tf --device cpu
```

### 4.5 Export FT-Transformer Suffixed Artifacts (`_ft_tf`)

```bash
python -m model.ft_tf.train.export_suffix_outputs \
  --project-root . \
  --suffix ft_tf \
  --seed 42
```

This creates:
- `model/ft_tf/inference/params/ft_transformer_artifact_ft_tf.pt`
- `model/ft_tf/results/test_metrics_ft_tf.json`
- `model/ft_tf/results/test_predictions_ft_tf.csv`
- `model/ft_tf/results/inference_predictions_ft_tf.csv`

### 4.6 FT-Transformer Batch Inference

```bash
python -m model.ft_tf.inference.predict \
  --checkpoint model/ft_tf/inference/params/ft_transformer_artifact_ft_tf.pt \
  --data census-bureau.data \
  --columns census-bureau.columns \
  --output model/ft_tf/results/inference_predictions_ft_tf.csv \
  --threshold 0.82 \
  --device cuda
```

### 4.7 Run Segmentation Pipeline

```bash
python -m segment.run_segmentation \
  --project-root . \
  --output-dir segment/results \
  --seed 42
```

Outputs:
- `segment/results/segment_assignments.csv`
- `segment/results/segment_profiles.csv`
- `segment/results/segment_diagnostics.json`
- `segment/results/segment_marketing_actions.md`

### 4.8 Run Classifier-Informed Segmentation (V2)

```bash
python -m segment.run_classifier_informed \
  --project-root . \
  --output-dir segment/v2_from_model_score \
  --seed 42
```

Outputs:
- `segment/v2_from_model_score/segment_assignments.csv`
- `segment/v2_from_model_score/segment_profiles.csv`
- `segment/v2_from_model_score/segment_diagnostics.json`
- `segment/v2_from_model_score/segment_marketing_actions.md`

Note:
- The checked-in artifact snapshot in this repository still lives under `segment/v2_from_ft_score/` for historical naming reasons.
- In the final report, V2 is treated generically as a classifier-informed segmentation that adds a model-derived propensity score as an auxiliary feature.

### 4.9 Generate Report Figures

```bash
python plot/scripts/generate_report_figures.py
```

Outputs:
- `plot/fig/*.png`
- `plot/fig/*.csv`

### 4.10 Build The Report

```bash
xelatex report.tex
```

Main output:
- `report.pdf`

### 4.11 Run Tests

```bash
pytest -q tests/
```

## 5. Output Artifacts

### 5.1 Classification Outputs

Each prediction output includes:
- `logit`
- `probability = sigmoid(logit)`
- `predicted_label` (thresholded)
- `threshold`
- model metadata

### 5.2 Important Result Files

- Baseline:
  - `model/baseline/results/kfold_metrics.csv`
  - `model/baseline/results/test_metrics.json`
  - `model/baseline/results/threshold_report.json`
  - `model/baseline/results/test_predictions.csv`
- FT-Transformer:
  - `model/ft_tf/results/kfold_metrics_ft_tf.csv`
  - `model/ft_tf/results/test_metrics_ft_tf.json`
  - `model/ft_tf/results/threshold_report_ft_tf.json`
  - `model/ft_tf/results/test_predictions_ft_tf.csv`
- Segmentation:
  - `segment/results/segment_profiles.csv`
  - `segment/results/segment_diagnostics.json`
  - `segment/v2_from_ft_score/segment_profiles.csv`
  - `segment/v2_from_ft_score/segment_diagnostics.json`

## 6. Final Results (Current Artifacts)

### 6.1 Supervised Classification

From `model/baseline/results/test_metrics.json`:
- model: `lightgbm`
- ROC-AUC: `0.9549`
- PR-AUC: `0.6846`
- weighted log loss: `0.1139`

From `model/baseline/results/test_metrics_grid.json`:
- model: `xgboost`
- ROC-AUC: `0.9555`
- PR-AUC: `0.6800`
- weighted log loss: `0.1141`

From `model/ft_tf/results/test_metrics_ft_tf.json`:
- model: `ft_transformer`
- ROC-AUC: `0.9413`
- PR-AUC: `0.5776`
- weighted log loss: `0.3115`

Threshold summaries:
- Baseline (`threshold_report.json`): best validation threshold around `0.38` (weighted F1 policy).
- FT (`threshold_report_ft_tf.json`): best validation threshold around `0.82` (weighted F1 policy).

### 6.2 Segmentation

From `segment/results/segment_diagnostics.json` (base artifact snapshot):
- selected number of clusters: `6`
- silhouette: `0.3839`
- Davies-Bouldin: `1.0828`
- Calinski-Harabasz: `5195.896`
- stability pairwise agreement: `0.8836`

Final report note:
- The checked-in `segment/results/` directory is one baseline segmentation artifact snapshot.
- The final report compares the two reportable segmentation schemes instead: `V1 (k=5)` and classifier-informed `V2 (k=8)`.
- `V1` is the preferred client-facing segmentation, while `V2` is retained as a downstream activation variant.

Segment profiles and actionable recommendations are exported to:
- `segment/results/segment_profiles.csv`
- `segment/results/segment_marketing_actions.md`
- `segment/v2_from_ft_score/segment_profiles.csv`
- `segment/v2_from_ft_score/segment_marketing_actions.md`

## 7. Reproducibility Notes

- Default seed is `42` in training and segmentation scripts.
- Keep the same package versions and hardware setup for closest reproducibility.
- FT-Transformer results may vary more across hardware and GPU settings than tree-based baselines.
