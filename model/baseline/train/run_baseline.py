from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from model.baseline.dataio.dataset import TARGET_COL, WEIGHT_COL, clean_label_to_binary, load_raw_data
from model.baseline.models.classification import predict_logits, run_group_kfold_baseline, save_artifacts, weighted_metrics
from model.baseline.preprocess.split import make_splits


def _suffix_token(file_suffix: str) -> str:
    if not file_suffix:
        return ""
    return file_suffix if file_suffix.startswith("_") else f"_{file_suffix}"


def _select_threshold_by_weighted_f1(
    y_true: np.ndarray,
    probs: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    best = None
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (probs >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true,
            pred,
            average="binary",
            sample_weight=weights,
            zero_division=0,
        )
        candidate = {
            "threshold": float(round(thr, 4)),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        }
        if best is None or candidate["f1"] > best["f1"]:
            best = candidate
    return best if best is not None else {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def main(
    data_path: str,
    columns_path: str,
    output_root: str,
    seed: int = 42,
    n_splits: int = 5,
    file_suffix: str = "",
    parallel_search: bool = True,
    max_workers: int | None = None,
    threshold: float | None = None,
) -> None:
    df = load_raw_data(data_path, columns_path)
    df["_row_id"] = np.arange(len(df))

    splits = make_splits(df, seed=seed)
    train_valid_df = pd.concat([splits["train"], splits["valid"]], axis=0).copy()
    test_df = splits["test"].copy()

    metrics_df, best_model, artifact = run_group_kfold_baseline(
        train_valid_df,
        seed=seed,
        n_splits=n_splits,
        parallel_search=parallel_search,
        max_workers=max_workers,
    )
    # Threshold policy:
    # - manual threshold if provided
    # - else search weighted-F1 optimal threshold on validation split
    valid_df = splits["valid"].copy()
    y_valid = clean_label_to_binary(valid_df[TARGET_COL]).to_numpy()
    w_valid = pd.to_numeric(valid_df[WEIGHT_COL], errors="coerce").fillna(1.0).to_numpy()
    valid_logits = predict_logits(artifact, valid_df)
    valid_probs = 1.0 / (1.0 + np.exp(-valid_logits))

    threshold_source = "manual_input" if threshold is not None else "weighted_f1_on_valid"
    if threshold is None:
        best_valid_thr = _select_threshold_by_weighted_f1(y_valid, valid_probs, w_valid)
        threshold = float(best_valid_thr["threshold"])
    else:
        p, r, f1, _ = precision_recall_fscore_support(
            y_valid,
            (valid_probs >= threshold).astype(int),
            average="binary",
            sample_weight=w_valid,
            zero_division=0,
        )
        best_valid_thr = {
            "threshold": float(threshold),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        }

    y_test = clean_label_to_binary(test_df[TARGET_COL]).to_numpy()
    w_test = pd.to_numeric(test_df[WEIGHT_COL], errors="coerce").fillna(1.0).to_numpy()
    logits = predict_logits(artifact, test_df)
    test_metrics = weighted_metrics(y_test, logits, w_test)
    test_metrics["model"] = best_model

    out_dir = Path(output_root) / "model" / "baseline"
    threshold_info = {
        "model": best_model,
        "selection_metric": "weighted_f1",
        "selection_source": threshold_source,
        "best_threshold_on_valid": best_valid_thr,
    }
    artifact["threshold"] = float(threshold)
    artifact["threshold_info"] = threshold_info
    save_artifacts(
        artifact,
        metrics_df,
        test_metrics,
        out_dir,
        file_suffix=file_suffix,
        threshold_info=threshold_info,
    )

    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= threshold).astype(int)
    suffix = _suffix_token(file_suffix)
    # Keep full test input for downstream error analysis.
    score_df = test_df.copy()
    score_df["true_label_binary"] = y_test
    score_df["logit"] = logits
    score_df["probability"] = probs
    score_df["predicted_label"] = pred
    score_df["threshold"] = threshold
    score_df["model_name"] = best_model
    score_df.to_csv(out_dir / "results" / f"test_predictions{suffix}.csv", index=False)

    print(json.dumps({"best_model": best_model, "test_metrics": test_metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline binary classification pipeline.")
    parser.add_argument("--data", default="census-bureau.data")
    parser.add_argument("--columns", default="census-bureau.columns")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--file-suffix", default="")
    parser.add_argument("--no-parallel-search", action="store_true")
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    main(
        args.data,
        args.columns,
        args.output_root,
        seed=args.seed,
        n_splits=args.n_splits,
        file_suffix=args.file_suffix,
        parallel_search=not args.no_parallel_search,
        max_workers=None if args.max_workers <= 0 else args.max_workers,
        threshold=args.threshold,
    )
