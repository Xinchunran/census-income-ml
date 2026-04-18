from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from model.baseline.dataio.dataset import WEIGHT_COL, clean_label_to_binary, load_raw_data
from model.baseline.models.classification import predict_logits
from model.baseline.preprocess.split import make_splits


def main(root_dir: str, artifact_path: str, seed: int = 42) -> None:
    root = Path(root_dir)
    artifact = joblib.load(artifact_path)

    df = load_raw_data(root / "census-bureau.data", root / "census-bureau.columns")
    df["_row_id"] = np.arange(len(df))
    splits = make_splits(df, seed=seed)
    valid_df = splits["valid"].copy()
    test_df = splits["test"].copy()

    y_valid = clean_label_to_binary(valid_df["label"]).to_numpy()
    w_valid = pd.to_numeric(valid_df[WEIGHT_COL], errors="coerce").fillna(1.0).to_numpy()
    valid_probs = 1.0 / (1.0 + np.exp(-predict_logits(artifact, valid_df)))

    best = None
    for thr in np.linspace(0.05, 0.95, 181):
        pred = (valid_probs >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_valid,
            pred,
            average="binary",
            sample_weight=w_valid,
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

    y_test = clean_label_to_binary(test_df["label"]).to_numpy()
    w_test = pd.to_numeric(test_df[WEIGHT_COL], errors="coerce").fillna(1.0).to_numpy()
    test_probs = 1.0 / (1.0 + np.exp(-predict_logits(artifact, test_df)))
    test_pred = (test_probs >= best["threshold"]).astype(int)
    p_t, r_t, f1_t, _ = precision_recall_fscore_support(
        y_test,
        test_pred,
        average="binary",
        sample_weight=w_test,
        zero_division=0,
    )

    report = {
        "model": artifact["model_name"],
        "selection_metric": "weighted_f1_on_valid",
        "best_threshold": best,
        "test_at_best_threshold": {"precision": float(p_t), "recall": float(r_t), "f1": float(f1_t)},
    }

    params_out = root / "model" / "baseline" / "inference" / "params" / f"{artifact['model_name']}_threshold.json"
    results_out = root / "model" / "baseline" / "results" / "threshold_report.json"
    params_out.write_text(
        json.dumps({"threshold": best["threshold"], "selection_metric": "weighted_f1_on_valid"}, indent=2),
        encoding="utf-8",
    )
    results_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune classification threshold on validation split.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument(
        "--artifact-path",
        default="model/baseline/inference/params/lightgbm_artifact.pkl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.root_dir, args.artifact_path, args.seed)
