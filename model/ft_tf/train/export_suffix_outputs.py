from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support

from model.baseline.dataio.dataset import load_raw_data as load_baseline_raw
from model.baseline.models.classification import weighted_metrics
from model.baseline.preprocess.split import make_splits
from model.ft_tf.dataio.dataset import TARGET_COL, WEIGHT_COL, clean_label_to_binary, load_raw_data
from model.ft_tf.models.classification import build_model_from_artifact, predict_logits_from_parts


def _select_threshold_by_weighted_f1(y_true: np.ndarray, probs: np.ndarray, weights: np.ndarray) -> dict[str, float]:
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


def main(project_root: str, suffix: str, seed: int = 42) -> None:
    root = Path(project_root)
    suffix_token = suffix if suffix.startswith("_") else f"_{suffix}"
    model_dir = root / "model" / "ft_tf"
    params_dir = model_dir / "inference" / "params"
    results_dir = model_dir / "results"
    params_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = params_dir / "ft_transformer_artifact.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model, preprocessor = build_model_from_artifact(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    baseline_df = load_baseline_raw(root / "census-bureau.data", root / "census-bureau.columns")
    ft_df = load_raw_data(root / "census-bureau.data", root / "census-bureau.columns")
    ft_df["_row_id"] = np.arange(len(ft_df))
    splits = make_splits(ft_df, seed=seed)

    valid_df = splits["valid"].copy()
    y_valid = clean_label_to_binary(valid_df[TARGET_COL]).to_numpy()
    w_valid = pd.to_numeric(valid_df[WEIGHT_COL], errors="coerce").fillna(1.0).to_numpy()
    valid_logits = predict_logits_from_parts(model, preprocessor, valid_df, batch_size=4096, device=device)
    valid_probs = 1.0 / (1.0 + np.exp(-valid_logits))
    best_valid = _select_threshold_by_weighted_f1(y_valid, valid_probs, w_valid)
    threshold = float(best_valid["threshold"])

    test_df = splits["test"].copy()
    y_test = clean_label_to_binary(test_df[TARGET_COL]).to_numpy()
    w_test = pd.to_numeric(test_df[WEIGHT_COL], errors="coerce").fillna(1.0).to_numpy()
    test_logits = predict_logits_from_parts(model, preprocessor, test_df, batch_size=4096, device=device)
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    test_pred = (test_probs >= threshold).astype(int)
    test_metrics = weighted_metrics(y_test, test_logits, w_test)
    test_metrics["model"] = checkpoint.get("model_name", "ft_transformer")

    threshold_info = {
        "model": checkpoint.get("model_name", "ft_transformer"),
        "selection_metric": "weighted_f1",
        "selection_source": "weighted_f1_on_valid",
        "best_threshold_on_valid": best_valid,
        "dataset_consistency": {
            "same_shape": baseline_df.shape == ft_df.drop(columns=["_row_id"]).shape,
            "same_columns": list(baseline_df.columns) == [c for c in ft_df.columns if c != "_row_id"],
            "rows": int(len(ft_df)),
            "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        },
    }

    # save suffixed checkpoint/params
    checkpoint_out = dict(checkpoint)
    checkpoint_out["threshold"] = threshold
    checkpoint_out["threshold_info"] = threshold_info
    model_name = checkpoint.get("model_name", "ft_transformer")
    torch.save(checkpoint_out, params_dir / f"{model_name}_artifact{suffix_token}.pt")
    (params_dir / f"{model_name}_best_params{suffix_token}.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "model_params": checkpoint.get("model_params", {}),
                "training_params": checkpoint.get("training_params", {}),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (params_dir / f"{model_name}_threshold{suffix_token}.json").write_text(
        json.dumps(threshold_info, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # save suffixed outputs
    if (results_dir / "kfold_metrics.csv").exists():
        pd.read_csv(results_dir / "kfold_metrics.csv").to_csv(results_dir / f"kfold_metrics{suffix_token}.csv", index=False)
    (results_dir / f"test_metrics{suffix_token}.json").write_text(
        json.dumps(test_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (results_dir / f"threshold_report{suffix_token}.json").write_text(
        json.dumps(threshold_info, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    test_output = test_df.copy()
    test_output["true_label_binary"] = y_test
    test_output["logit"] = test_logits
    test_output["probability"] = test_probs
    test_output["predicted_label"] = test_pred
    test_output["threshold"] = threshold
    test_output["model_name"] = model_name
    test_output.to_csv(results_dir / f"test_predictions{suffix_token}.csv", index=False)

    full_logits = predict_logits_from_parts(model, preprocessor, ft_df, batch_size=4096, device=device)
    full_probs = 1.0 / (1.0 + np.exp(-full_logits))
    inference_output = pd.DataFrame(
        {
            "logit": full_logits,
            "probability": full_probs,
            "predicted_label": (full_probs >= threshold).astype(int),
            "threshold": threshold,
            "model_name": model_name,
        }
    )
    inference_output.to_csv(results_dir / f"inference_predictions{suffix_token}.csv", index=False)

    print(
        json.dumps(
            {
                "status": "ok",
                "suffix": suffix_token,
                "model": model_name,
                "threshold": threshold,
                "dataset_consistency": threshold_info["dataset_consistency"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ft_tf outputs with a suffix using existing checkpoint.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--suffix", default="ft_tf")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.project_root, args.suffix, args.seed)
