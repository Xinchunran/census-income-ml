from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from model.baseline.dataio.dataset import load_raw_data
from model.baseline.models.classification import predict_logits as baseline_predict_logits
from model.baseline.preprocess.split import make_splits
from model.ft_tf.models.classification import build_model_from_artifact, predict_logits_from_parts
from segment.pipeline_v2 import run_segmentation_v2


def _build_scores_from_prediction_file(
    project_root: Path,
    prediction_path: Path,
    seed: int,
    score_column_in_file: str = "probability",
    score_source: str = "prediction_file",
) -> pd.DataFrame:
    pred_df = pd.read_csv(prediction_path)
    if score_column_in_file not in pred_df.columns:
        raise ValueError(f"Prediction file missing `{score_column_in_file}`: {prediction_path}")

    df = load_raw_data(project_root / "census-bureau.data", project_root / "census-bureau.columns")
    df["_row_id"] = np.arange(len(df))
    if len(pred_df) != len(df):
        raise ValueError(f"Prediction file row count mismatch: file={len(pred_df)} data={len(df)}")

    pred_df = pred_df.reset_index(drop=True)
    pred_df["_row_id"] = np.arange(len(pred_df))

    splits = make_splits(df, seed=seed)
    split_map: dict[int, str] = {}
    for split_name, split_df in splits.items():
        for rid in split_df["_row_id"].to_numpy():
            split_map[int(rid)] = split_name

    out = pd.DataFrame(
        {
            "row_id": pred_df["_row_id"].astype(int),
            "split": pred_df["_row_id"].astype(int).map(split_map),
            "income_score": pd.to_numeric(pred_df[score_column_in_file], errors="coerce"),
            "score_source": score_source,
        }
    )
    if out["split"].isna().any() or out["income_score"].isna().any():
        raise ValueError(f"Failed to build valid score table from prediction file: {prediction_path}")
    return out


def _build_baseline_scores(project_root: Path, artifact_path: Path, seed: int) -> pd.DataFrame:
    artifact = joblib.load(artifact_path)
    df = load_raw_data(project_root / "census-bureau.data", project_root / "census-bureau.columns")
    df["_row_id"] = np.arange(len(df))
    splits = make_splits(df, seed=seed)

    rows: list[pd.DataFrame] = []
    for split_name, split_df in splits.items():
        logits = baseline_predict_logits(artifact, split_df)
        probs = 1.0 / (1.0 + np.exp(-logits))
        rows.append(
            pd.DataFrame(
                {
                    "row_id": split_df["_row_id"].to_numpy(),
                    "split": split_name,
                    "income_score": probs,
                    "score_source": f"baseline:{artifact.get('model_name', 'unknown')}",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _build_ft_tf_scores(project_root: Path, checkpoint_path: Path, seed: int, device: str) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model, preprocessor = build_model_from_artifact(checkpoint)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_raw_data(project_root / "census-bureau.data", project_root / "census-bureau.columns")
    df["_row_id"] = np.arange(len(df))
    splits = make_splits(df, seed=seed)

    rows: list[pd.DataFrame] = []
    for split_name, split_df in splits.items():
        logits = predict_logits_from_parts(model, preprocessor, split_df, batch_size=4096, device=device)
        probs = 1.0 / (1.0 + np.exp(-logits))
        rows.append(
            pd.DataFrame(
                {
                    "row_id": split_df["_row_id"].to_numpy(),
                    "split": split_name,
                    "income_score": probs,
                    "score_source": f"ft_tf:{checkpoint.get('model_name', 'ft_transformer')}",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two classifier-informed segmentation variants (baseline + ft_tf).")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="segment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-artifact", default="model/baseline/inference/params/lightgbm_artifact.pkl")
    parser.add_argument("--ft-checkpoint", default="model/ft_tf/inference/params/ft_transformer_artifact_ft_tf.pt")
    parser.add_argument("--baseline-prediction-file", default="model/baseline/results/inference_predictions_grid.csv")
    parser.add_argument("--ft-prediction-file", default="model/ft_tf/results/inference_predictions_ft_tf.csv")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_pred_path = project_root / args.baseline_prediction_file
    if baseline_pred_path.exists():
        baseline_score_df = _build_scores_from_prediction_file(
            project_root=project_root,
            prediction_path=baseline_pred_path,
            seed=args.seed,
            score_column_in_file="probability",
            score_source=f"baseline_prediction_file:{baseline_pred_path.name}",
        )
    else:
        baseline_score_df = _build_baseline_scores(project_root, project_root / args.baseline_artifact, seed=args.seed)
    baseline_score_path = output_root / "baseline_informed_results" / "score_table.csv"
    baseline_score_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_score_df.to_csv(baseline_score_path, index=False)
    baseline_diag = run_segmentation_v2(
        project_root=project_root,
        output_root=output_root,
        score_path=baseline_score_path,
        random_state=args.seed,
        score_column="income_score",
        output_subdir="baseline_informed_results",
    )

    ft_pred_path = project_root / args.ft_prediction_file
    if ft_pred_path.exists():
        ft_score_df = _build_scores_from_prediction_file(
            project_root=project_root,
            prediction_path=ft_pred_path,
            seed=args.seed,
            score_column_in_file="probability",
            score_source=f"ft_tf_prediction_file:{ft_pred_path.name}",
        )
    else:
        ft_checkpoint = project_root / args.ft_checkpoint
        if not ft_checkpoint.exists():
            fallback = project_root / "model/ft_tf/inference/params/ft_transformer_artifact.pt"
            if fallback.exists():
                ft_checkpoint = fallback
        ft_score_df = _build_ft_tf_scores(project_root, ft_checkpoint, seed=args.seed, device=args.device)
    ft_score_path = output_root / "ft_tf_informed_results" / "score_table.csv"
    ft_score_path.parent.mkdir(parents=True, exist_ok=True)
    ft_score_df.to_csv(ft_score_path, index=False)
    ft_diag = run_segmentation_v2(
        project_root=project_root,
        output_root=output_root,
        score_path=ft_score_path,
        random_state=args.seed,
        score_column="income_score",
        output_subdir="ft_tf_informed_results",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "baseline_output": str(output_root / "baseline_informed_results"),
                "ft_tf_output": str(output_root / "ft_tf_informed_results"),
                "baseline_n_clusters": baseline_diag.get("n_clusters"),
                "ft_tf_n_clusters": ft_diag.get("n_clusters"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
