from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.baseline.preprocess.split import make_splits
from model.baseline.models.classification import weighted_metrics
from model.ft_tf.dataio.dataset import TARGET_COL, WEIGHT_COL, clean_label_to_binary, load_raw_data
from model.ft_tf.models.classification import (
    build_model_from_artifact,
    predict_logits_from_parts,
    run_group_kfold_ft_tf,
    save_artifacts,
)


def main(
    data_path: str,
    columns_path: str,
    output_root: str,
    seed: int = 42,
    n_splits: int = 5,
    max_epochs: int = 6,
    batch_size: int = 512,
    max_train_rows: int = 0,
    max_valid_rows: int = 0,
    verbose: bool = True,
    sample_rows: int = 0,
    device: str = "cuda",
    gpu_ids: str = "",
    use_data_parallel: bool = True,
) -> None:
    print("[ft_tf] loading data")
    df = load_raw_data(data_path, columns_path)
    if sample_rows and sample_rows > 0 and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=seed).reset_index(drop=True)
        print(f"[ft_tf] sampled rows: {len(df)}")
    df["_row_id"] = np.arange(len(df))

    splits = make_splits(df, seed=seed)
    train_valid_df = pd.concat([splits["train"], splits["valid"]], axis=0).copy()
    test_df = splits["test"].copy()

    print("[ft_tf] running cv/final fit")
    gpu_id_list = [int(x) for x in gpu_ids.split(",") if x.strip() != ""]
    metrics_df, artifact = run_group_kfold_ft_tf(
        train_valid_df,
        seed=seed,
        n_splits=n_splits,
        training_params_override={
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "max_train_rows": max_train_rows,
            "max_valid_rows": max_valid_rows,
            "verbose": verbose,
            "device": device,
            "gpu_ids": gpu_id_list,
            "use_data_parallel": use_data_parallel,
        },
    )
    print("[ft_tf] building model from artifact")
    model, preprocessor = build_model_from_artifact(artifact)

    y_test = clean_label_to_binary(test_df[TARGET_COL]).to_numpy()
    w_test = pd.to_numeric(test_df[WEIGHT_COL], errors="coerce").fillna(1.0).to_numpy()
    print("[ft_tf] predicting test")
    logits = predict_logits_from_parts(
        model=model,
        preprocessor=preprocessor,
        df=test_df,
        batch_size=artifact.training_params["batch_size"],
        device=artifact.training_params.get("device", "cuda"),
    )
    test_metrics = weighted_metrics(y_test, logits, w_test)
    test_metrics["model"] = artifact.model_name

    out_dir = Path(output_root) / "model" / "ft_tf"
    print("[ft_tf] saving artifacts")
    save_artifacts(artifact, metrics_df, test_metrics, out_dir)

    probs = 1.0 / (1.0 + np.exp(-logits))
    threshold = 0.5
    pred = (probs >= threshold).astype(int)

    score_df = test_df.copy()
    score_df["true_label_binary"] = y_test
    score_df["logit"] = logits
    score_df["probability"] = probs
    score_df["predicted_label"] = pred
    score_df["threshold"] = threshold
    score_df["model_name"] = artifact.model_name
    score_df.to_csv(out_dir / "results" / "test_predictions.csv", index=False)
    print("[ft_tf] done")

    print(json.dumps({"model": artifact.model_name, "test_metrics": test_metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FT-Transformer classification pipeline.")
    parser.add_argument("--data", default="census-bureau.data")
    parser.add_argument("--columns", default="census-bureau.columns")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-valid-rows", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--no-data-parallel", action="store_true")
    args = parser.parse_args()
    main(
        args.data,
        args.columns,
        args.output_root,
        seed=args.seed,
        n_splits=args.n_splits,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        max_train_rows=args.max_train_rows,
        max_valid_rows=args.max_valid_rows,
        verbose=args.verbose,
        sample_rows=args.sample_rows,
        device=args.device,
        gpu_ids=args.gpu_ids,
        use_data_parallel=not args.no_data_parallel,
    )
