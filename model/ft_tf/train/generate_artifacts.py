from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from model.baseline.models.classification import weighted_metrics
from model.baseline.preprocess.split import make_splits
from model.ft_tf.dataio.dataset import TARGET_COL, WEIGHT_COL, clean_label_to_binary, load_raw_data
from model.ft_tf.models.classification import (
    build_model_from_artifact,
    predict_logits_from_parts,
    run_group_kfold_ft_tf,
    save_artifacts,
)


def main() -> None:
    print("[gen] load")
    df = load_raw_data("census-bureau.data", "census-bureau.columns")
    df["_row_id"] = np.arange(len(df))
    splits = make_splits(df, seed=42)
    train_valid_df = pd.concat([splits["train"], splits["valid"]], axis=0).copy()
    test_df = splits["test"].copy()

    print("[gen] fit")
    metrics_df, artifact = run_group_kfold_ft_tf(
        train_valid_df,
        seed=42,
        n_splits=1,
        training_params_override={
            "max_epochs": 0,
            "batch_size": 256,
            "max_train_rows": 1000,
            "max_valid_rows": 1000,
            "max_eval_rows": 1000,
            "verbose": True,
        },
    )

    print("[gen] predict")
    model, preprocessor = build_model_from_artifact(artifact)
    logits = predict_logits_from_parts(model, preprocessor, test_df, batch_size=1024)

    y_test = clean_label_to_binary(test_df[TARGET_COL]).to_numpy()
    w_test = pd.to_numeric(test_df[WEIGHT_COL], errors="coerce").fillna(1.0).to_numpy()
    test_metrics = weighted_metrics(y_test, logits, w_test)
    test_metrics["model"] = artifact.model_name

    print("[gen] save")
    out_dir = Path("model/ft_tf")
    save_artifacts(artifact, metrics_df, test_metrics, out_dir)

    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= 0.5).astype(int)
    score_df = test_df.copy()
    score_df["true_label_binary"] = y_test
    score_df["logit"] = logits
    score_df["probability"] = probs
    score_df["predicted_label"] = pred
    score_df["threshold"] = 0.5
    score_df["model_name"] = artifact.model_name
    score_df.to_csv(out_dir / "results" / "test_predictions.csv", index=False)
    print(json.dumps({"model": artifact.model_name, "test_metrics": test_metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
