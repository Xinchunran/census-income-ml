from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from model.baseline.dataio.dataset import load_raw_data
from model.baseline.models.classification import predict_logits, sigmoid


def main(
    params_pkl: str,
    data_path: str,
    columns_path: str,
    output_path: str,
    threshold: float | None = None,
) -> None:
    artifact = joblib.load(params_pkl)
    df = load_raw_data(data_path, columns_path)
    logits = predict_logits(artifact, df)
    probs = sigmoid(logits)
    if threshold is not None:
        used_threshold = float(threshold)
    else:
        used_threshold = float(artifact.get("threshold", 0.5))

    out = pd.DataFrame(
        {
            "logit": logits,
            "probability": probs,
            "predicted_label": (probs >= used_threshold).astype(int),
            "threshold": used_threshold,
        }
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference with saved baseline artifact.")
    parser.add_argument("--params-pkl", required=True)
    parser.add_argument("--data", default="census-bureau.data")
    parser.add_argument("--columns", default="census-bureau.columns")
    parser.add_argument("--output", default="model/baseline/results/inference_predictions.csv")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    main(args.params_pkl, args.data, args.columns, args.output, args.threshold)
