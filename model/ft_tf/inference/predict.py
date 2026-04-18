from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.ft_tf.dataio.dataset import load_raw_data
from model.ft_tf.models.classification import build_model_from_artifact, predict_logits_from_parts


def main(
    checkpoint_path: str,
    data_path: str,
    columns_path: str,
    output_path: str,
    threshold: float = 0.5,
    device: str = "cuda",
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model, preprocessor = build_model_from_artifact(checkpoint)
    df = load_raw_data(data_path, columns_path)
    logits = predict_logits_from_parts(model=model, preprocessor=preprocessor, df=df, batch_size=4096, device=device)
    probs = 1.0 / (1.0 + np.exp(-logits))

    out = pd.DataFrame(
        {
            "logit": logits,
            "probability": probs,
            "predicted_label": (probs >= threshold).astype(int),
            "threshold": threshold,
            "model_name": checkpoint.get("model_name", "ft_transformer"),
        }
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference with saved FT-Transformer checkpoint (.pt).")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="census-bureau.data")
    parser.add_argument("--columns", default="census-bureau.columns")
    parser.add_argument("--output", default="model/ft_tf/results/inference_predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args.checkpoint, args.data, args.columns, args.output, args.threshold, args.device)
