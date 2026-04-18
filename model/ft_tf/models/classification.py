from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold

from model.ft_tf.models.ft_transformer_model import (
    FTTransformer,
    TrainingConfig,
    fit_ft_transformer,
    make_dataloader,
    predict_logits as nn_predict_logits,
)
from model.baseline.models.classification import weighted_metrics
from model.ft_tf.dataio.dataset import TARGET_COL, WEIGHT_COL, clean_label_to_binary
from model.ft_tf.preprocess.features import FTPreprocessor, build_preprocessor


@dataclass
class FTArtifact:
    model_name: str
    model_params: dict[str, Any]
    training_params: dict[str, Any]
    preprocessor: FTPreprocessor
    state_dict: dict[str, Any]


def _make_groups(df: pd.DataFrame) -> np.ndarray:
    if "_row_id" in df.columns:
        return df["_row_id"].to_numpy()
    return np.arange(len(df), dtype=int)


def _default_model_params() -> dict[str, Any]:
    return {
        "d_token": 32,
        "n_blocks": 2,
        "n_heads": 4,
        "attention_dropout": 0.1,
        "ffn_dropout": 0.1,
        "residual_dropout": 0.0,
        "head_hidden_dim": 32,
        "head_dropout": 0.1,
    }


def _default_training_params() -> dict[str, Any]:
    return {
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 512,
        "max_epochs": 6,
        "patience": 3,
        "grad_clip_norm": 1.0,
        "num_workers": 0,
        "verbose": False,
        "max_train_rows": 0,
        "max_valid_rows": 0,
        "max_eval_rows": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_ids": [],
        "use_data_parallel": True,
    }


def _build_model(preprocessor: FTPreprocessor, model_params: dict[str, Any]) -> FTTransformer:
    return FTTransformer(
        n_num_features=len(preprocessor.numeric_cols),
        cat_cardinalities=preprocessor.cat_cardinalities,
        **model_params,
    )


def _make_training_config(params: dict[str, Any], y: np.ndarray, w: np.ndarray) -> TrainingConfig:
    y = y.astype(float)
    w = w.astype(float)
    pos_weight = None
    pos = float((w * y).sum())
    neg = float((w * (1.0 - y)).sum())
    if pos > 0:
        pos_weight = neg / pos
    return TrainingConfig(
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
        batch_size=int(params["batch_size"]),
        max_epochs=int(params["max_epochs"]),
        patience=int(params["patience"]),
        grad_clip_norm=float(params["grad_clip_norm"]) if params["grad_clip_norm"] is not None else None,
        num_workers=int(params["num_workers"]),
        verbose=bool(params["verbose"]),
        positive_class_weight=pos_weight,
        device=str(params.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
    )


def _maybe_wrap_multi_gpu(model: FTTransformer, training_params: dict[str, Any]) -> FTTransformer | nn.DataParallel:
    use_dp = bool(training_params.get("use_data_parallel", True))
    if not use_dp:
        return model
    if not torch.cuda.is_available():
        return model
    gpu_ids = training_params.get("gpu_ids", [])
    if gpu_ids is None or len(gpu_ids) == 0:
        gpu_ids = list(range(torch.cuda.device_count()))
    gpu_ids = [int(x) for x in gpu_ids]
    if len(gpu_ids) <= 1:
        return model
    print(f"[ft_tf] using DataParallel on GPUs: {gpu_ids}")
    return nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])


def _fit_single_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    model_params: dict[str, Any],
    training_params: dict[str, Any],
) -> tuple[FTTransformer, FTPreprocessor]:
    print("[ft_tf] preparing fold data")
    max_train_rows = int(training_params.get("max_train_rows", 0))
    max_valid_rows = int(training_params.get("max_valid_rows", 0))
    if max_train_rows > 0 and len(train_df) > max_train_rows:
        train_df = train_df.sample(n=max_train_rows, random_state=42).copy()
    if max_valid_rows > 0 and len(valid_df) > max_valid_rows:
        valid_df = valid_df.sample(n=max_valid_rows, random_state=43).copy()

    print("[ft_tf] fitting preprocessor")
    preprocessor = build_preprocessor(train_df)
    x_num_tr, x_cat_tr = preprocessor.transform(train_df)
    x_num_va, x_cat_va = preprocessor.transform(valid_df)

    y_tr = clean_label_to_binary(train_df[TARGET_COL]).to_numpy(dtype=np.float32)
    y_va = clean_label_to_binary(valid_df[TARGET_COL]).to_numpy(dtype=np.float32)
    w_tr = pd.to_numeric(train_df[WEIGHT_COL], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    w_va = pd.to_numeric(valid_df[WEIGHT_COL], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)

    print("[ft_tf] building dataloaders")
    train_loader = make_dataloader(
        x_num=x_num_tr,
        x_cat=x_cat_tr,
        y=y_tr,
        sample_weight=w_tr,
        batch_size=int(training_params["batch_size"]),
        shuffle=True,
        num_workers=int(training_params["num_workers"]),
    )
    valid_loader = make_dataloader(
        x_num=x_num_va,
        x_cat=x_cat_va,
        y=y_va,
        sample_weight=w_va,
        batch_size=int(training_params["batch_size"]),
        shuffle=False,
        num_workers=int(training_params["num_workers"]),
    )

    print("[ft_tf] training model")
    model = _build_model(preprocessor, model_params)
    model = _maybe_wrap_multi_gpu(model, training_params)
    cfg = _make_training_config(training_params, y_tr, w_tr)
    model, _ = fit_ft_transformer(model, train_loader, valid_loader, cfg)
    print("[ft_tf] model trained")
    trained_model = model.module if isinstance(model, nn.DataParallel) else model
    return trained_model, preprocessor


def predict_logits_from_parts(
    model: FTTransformer,
    preprocessor: FTPreprocessor,
    df: pd.DataFrame,
    batch_size: int = 4096,
    num_workers: int = 0,
    device: str | None = None,
) -> np.ndarray:
    x_num, x_cat = preprocessor.transform(df)
    loader = make_dataloader(
        x_num=x_num,
        x_cat=x_cat,
        y=None,
        sample_weight=None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Ensure model and batch tensors are on the same device during inference.
    model = model.to(device)
    return nn_predict_logits(model, loader, device=device)


def run_group_kfold_ft_tf(
    train_valid_df: pd.DataFrame,
    seed: int = 42,
    n_splits: int = 5,
    model_params_override: dict[str, Any] | None = None,
    training_params_override: dict[str, Any] | None = None,
):
    y = clean_label_to_binary(train_valid_df[TARGET_COL])
    strat = y.astype(str) + "_" + train_valid_df["year"].astype(str)
    groups = _make_groups(train_valid_df)

    model_params = _default_model_params()
    training_params = _default_training_params()
    if model_params_override:
        model_params.update(model_params_override)
    if training_params_override:
        training_params.update(training_params_override)
    rows: list[dict[str, Any]] = []
    if n_splits >= 2:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = enumerate(cv.split(train_valid_df, strat, groups), start=1)
    else:
        idx = np.arange(len(train_valid_df))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        cut = int(len(idx) * 0.8)
        split_iter = [(1, (idx[:cut], idx[cut:]))]

    for fold_idx, (tr_idx, va_idx) in split_iter:
        tr_df = train_valid_df.iloc[tr_idx].copy()
        va_df = train_valid_df.iloc[va_idx].copy()
        print(f"[ft_tf] fold {fold_idx} fit start: train={len(tr_df)} valid={len(va_df)}")
        model, preprocessor = _fit_single_model(tr_df, va_df, model_params, training_params)
        print(f"[ft_tf] fold {fold_idx} fit done; scoring valid")
        eval_df = va_df
        max_eval_rows = int(training_params.get("max_eval_rows", 0))
        if max_eval_rows > 0 and len(eval_df) > max_eval_rows:
            eval_df = eval_df.sample(n=max_eval_rows, random_state=44).copy()
        logits = predict_logits_from_parts(model, preprocessor, eval_df, batch_size=training_params["batch_size"])
        y_va = clean_label_to_binary(eval_df[TARGET_COL]).to_numpy()
        w_va = pd.to_numeric(eval_df[WEIGHT_COL], errors="coerce").fillna(1.0).to_numpy()
        metrics = weighted_metrics(y_va, logits, w_va)
        metrics["model"] = "ft_transformer"
        metrics["fold"] = fold_idx
        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)

    # final fit on full train_valid (no separate valid here, use internal early stopping with held-out copy)
    shuffled = train_valid_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cut = int(len(shuffled) * 0.9)
    final_tr = shuffled.iloc[:cut].copy()
    final_va = shuffled.iloc[cut:].copy()
    print(f"[ft_tf] final fit start: train={len(final_tr)} valid={len(final_va)}")
    final_model, final_preprocessor = _fit_single_model(final_tr, final_va, model_params, training_params)
    print("[ft_tf] final fit done")

    artifact = FTArtifact(
        model_name="ft_transformer",
        model_params=model_params,
        training_params=training_params,
        preprocessor=final_preprocessor,
        state_dict=final_model.state_dict(),
    )
    return metrics_df, artifact


def build_model_from_artifact(artifact: FTArtifact | dict[str, Any]) -> tuple[FTTransformer, FTPreprocessor]:
    if isinstance(artifact, dict):
        preprocessor = FTPreprocessor.from_state(artifact["preprocessor_state"])
        model_params = artifact["model_params"]
        state_dict = artifact["model_state_dict"]
    else:
        preprocessor = artifact.preprocessor
        model_params = artifact.model_params
        state_dict = artifact.state_dict
    model = _build_model(preprocessor, model_params)
    model.load_state_dict(state_dict)
    model.eval()
    return model, preprocessor


def save_artifacts(artifact: FTArtifact, metrics_df: pd.DataFrame, test_metrics: dict[str, float], output_root: str | Path) -> None:
    output_root = Path(output_root)
    params_dir = output_root / "inference" / "params"
    results_dir = output_root / "results"
    params_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = artifact.model_name
    checkpoint_path = params_dir / f"{model_name}_artifact.pt"
    params_json_path = params_dir / f"{model_name}_best_params.json"

    torch.save(
        {
            "model_name": model_name,
            "model_params": artifact.model_params,
            "training_params": artifact.training_params,
            "preprocessor_state": artifact.preprocessor.to_state(),
            "model_state_dict": artifact.state_dict,
        },
        checkpoint_path,
    )
    params_json_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "model_params": artifact.model_params,
                "training_params": artifact.training_params,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    metrics_df.to_csv(results_dir / "kfold_metrics.csv", index=False)
    (results_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, sort_keys=True), encoding="utf-8")


def save_artifacts_with_suffix(
    artifact: FTArtifact,
    metrics_df: pd.DataFrame,
    test_metrics: dict[str, float],
    output_root: str | Path,
    file_suffix: str = "",
    threshold_info: dict[str, Any] | None = None,
) -> None:
    output_root = Path(output_root)
    params_dir = output_root / "inference" / "params"
    results_dir = output_root / "results"
    params_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    suffix = file_suffix if file_suffix.startswith("_") else (f"_{file_suffix}" if file_suffix else "")
    model_name = artifact.model_name
    checkpoint_path = params_dir / f"{model_name}_artifact{suffix}.pt"
    params_json_path = params_dir / f"{model_name}_best_params{suffix}.json"

    checkpoint = {
        "model_name": model_name,
        "model_params": artifact.model_params,
        "training_params": artifact.training_params,
        "preprocessor_state": artifact.preprocessor.to_state(),
        "model_state_dict": artifact.state_dict,
    }
    if threshold_info is not None:
        checkpoint["threshold"] = float(threshold_info["best_threshold_on_valid"]["threshold"])
        checkpoint["threshold_info"] = threshold_info

    torch.save(checkpoint, checkpoint_path)
    params_json_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "model_params": artifact.model_params,
                "training_params": artifact.training_params,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    metrics_df.to_csv(results_dir / f"kfold_metrics{suffix}.csv", index=False)
    (results_dir / f"test_metrics{suffix}.json").write_text(
        json.dumps(test_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if threshold_info is not None:
        (params_dir / f"{model_name}_threshold{suffix}.json").write_text(
            json.dumps(threshold_info, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (results_dir / f"threshold_report{suffix}.json").write_text(
            json.dumps(threshold_info, indent=2, sort_keys=True),
            encoding="utf-8",
        )
