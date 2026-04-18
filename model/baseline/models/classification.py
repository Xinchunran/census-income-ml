from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from model.baseline.dataio.dataset import TARGET_COL, WEIGHT_COL, clean_label_to_binary, get_feature_columns
from model.baseline.preprocess.features import build_preprocessor

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def weighted_metrics(y_true: np.ndarray, logits: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    probs = sigmoid(logits)
    return {
        "roc_auc": float(roc_auc_score(y_true, probs, sample_weight=weights)),
        "pr_auc": float(average_precision_score(y_true, probs, sample_weight=weights)),
        "weighted_log_loss": float(log_loss(y_true, probs, sample_weight=weights, labels=[0, 1])),
    }


def _make_groups(df: pd.DataFrame) -> np.ndarray:
    if "_row_id" in df.columns:
        return df["_row_id"].to_numpy()
    return np.arange(len(df), dtype=int)


def _param_grids(seed: int, n_jobs_per_model: int) -> dict[str, list[dict[str, Any]]]:
    grids: dict[str, list[dict[str, Any]]] = {}
    if CatBoostClassifier is not None:
        grids["catboost"] = [
            {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "depth": 6,
                "learning_rate": 0.06,
                "iterations": 260,
                "l2_leaf_reg": 4.0,
                "random_seed": seed,
                "verbose": False,
                "allow_writing_files": False,
            },
            {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "depth": 8,
                "learning_rate": 0.05,
                "iterations": 320,
                "l2_leaf_reg": 6.0,
                "random_seed": seed,
                "verbose": False,
                "allow_writing_files": False,
            },
            {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "depth": 10,
                "learning_rate": 0.04,
                "iterations": 360,
                "l2_leaf_reg": 8.0,
                "random_seed": seed,
                "verbose": False,
                "allow_writing_files": False,
            },
        ]
    if XGBClassifier is not None:
        grids["xgboost"] = [
            {
                "objective": "binary:logistic",
                "n_estimators": 220,
                "max_depth": 6,
                "learning_rate": 0.07,
                "subsample": 0.9,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "min_child_weight": 1.0,
                "random_state": seed,
                "n_jobs": n_jobs_per_model,
                "eval_metric": "auc",
            },
            {
                "objective": "binary:logistic",
                "n_estimators": 300,
                "max_depth": 7,
                "learning_rate": 0.05,
                "subsample": 0.85,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.0,
                "reg_lambda": 1.2,
                "min_child_weight": 2.0,
                "random_state": seed,
                "n_jobs": n_jobs_per_model,
                "eval_metric": "auc",
            },
            {
                "objective": "binary:logistic",
                "n_estimators": 360,
                "max_depth": 8,
                "learning_rate": 0.04,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.2,
                "reg_lambda": 1.5,
                "min_child_weight": 1.0,
                "random_state": seed,
                "n_jobs": n_jobs_per_model,
                "eval_metric": "auc",
            },
        ]
    if LGBMClassifier is not None:
        grids["lightgbm"] = [
            {
                "objective": "binary",
                "n_estimators": 220,
                "learning_rate": 0.07,
                "num_leaves": 63,
                "subsample": 0.9,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "random_state": seed,
                "n_jobs": n_jobs_per_model,
            },
            {
                "objective": "binary",
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 95,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.1,
                "reg_lambda": 1.2,
                "random_state": seed,
                "n_jobs": n_jobs_per_model,
            },
            {
                "objective": "binary",
                "n_estimators": 360,
                "learning_rate": 0.04,
                "num_leaves": 127,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.0,
                "reg_lambda": 1.5,
                "random_state": seed,
                "n_jobs": n_jobs_per_model,
            },
        ]
    if not grids:
        raise ImportError("catboost/xgboost/lightgbm 都不可用，请先安装后再训练 baseline。")
    return grids


def _instantiate_model(model_name: str, params: dict[str, Any]) -> Any:
    if model_name == "catboost" and CatBoostClassifier is not None:
        return CatBoostClassifier(**params)
    if model_name == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(**params)
    if model_name == "lightgbm" and LGBMClassifier is not None:
        return LGBMClassifier(**params)
    raise ValueError(f"Unsupported model: {model_name}")


def _resolve_parallel_settings(total_candidates: int, max_workers: int | None) -> tuple[int, int]:
    cpu_count = max(1, os.cpu_count() or 1)
    workers = total_candidates if max_workers is None else max_workers
    workers = max(1, min(workers, total_candidates, cpu_count))
    n_jobs_per_model = max(1, cpu_count // workers)
    return workers, n_jobs_per_model


def _fit_and_predict_logits(model_name: str, model: Any, train_df: pd.DataFrame, valid_df: pd.DataFrame, w_train: np.ndarray):
    if model_name == "catboost":
        feat_cols = get_feature_columns(train_df)
        cat_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(train_df[c])]
        X_train = train_df[feat_cols].copy()
        X_valid = valid_df[feat_cols].copy()
        for c in cat_cols:
            X_train[c] = X_train[c].astype(object).where(X_train[c].notna(), "__NA__").astype(str)
            X_valid[c] = X_valid[c].astype(object).where(X_valid[c].notna(), "__NA__").astype(str)
        y_train = clean_label_to_binary(train_df[TARGET_COL]).to_numpy()
        model.fit(X_train, y_train, sample_weight=w_train, cat_features=cat_cols)
        logits = model.predict(X_valid, prediction_type="RawFormulaVal")
        return np.asarray(logits, dtype=float), None

    preprocessor = build_preprocessor(train_df)
    X_train = preprocessor.transform(train_df)
    X_valid = preprocessor.transform(valid_df)
    y_train = clean_label_to_binary(train_df[TARGET_COL]).to_numpy()
    model.fit(X_train, y_train, sample_weight=w_train)
    if model_name == "xgboost":
        return np.asarray(model.predict(X_valid, output_margin=True), dtype=float), preprocessor
    probs = model.predict_proba(X_valid)[:, 1]
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    logits = np.log(probs / (1.0 - probs))
    return logits, preprocessor


def _evaluate_candidate(
    train_valid_df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
    candidate_id: int,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
        tr_df = train_valid_df.iloc[tr_idx]
        va_df = train_valid_df.iloc[va_idx]
        y_va = clean_label_to_binary(va_df[TARGET_COL]).to_numpy()
        w_tr = pd.to_numeric(tr_df[WEIGHT_COL], errors="coerce").fillna(1.0).to_numpy()
        w_va = pd.to_numeric(va_df[WEIGHT_COL], errors="coerce").fillna(1.0).to_numpy()

        model = _instantiate_model(model_name, params)
        logits, _ = _fit_and_predict_logits(model_name, model, tr_df, va_df, w_tr)
        metrics = weighted_metrics(y_va, logits, w_va)
        metrics["model"] = model_name
        metrics["candidate_id"] = candidate_id
        metrics["model_variant"] = f"{model_name}_g{candidate_id}"
        metrics["fold"] = fold_idx
        rows.append(metrics)
    return rows


def run_group_kfold_baseline(
    train_valid_df: pd.DataFrame,
    seed: int = 42,
    n_splits: int = 5,
    parallel_search: bool = True,
    max_workers: int | None = None,
):
    y = clean_label_to_binary(train_valid_df[TARGET_COL])
    strat = y.astype(str) + "_" + train_valid_df["year"].astype(str)
    groups = _make_groups(train_valid_df)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    split_list = list(cv.split(train_valid_df, strat, groups))
    grids_probe = _param_grids(seed, n_jobs_per_model=1)
    total_candidates = sum(len(v) for v in grids_probe.values())
    workers, n_jobs_per_model = _resolve_parallel_settings(total_candidates, max_workers)
    grids = _param_grids(seed, n_jobs_per_model=n_jobs_per_model)

    candidate_jobs: list[tuple[str, int, dict[str, Any]]] = []
    for model_name, candidate_params in grids.items():
        for candidate_id, params in enumerate(candidate_params, start=1):
            candidate_jobs.append((model_name, candidate_id, params))

    rows: list[dict[str, Any]] = []
    if parallel_search and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_evaluate_candidate, train_valid_df, split_list, model_name, candidate_id, params)
                for model_name, candidate_id, params in candidate_jobs
            ]
            for fut in as_completed(futures):
                rows.extend(fut.result())
    else:
        for model_name, candidate_id, params in candidate_jobs:
            rows.extend(_evaluate_candidate(train_valid_df, split_list, model_name, candidate_id, params))

    metrics_df = pd.DataFrame(rows)
    summary = (
        metrics_df.groupby(["model", "candidate_id", "model_variant"])[["roc_auc", "pr_auc", "weighted_log_loss"]]
        .mean()
        .reset_index()
    )
    best_row = summary.sort_values(["pr_auc", "roc_auc"], ascending=False).iloc[0]
    best_model = str(best_row["model"])
    best_candidate_id = int(best_row["candidate_id"])
    best_params = grids[best_model][best_candidate_id - 1]
    best_model_obj = _instantiate_model(best_model, best_params)
    y_full = clean_label_to_binary(train_valid_df[TARGET_COL]).to_numpy()
    w_full = pd.to_numeric(train_valid_df[WEIGHT_COL], errors="coerce").fillna(1.0).to_numpy()
    if best_model == "catboost":
        feat_cols = get_feature_columns(train_valid_df)
        cat_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(train_valid_df[c])]
        X_full = train_valid_df[feat_cols].copy()
        for c in cat_cols:
            X_full[c] = X_full[c].astype(object).where(X_full[c].notna(), "__NA__").astype(str)
        best_model_obj.fit(X_full, y_full, sample_weight=w_full, cat_features=cat_cols)
        artifact = {
            "model_name": best_model,
            "model_variant": f"{best_model}_g{best_candidate_id}",
            "model": best_model_obj,
            "preprocessor": None,
            "params": best_params,
            "feature_names": feat_cols,
        }
    else:
        preprocessor = build_preprocessor(train_valid_df)
        X_full = preprocessor.transform(train_valid_df)
        best_model_obj.fit(X_full, y_full, sample_weight=w_full)
        feature_names = []
        if hasattr(preprocessor.transformer, "get_feature_names_out"):
            feature_names = list(preprocessor.transformer.get_feature_names_out())
        artifact = {
            "model_name": best_model,
            "model_variant": f"{best_model}_g{best_candidate_id}",
            "model": best_model_obj,
            "preprocessor": preprocessor,
            "params": best_params,
            "feature_names": feature_names,
        }
    return metrics_df, best_model, artifact


def predict_logits(artifact: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    model_name = artifact["model_name"]
    model = artifact["model"]
    if model_name == "catboost":
        feat_cols = [c for c in df.columns if c not in {TARGET_COL, WEIGHT_COL}]
        X = df[feat_cols].copy()
        cat_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(df[c])]
        for c in cat_cols:
            X[c] = X[c].astype(object).where(X[c].notna(), "__NA__").astype(str)
        return np.asarray(model.predict(X, prediction_type="RawFormulaVal"), dtype=float)
    preprocessor = artifact["preprocessor"]
    X = preprocessor.transform(df)
    if model_name == "xgboost":
        return np.asarray(model.predict(X, output_margin=True), dtype=float)
    probs = model.predict_proba(X)[:, 1]
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return np.log(probs / (1.0 - probs))


def _suffix_token(file_suffix: str) -> str:
    if not file_suffix:
        return ""
    return file_suffix if file_suffix.startswith("_") else f"_{file_suffix}"


def save_artifacts(
    artifact: dict[str, Any],
    metrics_df: pd.DataFrame,
    test_metrics: dict[str, float],
    output_root: str | Path,
    file_suffix: str = "",
    threshold_info: dict[str, Any] | None = None,
):
    output_root = Path(output_root)
    params_dir = output_root / "inference" / "params"
    results_dir = output_root / "results"
    params_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = _suffix_token(file_suffix)

    model_name = artifact["model_name"]
    (params_dir / f"{model_name}_best_params{suffix}.json").write_text(
        json.dumps(artifact["params"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    joblib.dump(artifact, params_dir / f"{model_name}_artifact{suffix}.pkl")
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

    summary = (
        metrics_df.groupby(["model", "candidate_id", "model_variant"])[["roc_auc", "pr_auc", "weighted_log_loss"]]
        .mean()
        .reset_index()
        .sort_values(["pr_auc", "roc_auc"], ascending=False)
    )
    summary.to_csv(results_dir / f"grid_search_summary{suffix}.csv", index=False)
    (results_dir / f"grid_search_summary{suffix}.json").write_text(
        summary.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    fi_path = results_dir / f"{model_name}_feature_importance{suffix}.csv"
    model = artifact["model"]
    if model_name == "catboost":
        model.get_feature_importance(prettified=True).to_csv(fi_path, index=False)
        return
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    names = artifact.get("feature_names", [])
    features = names if len(names) == len(importances) else list(range(len(importances)))
    pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False).to_csv(
        fi_path, index=False
    )
