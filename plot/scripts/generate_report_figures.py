from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.baseline.dataio.dataset import clean_label_to_binary, load_raw_data
from model.baseline.models.classification import predict_logits, sigmoid
from model.baseline.preprocess.split import make_splits


FIG_DIR = ROOT / "plot" / "fig"
FIGSIZE = (3.5, 3.5)
FONT_SIZE = 12
BOOTSTRAP_N = 200
BOOTSTRAP_SEED = 42


plt.rcParams.update(
    {
        "figure.figsize": FIGSIZE,
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": 9,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
)


MODEL_SPECS = {
    "LightGBM": {
        "path": ROOT / "model" / "baseline" / "results" / "test_predictions.csv",
        "threshold": 0.38,
        "color": "#4C78A8",
    },
    "XGBoost": {
        "path": ROOT / "model" / "baseline" / "results" / "test_predictions_grid.csv",
        "threshold": 0.50,
        "color": "#F58518",
    },
    "FT-Transformer": {
        "path": ROOT / "model" / "ft_tf" / "results" / "test_predictions_ft_tf.csv",
        "threshold": 0.82,
        "color": "#54A24B",
    },
}


SEGMENT_SPECS = {
    "V1": {
        "label": "V1 raw",
        "scheme_name": "v1_raw_feature_unsupervised",
        "assignments": ROOT / "segment" / "results" / "segment_assignments.csv",
        "profiles": ROOT / "segment" / "results" / "segment_profiles.csv",
        "diagnostics": ROOT / "segment" / "results" / "segment_diagnostics.json",
        "color": "#4C78A8",
        "marker": "o",
    },
    "V2": {
        "label": "V2 informed",
        "scheme_name": "v2_score_augmented_classifier_informed",
        "assignments": ROOT / "segment" / "v2_from_ft_score" / "segment_assignments.csv",
        "profiles": ROOT / "segment" / "v2_from_ft_score" / "segment_profiles.csv",
        "diagnostics": ROOT / "segment" / "v2_from_ft_score" / "segment_diagnostics.json",
        "color": "#F58518",
        "marker": "^",
    },
}


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    total_weight = weights.sum()
    if total_weight <= 0:
        return float(np.nan)
    return float(np.dot(values, weights) / total_weight)


def weighted_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    weight = np.asarray(weight, dtype=float)

    tp = float(weight[(y_true == 1) & (y_pred == 1)].sum())
    fp = float(weight[(y_true == 0) & (y_pred == 1)].sum())
    fn = float(weight[(y_true == 1) & (y_pred == 0)].sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    predicted_positive_share = float(weight[y_pred == 1].sum() / weight.sum()) if weight.sum() > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive_share": predicted_positive_share,
    }


def weighted_brier_score(y_true: np.ndarray, y_prob: np.ndarray, weight: np.ndarray) -> float:
    return weighted_mean((np.asarray(y_true) - np.asarray(y_prob)) ** 2, np.asarray(weight))


def clean_feature_name(raw: str) -> str:
    return (
        raw.replace("num__", "")
        .replace("cat__", "")
        .replace("_", " ")
        .replace("detailed household summary in household", "household summary")
        .replace("full or part time employment stat", "employment stat")
    )


def load_predictions(path: Path) -> pd.DataFrame:
    usecols = ["true_label_binary", "probability", "weight"]
    frame = pd.read_csv(path, usecols=usecols)
    out = frame.rename(columns={"true_label_binary": "label"}).copy()
    out["label"] = out["label"].astype(int)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    out["probability"] = pd.to_numeric(out["probability"], errors="coerce").fillna(0.0)
    return out


def get_model_frame(model_name: str) -> pd.DataFrame:
    spec = MODEL_SPECS[model_name]
    df = load_predictions(spec["path"]).copy()
    df["model"] = model_name
    df["threshold"] = spec["threshold"]
    return df


def load_raw_with_row_id() -> pd.DataFrame:
    df = load_raw_data(ROOT / "census-bureau.data", ROOT / "census-bureau.columns").copy()
    df["_row_id"] = np.arange(len(df))
    df["label_binary"] = clean_label_to_binary(df["label"]).astype(int)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    return df


def load_segmentation_profiles(scheme_key: str) -> pd.DataFrame:
    df = pd.read_csv(SEGMENT_SPECS[scheme_key]["profiles"]).copy()
    df["scheme_key"] = scheme_key
    df["scheme_label"] = SEGMENT_SPECS[scheme_key]["label"]
    return df


def load_segmentation_assignments_merged(scheme_key: str) -> pd.DataFrame:
    raw = load_raw_with_row_id()
    assignments = pd.read_csv(SEGMENT_SPECS[scheme_key]["assignments"]).copy()
    merged = raw.merge(assignments, left_on="_row_id", right_on="row_id", how="inner", suffixes=("_raw", "_assign"))
    if "weight_raw" in merged.columns:
        merged["weight"] = pd.to_numeric(merged["weight_raw"], errors="coerce").fillna(0.0)
    elif "weight" in merged.columns:
        merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce").fillna(0.0)
    merged["scheme_key"] = scheme_key
    merged["scheme_label"] = SEGMENT_SPECS[scheme_key]["label"]
    return merged


def calibration_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    work = df.sort_values("probability").reset_index(drop=True)
    bins = np.array_split(np.arange(len(work)), n_bins)
    rows: list[dict[str, float | int]] = []
    for bin_idx, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            continue
        part = work.iloc[idx]
        rows.append(
            {
                "bin": bin_idx,
                "mean_pred": weighted_mean(part["probability"].to_numpy(), part["weight"].to_numpy()),
                "observed_rate": weighted_mean(part["label"].to_numpy(), part["weight"].to_numpy()),
                "weight_share": float(part["weight"].sum() / work["weight"].sum()),
            }
        )
    return pd.DataFrame(rows)


def calibration_stats(df: pd.DataFrame) -> dict[str, float]:
    table = calibration_table(df)
    ece = float((table["weight_share"] * (table["mean_pred"] - table["observed_rate"]).abs()).sum())
    brier = weighted_brier_score(
        df["label"].to_numpy(),
        df["probability"].to_numpy(),
        df["weight"].to_numpy(),
    )
    return {"ece": ece, "brier": brier}


def model_test_summary() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for model_name in MODEL_SPECS:
        df = get_model_frame(model_name)
        y_true = df["label"].to_numpy()
        y_score = df["probability"].to_numpy()
        weight = df["weight"].to_numpy()
        precision, recall, _ = precision_recall_curve(y_true, y_score, sample_weight=weight)
        op_metrics = weighted_binary_metrics(y_true, (y_score >= MODEL_SPECS[model_name]["threshold"]).astype(int), weight)
        rows.append(
            {
                "model": model_name,
                "roc_auc": roc_auc_score(y_true, y_score, sample_weight=weight),
                "pr_auc": auc(recall, precision),
                "operating_f1": op_metrics["f1"],
            }
        )
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def save_model_summary() -> None:
    summary = model_test_summary()
    summary.to_csv(FIG_DIR / "classification_test_summary.csv", index=False)


def save_segmentation_summary() -> None:
    rows = []
    for scheme_key, spec in SEGMENT_SPECS.items():
        data = json.loads(spec["diagnostics"].read_text())
        rows.append(
            {
                "scheme_key": scheme_key,
                "scheme_name": spec["scheme_name"],
                "n_clusters": data["n_clusters"],
                "best_k": data["best_k_result"]["k"],
                "best_score": data["best_k_result"]["score"],
                "silhouette": data["best_k_result"]["silhouette"],
                "stability_pairwise_agreement": data["stability_pairwise_agreement"],
            }
        )
    pd.DataFrame(rows).to_csv(FIG_DIR / "segmentation_summary.csv", index=False)


def get_lightgbm_valid_predictions(seed: int = 42) -> pd.DataFrame:
    raw = load_raw_with_row_id()
    splits = make_splits(raw, seed=seed)
    valid_df = splits["valid"].copy()
    artifact = joblib.load(ROOT / "model" / "baseline" / "inference" / "params" / "lightgbm_artifact.pkl")
    logits = predict_logits(artifact, valid_df)
    probs = sigmoid(logits)
    return pd.DataFrame(
        {
            "label": clean_label_to_binary(valid_df["label"]).astype(int),
            "probability": probs,
            "weight": pd.to_numeric(valid_df["weight"], errors="coerce").fillna(0.0),
        }
    )


def plot_eda_overview() -> None:
    df = load_raw_with_row_id().copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["weeks worked in year"] = pd.to_numeric(df["weeks worked in year"], errors="coerce")
    df["income_label"] = np.where(df["label_binary"] == 1, ">50K", "<=50K")

    neg_share = 1.0 - df["label_binary"].mean()
    pos_share = df["label_binary"].mean()
    weighted_pos = weighted_mean(df["label_binary"].to_numpy(), df["weight"].to_numpy())
    weighted_neg = 1.0 - weighted_pos

    fig, axes = plt.subplots(2, 2, figsize=(5.4, 4.1))

    prevalence = pd.DataFrame(
        {
            "setting": ["Unweighted", "Unweighted", "Weighted", "Weighted"],
            "class": ["<=50K", ">50K", "<=50K", ">50K"],
            "share": [neg_share, pos_share, weighted_neg, weighted_pos],
        }
    )
    pivot = prevalence.pivot(index="setting", columns="class", values="share")
    pivot[["<=50K", ">50K"]].plot(
        kind="bar",
        ax=axes[0, 0],
        color=["#4C78A8", "#E45756"],
        width=0.65,
    )
    axes[0, 0].set_title("Class Balance")
    axes[0, 0].set_ylabel("Population Share")
    axes[0, 0].tick_params(axis="x", rotation=0)
    axes[0, 0].legend(frameon=False, loc="upper right")

    axes[0, 1].hist(df["weight"], bins=25, color="#72B7B2", edgecolor="white")
    axes[0, 1].set_title("Survey Weight Distribution")
    axes[0, 1].set_xlabel("Weight")
    axes[0, 1].set_ylabel("Count")

    age_data = [df.loc[df["label_binary"] == 0, "age"].dropna(), df.loc[df["label_binary"] == 1, "age"].dropna()]
    age_box = axes[1, 0].boxplot(age_data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, color in zip(age_box["boxes"], ["#4C78A8", "#E45756"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1, 0].set_xticks([1, 2], ["<=50K", ">50K"])
    axes[1, 0].set_title("Age By Income Label")
    axes[1, 0].set_ylabel("Age")

    weeks_data = [
        df.loc[df["label_binary"] == 0, "weeks worked in year"].dropna(),
        df.loc[df["label_binary"] == 1, "weeks worked in year"].dropna(),
    ]
    weeks_box = axes[1, 1].boxplot(weeks_data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, color in zip(weeks_box["boxes"], ["#4C78A8", "#E45756"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1, 1].set_xticks([1, 2], ["<=50K", ">50K"])
    axes[1, 1].set_title("Weeks Worked By Income Label")
    axes[1, 1].set_ylabel("Weeks Worked")

    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.2)

    fig.tight_layout(pad=0.4, w_pad=0.35, h_pad=0.45)
    fig.savefig(FIG_DIR / "eda_overview.png")
    plt.close(fig)


def plot_roc_curves() -> None:
    fig, ax = plt.subplots()
    for model_name, spec in MODEL_SPECS.items():
        df = get_model_frame(model_name)
        fpr, tpr, _ = roc_curve(df["label"], df["probability"], sample_weight=df["weight"])
        score = roc_auc_score(df["label"], df["probability"], sample_weight=df["weight"])
        ax.plot(fpr, tpr, linewidth=2, color=spec["color"], label=f"{model_name} ({score:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Weighted ROC On Test")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(FIG_DIR / "classification_roc_test.png")
    plt.close(fig)


def plot_pr_curves() -> None:
    fig, ax = plt.subplots()
    for model_name, spec in MODEL_SPECS.items():
        df = get_model_frame(model_name)
        precision, recall, _ = precision_recall_curve(df["label"], df["probability"], sample_weight=df["weight"])
        score = auc(recall, precision)
        ax.plot(recall, precision, linewidth=2, color=spec["color"], label=f"{model_name} ({score:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Weighted PR On Test")
    ax.legend(loc="lower left", frameon=False)
    fig.savefig(FIG_DIR / "classification_pr_test.png")
    plt.close(fig)


def plot_threshold_sweep() -> None:
    df = get_lightgbm_valid_predictions()
    y_true = df["label"].to_numpy()
    y_prob = df["probability"].to_numpy()
    weight = df["weight"].to_numpy()

    rows = []
    for thr in np.linspace(0.05, 0.95, 181):
        metrics = weighted_binary_metrics(y_true, (y_prob >= thr).astype(int), weight)
        rows.append({"threshold": thr, **metrics})
    summary = pd.DataFrame(rows)
    summary.to_csv(FIG_DIR / "lightgbm_threshold_sweep_valid.csv", index=False)

    fig, ax = plt.subplots()
    ax.plot(summary["threshold"], summary["precision"], color="#4C78A8", linewidth=2, label="Precision")
    ax.plot(summary["threshold"], summary["recall"], color="#F58518", linewidth=2, label="Recall")
    ax.plot(summary["threshold"], summary["f1"], color="#54A24B", linewidth=2, label="Weighted F1")
    ax.axvline(0.38, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("LightGBM Threshold Policy")

    ax2 = ax.twinx()
    ax2.plot(
        summary["threshold"],
        summary["predicted_positive_share"],
        color="0.4",
        linewidth=1.5,
        linestyle=":",
        label="Targeted Share",
    )
    ax2.set_ylabel("Predicted Positive Share")

    lines = ax.get_lines()[:3] + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="center right", frameon=False)
    fig.savefig(FIG_DIR / "lightgbm_threshold_sweep_valid.png")
    plt.close(fig)


def plot_calibration_compare() -> None:
    fig, ax = plt.subplots()
    rows = []
    for model_name, spec in MODEL_SPECS.items():
        df = get_model_frame(model_name)
        table = calibration_table(df)
        stats = calibration_stats(df)
        table["model"] = model_name
        table["ece"] = stats["ece"]
        table["brier"] = stats["brier"]
        rows.append(table)
        ax.plot(
            table["mean_pred"],
            table["observed_rate"],
            marker="o",
            linewidth=2,
            color=spec["color"],
            label=f"{model_name} ECE={stats['ece']:.3f}",
        )

    pd.concat(rows, ignore_index=True).to_csv(FIG_DIR / "classification_calibration_summary.csv", index=False)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Positive Rate")
    ax.set_title("Calibration On Test")
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(FIG_DIR / "classification_calibration_compare.png")
    plt.close(fig)


def plot_lightgbm_score_distribution() -> None:
    df = get_model_frame("LightGBM")
    fig, ax = plt.subplots()
    bins = np.linspace(0.0, 1.0, 21)
    for label_value, color, label_text in [(0, "#4C78A8", "True 0"), (1, "#E45756", "True 1")]:
        part = df.loc[df["label"] == label_value]
        weights = part["weight"].to_numpy()
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        ax.hist(
            part["probability"],
            bins=bins,
            weights=weights,
            alpha=0.45,
            color=color,
            label=label_text,
        )

    ax.axvline(0.38, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Weighted Density")
    ax.set_title("LightGBM Score Separation")
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "lightgbm_score_distribution.png")
    plt.close(fig)


def bootstrap_metric_intervals(n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for model_name in MODEL_SPECS:
        df = get_model_frame(model_name)
        y_true = df["label"].to_numpy()
        y_prob = df["probability"].to_numpy()
        weight = df["weight"].to_numpy()
        threshold = MODEL_SPECS[model_name]["threshold"]

        samples = {"roc_auc": [], "pr_auc": [], "operating_f1": []}
        n = len(df)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            y_b = y_true[idx]
            p_b = y_prob[idx]
            w_b = weight[idx]
            try:
                roc_value = roc_auc_score(y_b, p_b, sample_weight=w_b)
            except ValueError:
                continue
            precision, recall, _ = precision_recall_curve(y_b, p_b, sample_weight=w_b)
            pr_value = auc(recall, precision)
            f1_value = weighted_binary_metrics(y_b, (p_b >= threshold).astype(int), w_b)["f1"]
            samples["roc_auc"].append(roc_value)
            samples["pr_auc"].append(pr_value)
            samples["operating_f1"].append(f1_value)

        for metric_name, values in samples.items():
            point_value = (
                roc_auc_score(y_true, y_prob, sample_weight=weight)
                if metric_name == "roc_auc"
                else auc(*precision_recall_curve(y_true, y_prob, sample_weight=weight)[1::-1])
                if metric_name == "pr_auc"
                else weighted_binary_metrics(y_true, (y_prob >= threshold).astype(int), weight)["f1"]
            )
            rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "point_estimate": point_value,
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(FIG_DIR / "classification_bootstrap_ci.csv", index=False)
    return out


def plot_model_metric_comparison() -> None:
    summary = bootstrap_metric_intervals()
    metric_order = ["roc_auc", "pr_auc", "operating_f1"]
    metric_labels = ["ROC-AUC", "PR-AUC", "F1@Op"]
    fig, ax = plt.subplots()

    for model_name, spec in MODEL_SPECS.items():
        part = summary.loc[summary["model"] == model_name].set_index("metric").loc[metric_order]
        x = np.arange(len(metric_order))
        y = part["point_estimate"].to_numpy()
        yerr = np.vstack([y - part["ci_low"].to_numpy(), part["ci_high"].to_numpy() - y])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            color=spec["color"],
            label=model_name,
        )

    ax.set_xticks(np.arange(len(metric_order)), metric_labels)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("Metric Value")
    ax.set_title("Three-Model Performance")
    ax.legend(loc="lower left", frameon=False)
    fig.savefig(FIG_DIR / "classification_metric_comparison.png")
    plt.close(fig)


def plot_lightgbm_confusion_matrix() -> None:
    df = get_model_frame("LightGBM")
    tuned_threshold = MODEL_SPECS["LightGBM"]["threshold"]
    y_pred = (df["probability"] >= tuned_threshold).astype(int)
    cm = confusion_matrix(df["label"], y_pred, sample_weight=df["weight"], labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_rate = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots()
    image = ax.imshow(cm_rate, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    labels = np.array([["TN", "FP"], ["FN", "TP"]])

    for i in range(2):
        for j in range(2):
            weighted_count = cm[i, j] / 1_000_000
            pct = cm_rate[i, j] * 100
            ax.text(j, i, f"{labels[i, j]}\n{pct:.1f}%\n{weighted_count:.1f}M", ha="center", va="center", color="black")

    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title("LightGBM At Threshold 0.38")
    fig.savefig(FIG_DIR / "lightgbm_confusion_matrix_test.png")
    plt.close(fig)


def plot_lightgbm_feature_importance() -> None:
    path = ROOT / "model" / "baseline" / "results" / "lightgbm_feature_importance.csv"
    df = pd.read_csv(path).head(10).copy()
    df["feature"] = df["feature"].map(clean_feature_name)
    df = df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots()
    ax.barh(df["feature"], df["importance"], color=MODEL_SPECS["LightGBM"]["color"])
    ax.set_xlabel("Split Importance")
    ax.set_title("Top LightGBM Features")
    fig.savefig(FIG_DIR / "lightgbm_feature_importance.png")
    plt.close(fig)


def plot_segmentation_k_search() -> None:
    fig, ax = plt.subplots()
    for _, spec in SEGMENT_SPECS.items():
        data = json.loads(spec["diagnostics"].read_text())
        grid = pd.DataFrame(data["k_grid_results"])
        ax.plot(grid["k"], grid["score"], marker="o", linewidth=2, color=spec["color"], label=spec["label"])
        best_k = data["best_k_result"]["k"]
        best_score = data["best_k_result"]["score"]
        ax.scatter([best_k], [best_score], color=spec["color"], s=60, zorder=3)

    ax.set_xlabel("Number Of Clusters (k)")
    ax.set_ylabel("Composite Search Score")
    ax.set_title("Segmentation K Search")
    ax.legend(frameon=False)
    fig.savefig(FIG_DIR / "segmentation_k_search.png")
    plt.close(fig)


def plot_segmentation_stability_compare() -> None:
    rows = []
    for _, spec in SEGMENT_SPECS.items():
        data = json.loads(spec["diagnostics"].read_text())
        rows.append({"scheme": spec["label"], "stability": data["stability_pairwise_agreement"], "color": spec["color"]})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots()
    ax.bar(df["scheme"], df["stability"], color=df["color"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Pairwise Agreement")
    ax.set_title("Segmentation Stability")
    fig.savefig(FIG_DIR / "segmentation_stability_compare.png")
    plt.close(fig)


def plot_segmentation_positioning_compare() -> None:
    fig, ax = plt.subplots()
    for scheme_key, spec in SEGMENT_SPECS.items():
        df = load_segmentation_profiles(scheme_key)
        df["share_pct"] = df["segment_share_weighted"] * 100.0
        df["income_pct"] = df["income_rate_weighted"] * 100.0
        bubble_size = 220 + df["segment_share_weighted"] * 2400
        ax.scatter(
            df["share_pct"],
            df["income_pct"],
            s=bubble_size,
            alpha=0.72,
            color=spec["color"],
            marker=spec["marker"],
            label=spec["label"],
        )
        for _, row in df.iterrows():
            ax.annotate(
                f"{scheme_key}-S{int(row['segment_id'])}",
                (row["share_pct"], row["income_pct"]),
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlabel("Weighted Population Share (%)")
    ax.set_ylabel("Income Rate >50K (%)")
    ax.set_title("V1 Vs V2 Positioning")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(FIG_DIR / "segmentation_positioning_compare.png")
    plt.close(fig)


def plot_segmentation_profile_heatmap() -> None:
    cols = [
        "income_rate_delta_vs_population",
        "weeks_worked_delta_vs_population",
        "college_or_higher_delta_vs_population",
        "married_delta_vs_population",
        "capital_gains_positive_rate",
        "dividends_positive_rate",
    ]
    col_labels = ["Income d", "Weeks d", "College d", "Married d", "Cap gains", "Dividends"]

    frames = [load_segmentation_profiles("V1"), load_segmentation_profiles("V2")]
    df = pd.concat(frames, ignore_index=True)
    z = df[cols].copy()
    z = (z - z.mean()) / z.std(ddof=0).replace(0, 1)

    fig, ax = plt.subplots()
    image = ax.imshow(z.to_numpy(), cmap="coolwarm", aspect="auto", vmin=-2.0, vmax=2.0)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=35, ha="right")
    row_labels = [f"{scheme}-S{int(seg)}" for scheme, seg in zip(df["scheme_key"], df["segment_id"])]
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title("Segment Profile Heatmap")
    fig.savefig(FIG_DIR / "segmentation_profile_heatmap.png")
    plt.close(fig)

    out = df[["scheme_key", "segment_id"] + cols].copy()
    out.to_csv(FIG_DIR / "segmentation_profile_heatmap_source.csv", index=False)


def plot_segmentation_numeric_boxplots() -> None:
    merged = pd.concat(
        [load_segmentation_assignments_merged("V1"), load_segmentation_assignments_merged("V2")],
        ignore_index=True,
    )
    variables = [("age", "Age"), ("weeks worked in year", "Weeks Worked")]

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)
    order = []
    labels = []
    colors = []
    box_data: dict[str, list[np.ndarray]] = {var: [] for var, _ in variables}

    for scheme_key in ["V1", "V2"]:
        part = merged.loc[merged["scheme_key"] == scheme_key].copy()
        for segment_id in sorted(part["segment_id"].unique()):
            seg_part = part.loc[part["segment_id"] == segment_id]
            order.append((scheme_key, segment_id))
            labels.append(f"{scheme_key}-S{int(segment_id)}")
            colors.append(SEGMENT_SPECS[scheme_key]["color"])
            for var, _ in variables:
                box_data[var].append(pd.to_numeric(seg_part[var], errors="coerce").dropna().to_numpy())

    positions = np.arange(1, len(order) + 1)
    for ax, (var, title) in zip(axes, variables):
        bp = ax.boxplot(box_data[var], positions=positions, patch_artist=True, widths=0.6, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for median in bp["medians"]:
            median.set_color("black")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(positions, labels, rotation=90)
    axes[0].set_title("Numeric Profiles By Segment")
    legend_handles = [Patch(facecolor=SEGMENT_SPECS[k]["color"], alpha=0.55, label=SEGMENT_SPECS[k]["label"]) for k in ["V1", "V2"]]
    axes[0].legend(handles=legend_handles, frameon=False, loc="upper right")
    fig.savefig(FIG_DIR / "segmentation_numeric_boxplots.png")
    plt.close(fig)


def plot_segmentation_education_composition() -> None:
    merged = pd.concat(
        [load_segmentation_assignments_merged("V1"), load_segmentation_assignments_merged("V2")],
        ignore_index=True,
    )
    top_categories = (
        merged.groupby("education", as_index=False)["weight"].sum().sort_values("weight", ascending=False).head(4)["education"].tolist()
    )
    merged["education_plot"] = np.where(merged["education"].isin(top_categories), merged["education"], "Other")

    rows = []
    labels = []
    for scheme_key in ["V1", "V2"]:
        part = merged.loc[merged["scheme_key"] == scheme_key]
        for segment_id in sorted(part["segment_id"].unique()):
            seg_part = part.loc[part["segment_id"] == segment_id]
            labels.append(f"{scheme_key}-S{int(segment_id)}")
            total_weight = seg_part["weight"].sum()
            row = {"label": f"{scheme_key}-S{int(segment_id)}"}
            for cat in top_categories + ["Other"]:
                cat_weight = seg_part.loc[seg_part["education_plot"] == cat, "weight"].sum()
                row[cat] = cat_weight / total_weight if total_weight > 0 else 0.0
            rows.append(row)

    df = pd.DataFrame(rows)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    fig, ax = plt.subplots()
    bottom = np.zeros(len(df))
    for cat, color in zip(top_categories + ["Other"], colors):
        values = df[cat].to_numpy()
        ax.bar(df["label"], values, bottom=bottom, color=color, label=cat)
        bottom += values

    ax.set_ylabel("Weighted Share")
    ax.set_title("Education Mix By Segment")
    ax.set_xticks(np.arange(len(df)), df["label"], rotation=90)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(FIG_DIR / "segmentation_education_composition.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_eda_overview()
    save_model_summary()
    save_segmentation_summary()
    plot_roc_curves()
    plot_pr_curves()
    plot_threshold_sweep()
    plot_calibration_compare()
    plot_lightgbm_score_distribution()
    plot_model_metric_comparison()
    plot_lightgbm_confusion_matrix()
    plot_lightgbm_feature_importance()
    plot_segmentation_k_search()
    plot_segmentation_stability_compare()
    plot_segmentation_positioning_compare()
    plot_segmentation_profile_heatmap()
    plot_segmentation_numeric_boxplots()
    plot_segmentation_education_composition()
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
