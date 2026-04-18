from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model.baseline.dataio.dataset import WEIGHT_COL, clean_label_to_binary


@dataclass
class SegmenterArtifact:
    scheme_name: str
    feature_columns: list[str]
    preprocessor: ColumnTransformer
    model: MiniBatchKMeans
    n_clusters: int
    diagnostics: dict[str, Any]
    random_state: int


def _safe_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _age_band(series: pd.Series) -> pd.Series:
    age = pd.to_numeric(series, errors="coerce")
    bins = [-1, 24, 34, 44, 54, 64, 120]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    return pd.cut(age, bins=bins, labels=labels).astype(str)


def _weeks_band(series: pd.Series) -> pd.Series:
    weeks = pd.to_numeric(series, errors="coerce")
    bins = [-1, 0, 20, 40, 52, 60]
    labels = ["0", "1-20", "21-40", "41-52", "53+"]
    return pd.cut(weeks, bins=bins, labels=labels).astype(str)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "_row_id" not in work.columns:
        work["_row_id"] = np.arange(len(work))
    return work


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", _safe_ohe()),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def _pairwise_agreement(a: np.ndarray, b: np.ndarray, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.choice(n, size=min(1500, n), replace=False)
    a_s = a[idx]
    b_s = b[idx]
    agree: list[int] = []
    for i in range(len(idx) - 1):
        same_a = a_s[i + 1 :] == a_s[i]
        same_b = b_s[i + 1 :] == b_s[i]
        agree.extend((same_a == same_b).astype(int).tolist())
    return float(np.mean(agree)) if agree else 1.0


def _search_best_k(X_mat, random_state: int = 42) -> tuple[int, dict[str, Any]]:
    n_rows = int(X_mat.shape[0])
    if n_rows <= 3:
        return 2, {
            "k_grid_results": [],
            "best_k_result": {"k": 2, "score": 0.0, "silhouette": 0.0, "davies_bouldin": 0.0, "calinski_harabasz": 0.0},
        }

    n_components = max(2, min(20, X_mat.shape[1] - 1 if X_mat.shape[1] > 1 else 2))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_red = svd.fit_transform(X_mat)
    if X_red.shape[0] > 15000:
        sample_idx = np.random.default_rng(random_state).choice(X_red.shape[0], size=15000, replace=False)
        X_eval = X_red[sample_idx]
    else:
        X_eval = X_red

    k_min = 3 if n_rows >= 4 else 2
    k_max = min(8, n_rows - 1)
    if k_min > k_max:
        k_min = max(2, k_max)
    candidates: list[dict[str, Any]] = []
    for k in range(k_min, k_max + 1):
        model_full = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=20, batch_size=4096)
        labels = model_full.fit_predict(X_mat)
        model_eval = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=20, batch_size=4096)
        labels_eval = model_eval.fit_predict(X_eval)
        shares = pd.Series(labels).value_counts(normalize=True)
        min_share = float(shares.min())
        sil = float(silhouette_score(X_eval, labels_eval))
        db = float(davies_bouldin_score(X_eval, labels_eval))
        ch = float(calinski_harabasz_score(X_eval, labels_eval))
        score = sil + (1.0 / (1.0 + db)) + (np.log1p(ch) / 10.0) - (0.5 if min_share < 0.01 else 0.0)
        candidates.append(
            {
                "k": k,
                "score": float(score),
                "silhouette": sil,
                "davies_bouldin": db,
                "calinski_harabasz": ch,
                "min_cluster_share_row": min_share,
            }
        )

    best = sorted(candidates, key=lambda x: x["score"], reverse=True)[0]
    return int(best["k"]), {"k_grid_results": candidates, "best_k_result": best}


def fit_segmenter_from_features(
    train_df: pd.DataFrame,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
    scheme_name: str,
    random_state: int = 42,
    fixed_n_clusters: int | None = None,
) -> SegmenterArtifact:
    work = _prepare(train_df)
    X = feature_builder(work)
    preprocessor = _build_preprocessor(X)
    X_mat = preprocessor.fit_transform(X)

    if fixed_n_clusters is not None:
        best_k = int(fixed_n_clusters)
        search_diag = {"k_grid_results": [], "best_k_result": {"k": best_k, "selection": "fixed"}}
    else:
        best_k, search_diag = _search_best_k(X_mat, random_state=random_state)
    model = MiniBatchKMeans(n_clusters=best_k, random_state=random_state, n_init=30, batch_size=4096)
    labels = model.fit_predict(X_mat)

    model_alt = MiniBatchKMeans(n_clusters=best_k, random_state=random_state + 11, n_init=30, batch_size=4096)
    labels_alt = model_alt.fit_predict(X_mat)
    stability = _pairwise_agreement(np.asarray(labels), np.asarray(labels_alt), seed=42)

    diagnostics = {
        "scheme_name": scheme_name,
        "n_clusters": best_k,
        "cluster_size_distribution": pd.Series(labels).value_counts().sort_index().to_dict(),
        "stability_pairwise_agreement": stability,
        **search_diag,
    }

    return SegmenterArtifact(
        scheme_name=scheme_name,
        feature_columns=list(X.columns),
        preprocessor=preprocessor,
        model=model,
        n_clusters=best_k,
        diagnostics=diagnostics,
        random_state=random_state,
    )


def assign_segments_from_features(
    model: SegmenterArtifact,
    df: pd.DataFrame,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
) -> np.ndarray:
    work = _prepare(df)
    X = feature_builder(work)
    X = X.reindex(columns=model.feature_columns)
    X_mat = model.preprocessor.transform(X)
    return model.model.predict(X_mat).astype(int)


def _weighted_rate(y: pd.Series, w: pd.Series) -> float:
    y_num = pd.to_numeric(y, errors="coerce").fillna(0).astype(float)
    w_num = pd.to_numeric(w, errors="coerce").fillna(0).astype(float)
    denom = w_num.sum()
    return float((y_num * w_num).sum() / denom) if denom > 0 else 0.0


def _top_value(series: pd.Series, weights: pd.Series) -> str:
    tmp = pd.DataFrame({"v": series.astype(str), "w": pd.to_numeric(weights, errors="coerce").fillna(0.0)})
    agg = tmp.groupby("v", observed=True)["w"].sum().sort_values(ascending=False)
    return str(agg.index[0]) if len(agg) else "Unknown"


def build_segment_outputs(
    df: pd.DataFrame,
    assignments: pd.DataFrame,
    scheme_name: str,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    merged = df.copy()
    merged = merged.merge(assignments[["row_id", "segment_id", "split"]], left_on="_row_id", right_on="row_id", how="left")
    merged["income_label"] = clean_label_to_binary(merged["label"])
    merged["weight_num"] = pd.to_numeric(merged[WEIGHT_COL], errors="coerce").fillna(0.0)

    total_weight = merged["weight_num"].sum()
    pop_income = _weighted_rate(merged["income_label"], merged["weight_num"])
    pop_weeks = float(np.average(pd.to_numeric(merged["weeks worked in year"], errors="coerce").fillna(0), weights=merged["weight_num"]))
    pop_college = _weighted_rate(
        merged["education"].astype(str).isin(
            [
                "Bachelors degree(BA AB BS)",
                "Masters degree(MA MS MEng MEd MSW MBA)",
                "Doctorate degree(PhD EdD)",
                "Prof school degree (MD DDS DVM LLB JD)",
            ]
        ).astype(int),
        merged["weight_num"],
    )
    pop_married = _weighted_rate(merged["marital stat"].astype(str).str.contains("Married", na=False).astype(int), merged["weight_num"])

    rows: list[dict[str, Any]] = []
    weighted_income_by_segment: dict[str, float] = {}
    for seg_id, g in merged.groupby("segment_id", observed=True):
        w = g["weight_num"]
        seg_weight = float(w.sum())
        seg_income = _weighted_rate(g["income_label"], w)
        weeks_mean = float(np.average(pd.to_numeric(g["weeks worked in year"], errors="coerce").fillna(0), weights=w))
        college_rate = _weighted_rate(
            g["education"].astype(str).isin(
                [
                    "Bachelors degree(BA AB BS)",
                    "Masters degree(MA MS MEng MEd MSW MBA)",
                    "Doctorate degree(PhD EdD)",
                    "Prof school degree (MD DDS DVM LLB JD)",
                ]
            ).astype(int),
            w,
        )
        married_rate = _weighted_rate(g["marital stat"].astype(str).str.contains("Married", na=False).astype(int), w)
        row = {
            "segment_id": int(seg_id),
            "segment_name": f"Segment {int(seg_id)}",
            "segment_size_n": int(len(g)),
            "segment_weight_sum": seg_weight,
            "segment_share_weighted": float(seg_weight / total_weight) if total_weight > 0 else 0.0,
            "income_rate_weighted": seg_income,
            "income_rate_delta_vs_population": float(seg_income - pop_income),
            "top_age_band": _top_value(_age_band(g["age"]), w),
            "top_education": _top_value(g["education"], w),
            "top_marital_status": _top_value(g["marital stat"], w),
            "top_work_class": _top_value(g["class of worker"], w),
            "top_occupation": _top_value(g["major occupation code"], w),
            "weeks_worked_mean": weeks_mean,
            "weeks_worked_delta_vs_population": float(weeks_mean - pop_weeks),
            "college_or_higher_delta_vs_population": float(college_rate - pop_college),
            "married_delta_vs_population": float(married_rate - pop_married),
            "capital_gains_positive_rate": _weighted_rate((pd.to_numeric(g["capital gains"], errors="coerce").fillna(0) > 0).astype(int), w),
            "dividends_positive_rate": _weighted_rate((pd.to_numeric(g["dividends from stocks"], errors="coerce").fillna(0) > 0).astype(int), w),
            "scheme_name": scheme_name,
        }
        rows.append(row)
        weighted_income_by_segment[str(int(seg_id))] = seg_income

    profiles = pd.DataFrame(rows).sort_values("segment_id").reset_index(drop=True)

    action_lines = [
        f"# Segment Marketing Actions ({scheme_name})",
        "",
        "Below are business actions per segment. Each recommendation combines demographic profile, labor intensity, and asset signals.",
        "",
    ]
    for _, r in profiles.iterrows():
        action_lines.extend(
            [
                f"## Segment {int(r['segment_id'])} - {r['segment_name']}",
                f"- Who: Top education `{r['top_education']}`, marital status `{r['top_marital_status']}`, work class `{r['top_work_class']}`.",
                f"- Differentiator: income delta `{r['income_rate_delta_vs_population']:.3f}`, weeks-worked delta `{r['weeks_worked_delta_vs_population']:.2f}`.",
                "- Action: use tailored product tiering, message personalization, and channel optimization based on spending propensity.",
                "- Caution: monitor response drift across years and avoid overfitting to sparse categories.",
                "",
            ]
        )
    actions_md = "\n".join(action_lines)

    diagnostics_extra = {
        "scheme_name": scheme_name,
        "weighted_income_rate_by_segment": weighted_income_by_segment,
    }
    return profiles, diagnostics_extra, actions_md
