"""Vector-DB k-NN attribute prediction using DINOv3 features.

For each test image we retrieve the top-k most similar **train** images
(by cosine similarity of the frozen DINOv3 embeddings) and, per attribute,
take the majority-vote / mean of the neighbours' labels.

Key design points:
- Features are L2-normalised at extraction time, so cosine similarity is
  a single ``A @ B.T`` matmul.
- Retrieval is restricted to neighbours of the **same asset type**.
- Returns both per-fold metrics and summary aggregates.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from src.baseline import infer_target_column
from src.dinov3_features import feature_columns
from src.mlflow_utils import setup_mlflow, make_run_name, make_standard_tags

logger = logging.getLogger(__name__)


DEFAULT_K = 10

def _get_sorted_top_k(
    similarities: np.ndarray, k: int, num_train: int
) -> tuple[np.ndarray, np.ndarray]:
    """Find and sort top-k indices by similarity.

    Uses argpartition for efficiency, then sorts results in descending order.
    """
    k_eff = min(k, num_train)
    top_idx = np.argpartition(-similarities, kth=k_eff - 1, axis=1)[:, :k_eff]
    rows = np.arange(similarities.shape[0])[:, None]
    top_sims = similarities[rows, top_idx]
    order = np.argsort(-top_sims, axis=1)
    top_idx = top_idx[rows, order]
    top_sims = top_sims[rows, order]
    return top_idx, top_sims


@dataclass
class KNNIndex:
    """Cosine k-NN index over a feature matrix."""

    feature_matrix: np.ndarray
    asset_ids: np.ndarray

    def query(self, query_features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(top_k_indices, top_k_similarities)`` shapes (N_q, k)."""
        if self.feature_matrix.shape[0] == 0 or query_features.shape[0] == 0:
            return (
                np.zeros((query_features.shape[0], 0), dtype=int),
                np.zeros((query_features.shape[0], 0), dtype=float),
            )
        sims = query_features @ self.feature_matrix.T
        top_idx, top_sims = _get_sorted_top_k(sims, k, self.feature_matrix.shape[0])
        return top_idx, top_sims


def _build_index(train_subset: pd.DataFrame, feature_cols: list[str]) -> KNNIndex:
    """Build k-NN index from a train DataFrame with feature columns."""
    features = train_subset[feature_cols].to_numpy(dtype=np.float32)
    return KNNIndex(
        feature_matrix=features,
        asset_ids=train_subset["asset_id"].to_numpy(),
    )


def _predict_categorical_label(neighbor_labels: pd.Series) -> object:
    """Predict via majority vote among neighbors."""
    votes = neighbor_labels.dropna()
    if votes.empty:
        return np.nan
    return votes.value_counts().index[0]


def _predict_numeric_value(neighbor_values: pd.Series) -> float:
    """Predict via mean of numeric neighbors."""
    values = neighbor_values.dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def cross_validate_knn(
    labels: pd.DataFrame,
    asset_features: pd.DataFrame,
    target: str,
    *,
    target_file: str | None = None,
    feature_file: str | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    group_column: str = "asset_id",
    knn_k: int = DEFAULT_K,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate k-NN with grouped cross-validation on asset embeddings."""
    target_column = infer_target_column(labels, target)
    feats = feature_columns(asset_features.columns)
    if not feats:
        return pd.DataFrame(), pd.DataFrame()

    labelled = labels[["asset_id", target_column]].dropna(subset=[target_column])
    labelled = labelled.drop_duplicates(["asset_id", target_column])
    joined = labelled.merge(asset_features[["asset_id", *feats]], on="asset_id", how="inner")
    joined = joined.drop_duplicates("asset_id").reset_index(drop=True)

    if len(joined) < 2:
        return pd.DataFrame(), pd.DataFrame()

    # Detect task type
    is_bin = target_column.endswith("_bin")
    is_numeric = pd.api.types.is_numeric_dtype(joined[target_column])
    is_regression = is_numeric and not is_bin

    n_asset_groups = joined[group_column].nunique()
    target_splits = min(n_splits, n_asset_groups)
    if target_splits < 2:
        return pd.DataFrame(), pd.DataFrame()

    # StratifiedGroupKFold only works for classification
    if not is_regression:
        class_group_counts = joined.groupby(target_column)[group_column].nunique()
        if int(class_group_counts.min()) >= target_splits:
            splitter = StratifiedGroupKFold(
                n_splits=target_splits, shuffle=True, random_state=random_state
            )
            splitter_name = "StratifiedGroupKFold"
        else:
            splitter = GroupKFold(n_splits=target_splits, shuffle=True, random_state=random_state)
            splitter_name = "GroupKFold"
    else:
        splitter = GroupKFold(n_splits=target_splits, shuffle=True, random_state=random_state)
        splitter_name = "GroupKFold"

    y = joined[target_column]
    groups = joined[group_column]
    fold_rows: list[dict[str, object]] = []

    for fold_num, (train_idx, valid_idx) in enumerate(splitter.split(joined, y, groups), start=1):
        train_subset = joined.iloc[train_idx]
        valid_subset = joined.iloc[valid_idx]

        train_assets = set(train_subset[group_column])
        valid_assets = set(valid_subset[group_column])
        overlap = train_assets & valid_assets
        if overlap:
            raise AssertionError(f"Asset leakage in fold {fold_num}: {sorted(overlap)[:5]}")

        index = _build_index(train_subset, feats)
        if index.feature_matrix.shape[0] == 0:
            continue

        query_feats = valid_subset[feats].to_numpy(dtype=np.float32)
        top_idx, _ = index.query(query_feats, k=knn_k)

        # Predict: majority vote (classification) or mean (regression)
        preds = []
        for idx_row in top_idx:
            if len(idx_row) == 0:
                preds.append(np.nan)
                continue
            neighbor_labels = train_subset.iloc[idx_row][target_column]
            if is_regression:
                preds.append(_predict_numeric_value(neighbor_labels))
            else:
                preds.append(_predict_categorical_label(neighbor_labels))

        y_pred = pd.Series(preds)
        y_valid = y.iloc[valid_idx].reset_index(drop=True)

        if len(y_pred.dropna()) == 0:
            continue

        # Metrics 
        if is_regression:
            y_pred_num = pd.to_numeric(y_pred, errors="coerce")
            mask = y_pred_num.notna() & y_valid.notna()
            if mask.sum() < 2:
                continue
            metrics = {
                "task_type": "regression",
                "mae":  float(mean_absolute_error(y_valid[mask], y_pred_num[mask])),
                "rmse": float(np.sqrt(mean_squared_error(y_valid[mask], y_pred_num[mask]))),
                "r2":   float(r2_score(y_valid[mask], y_pred_num[mask])),
            }
        else:
            metrics = {
                "task_type": "classification",
                "accuracy":    float(accuracy_score(y_valid, y_pred)),
                "weighted_f1": float(f1_score(y_valid, y_pred, average="weighted", zero_division=0)),
                "macro_f1":    float(f1_score(y_valid, y_pred, average="macro", zero_division=0)),
            }

        fold_rows.append(
            {
                "target_column": target_column,
                "target_file": target_file,
                "feature_file": feature_file,
                "fold": fold_num,
                "n_folds": target_splits,
                "splitter": splitter_name,
                "knn_k": knn_k,
                "n_train_assets": len(train_assets),
                "n_valid_assets": len(valid_assets),
                **metrics,
            }
        )

    if not fold_rows:
        return pd.DataFrame(), pd.DataFrame()

    fold_df = pd.DataFrame(fold_rows)

    # Summarize: aggregate differently per task type
    task_type = fold_df["task_type"].iloc[0]
    group_cols = ["target_column", "target_file", "feature_file", "splitter", "knn_k", "task_type"]

    if task_type == "regression":
        summary = fold_df.groupby(group_cols, dropna=False).agg(
            n_folds=("fold", "max"),
            n_valid_assets=("n_valid_assets", "sum"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        ).reset_index()
    else:
        summary = fold_df.groupby(group_cols, dropna=False).agg(
            n_folds=("fold", "max"),
            n_valid_assets=("n_valid_assets", "sum"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_std=("weighted_f1", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        ).reset_index()
    # ─────────────────────────────────────────────────────────────────────────

    return summary, fold_df
