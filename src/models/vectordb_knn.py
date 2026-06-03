"""Vector-DB k-NN attribute prediction over the DINOv3 feature cache.

For each test image we retrieve the top-k most similar **train** images
(by cosine similarity of the cached DINOv3 embeddings) and, per
attribute, take the majority-vote / mean of the neighbours' labels.
The mean cosine similarity of the retrieved neighbours is logged as a
confidence proxy.

Key design points:
- Features are already L2-normalised at extraction time, so cosine
  similarity is a single ``A @ B.T`` matmul — no Qdrant install needed.
- Retrieval is restricted to neighbours of the **same asset type** as
  the query image (no point voting with stair images for a bridge's
  bridge_type).
- The same retrieval index can be reused as a "find similar BC Parks
  asset" tool — we expose ``retrieve_similar_assets`` for that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from src.data.schema import AttributeKind, Schema, load_schema
from src.embed.dinov3 import FeatureCache
from src.models.base import predict_per_asset_type

logger = logging.getLogger(__name__)

DEFAULT_K = 10
from src.utils.labels import MISSING_TOKENS as _MISSING_TOKENS

BIN_TARGET_TO_FEATURE_STEM = {
    "fall_height_bin": "dinov3_attr_fall_height_train_images",
    "steps_bin": "dinov3_attr_number_of_steps_train_images",
    "length_bin": "dinov3_attr_length_train_images",
    "width_bin": "dinov3_attr_width_train_images",
}


def _get_sorted_top_k(
    similarities: np.ndarray, k: int, num_train: int
) -> tuple[np.ndarray, np.ndarray]:
    """Find and sort top-k indices by similarity.

    Uses argpartition for efficiency, then sorts results in descending order.

    Args:
        similarities: (n_queries, n_train) similarity matrix from query_features @ train_features.T
        k: number of neighbors to retrieve
        num_train: total number of training samples (for bounds checking)

    Returns:
        (top_k_indices, top_k_similarities): both shape (n_queries, k_eff) where k_eff = min(k, num_train)
    """
    k_eff = min(k, num_train)
    # Get top-k indices using argpartition (faster than full sort)
    top_idx = np.argpartition(-similarities, kth=k_eff - 1, axis=1)[:, :k_eff]

    # Gather similarities for selected indices
    rows = np.arange(similarities.shape[0])[:, None]
    top_sims = similarities[rows, top_idx]

    # Sort by similarity (descending)
    order = np.argsort(-top_sims, axis=1)
    top_idx = top_idx[rows, order]
    top_sims = top_sims[rows, order]

    return top_idx, top_sims


@dataclass
class KNNIndex:
    """Cosine k-NN index over a feature matrix, scoped to a single asset type."""

    feature_matrix: np.ndarray
    image_paths: np.ndarray

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


def _extract_aligned_features(
    paths: list[str], cache: FeatureCache
) -> tuple[np.ndarray, np.ndarray]:
    """Get embeddings and keep mask from cache. Returns (features, keep_mask)."""
    feats, missing = cache.aligned_to(paths)
    keep = ~missing
    return feats, keep


def _build_index(train_subset: pd.DataFrame, cache: FeatureCache) -> KNNIndex:
    """Build k-NN index for a train subset with available embeddings."""
    feats, keep = _extract_aligned_features(train_subset["image_path"].tolist(), cache)
    return KNNIndex(
        feature_matrix=feats[keep],
        image_paths=train_subset.loc[train_subset.index[keep], "image_path"].to_numpy(),
    )


def _query_features(
    test_subset: pd.DataFrame, cache: FeatureCache
) -> tuple[np.ndarray, pd.Index]:
    """Extract test features and return valid indices."""
    feats, keep = _extract_aligned_features(test_subset["image_path"].tolist(), cache)
    return feats[keep], test_subset.index[keep]


from src.utils.labels import clean_labels as _clean_labels  # re-export


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


def _knn_one_attribute(
    asset_type_id: str,
    attribute_column: str,
    train_subset: pd.DataFrame,
    test_subset: pd.DataFrame,
    *,
    schema: Schema,
    cache: FeatureCache,
    k: int,
) -> pd.Series | None:
    """Predict one attribute for test samples using k-NN on train samples."""
    if attribute_column not in train_subset.columns:
        return None

    try:
        kind = schema.kind_of(attribute_column)
    except KeyError:
        kind = AttributeKind.ORDINAL_BIN
    index = _build_index(train_subset, cache)
    if index.feature_matrix.shape[0] == 0:
        return None

    query_feats, test_index = _query_features(test_subset, cache)
    if query_feats.shape[0] == 0:
        return None

    top_idx, _ = index.query(query_feats, k=k)

    # Categorical: majority vote among k neighbors
    if kind in {AttributeKind.CATEGORICAL, AttributeKind.BOOLEAN, AttributeKind.ORDINAL_BIN}:
        cleaned = _clean_labels(train_subset[attribute_column]).reindex(train_subset.index)
        path_to_label = dict(
            zip(train_subset["image_path"].astype(str).values, cleaned.values, strict=False)
        )
        neighbour_paths = index.image_paths[top_idx]
        preds = [
            _predict_categorical_label(
                pd.Series([path_to_label.get(str(p)) for p in row])
            )
            for row in neighbour_paths
        ]
        return pd.Series(preds, index=test_index, dtype=object)

    # Numeric: mean of neighbors
    path_to_value = dict(
        zip(
            train_subset["image_path"].astype(str).values,
            pd.to_numeric(train_subset[attribute_column], errors="coerce").values,
        )
    )
    neighbour_paths = index.image_paths[top_idx]
    preds = [
        _predict_numeric_value(
            pd.Series([path_to_value.get(str(p)) for p in row])
        )
        for row in neighbour_paths
    ]
    return pd.Series(preds, index=test_index)



def predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: Schema,
    *,
    feature_cache: FeatureCache,
    k: int = DEFAULT_K,
) -> pd.DataFrame:
    """Pipeline predict function compatible with :func:`src.models.run_pipeline`."""

    def per_attr(at, ac, train_subset, test_subset):
        return _knn_one_attribute(
            at,
            ac,
            train_subset,
            test_subset,
            schema=schema,
            cache=feature_cache,
            k=k,
        )

    return predict_per_asset_type(train_df, test_df, schema, per_attr)


def retrieve_similar_assets(
    query_image_paths: list[str],
    *,
    train_df: pd.DataFrame,
    feature_cache: FeatureCache,
    k: int = 5,
    same_asset_type: bool = True,
) -> pd.DataFrame:
    """Return a top-k similar-image table for partner-facing review.

    Output columns: ``query_image_path``, ``rank``, ``neighbor_asset_id``,
    ``neighbor_image_path``, ``cosine_similarity``, ``neighbor_profile_name``.
    """
    out_rows: list[dict] = []
    train_lookup = train_df.set_index("image_path", drop=False)
    query_feats, missing = feature_cache.aligned_to(query_image_paths)

    if same_asset_type:
        if not hasattr(retrieve_similar_assets, "_schema_cache"):
            retrieve_similar_assets._schema_cache = load_schema()

    for i, qpath in enumerate(query_image_paths):
        if missing[i]:
            continue
        if same_asset_type and qpath in train_lookup.index:
            profile_name = train_lookup.loc[qpath, "profile_name"]
            subset = train_df[train_df["profile_name"] == profile_name]
        else:
            subset = train_df
        index = _build_index(subset, feature_cache)
        if index.feature_matrix.shape[0] == 0:
            continue
        top_idx, top_sims = index.query(query_feats[i : i + 1], k=k)
        for rank, (idx, sim) in enumerate(
            zip(top_idx[0], top_sims[0], strict=False), start=1
        ):
            neighbour_path = index.image_paths[idx]
            row = train_lookup.loc[neighbour_path]
            out_rows.append(
                {
                    "query_image_path": qpath,
                    "rank": rank,
                    "neighbor_asset_id": int(row["asset_id"]),
                    "neighbor_image_path": neighbour_path,
                    "neighbor_profile_name": row.get("profile_name"),
                    "cosine_similarity": float(sim),
                }
            )
    return pd.DataFrame(out_rows)


def _make_group_splitter(
    labelled: pd.DataFrame,
    target_column: str,
    group_column: str,
    n_splits: int,
    random_state: int,
) -> tuple[object, str]:
    """Prefer StratifiedGroupKFold when every class has enough assets."""
    class_group_counts = labelled.groupby(target_column)[group_column].nunique()
    if int(class_group_counts.min()) >= n_splits:
        return (
            StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            ),
            "StratifiedGroupKFold",
        )
    return (
        GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        "GroupKFold",
    )


def _compute_classification_metrics(y_valid: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute classification metrics (accuracy, F1 scores)."""
    return {
        "accuracy": float(accuracy_score(y_valid, y_pred)),
        "weighted_f1": float(f1_score(y_valid, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_valid, y_pred, average="macro", zero_division=0)),
    }


def _compute_regression_metrics(y_valid: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute regression metrics (MAE, RMSE, R2)."""
    y_pred_numeric = pd.to_numeric(y_pred, errors="coerce")
    y_valid_numeric = pd.to_numeric(y_valid, errors="coerce")
    mask = y_pred_numeric.notna() & y_valid_numeric.notna()
    if mask.sum() < 2:
        return {}
    yp = y_pred_numeric[mask]
    yt = y_valid_numeric[mask]
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
    }



def cross_validate_knn_frame(
    df: pd.DataFrame,
    target: str,
    *,
    target_file: str | None = None,
    cache: FeatureCache,
    schema: Schema,
    k: int = DEFAULT_K,
    n_splits: int = 5,
    random_state: int = 42,
    group_column: str = "asset_id",
) -> pd.DataFrame:
    """Cross-validate k-NN one attribute with grouped asset folds."""
    if group_column not in df.columns:
        raise ValueError(f"Missing required group column {group_column!r}.")
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in DataFrame.")

    # Filter to rows with labels and prepare for CV
    labelled = df.loc[df[target].notna(), [target, group_column]].copy()
    original_indices = labelled.index.tolist()
    labelled = labelled.reset_index(drop=True)
    if len(labelled) < 2:
        return pd.DataFrame()

    n_asset_groups = labelled[group_column].nunique()
    target_splits = min(n_splits, n_asset_groups)
    if target_splits < 2:
        return pd.DataFrame()

    # Set up CV splitter and attribute metadata
    splitter, splitter_name = _make_group_splitter(
        labelled, target, group_column, target_splits, random_state
    )
    try:
        kind = schema.kind_of(target)
        is_classification = kind in {
            AttributeKind.CATEGORICAL,
            AttributeKind.BOOLEAN,
            AttributeKind.ORDINAL_BIN,
        }
    except KeyError:
        # _bin columns not in schema — treat as classification (majority vote)
        is_classification = True

    # Run k-fold CV
    rows: list[dict[str, object]] = []
    y = labelled[target]
    groups = labelled[group_column]

    for fold_num, (train_idx, valid_idx) in enumerate(
        splitter.split(labelled, y, groups), start=1
    ):
        # Get fold data
        train_original_indices = [original_indices[i] for i in train_idx]
        valid_original_indices = [original_indices[i] for i in valid_idx]

        train_subset = df.loc[train_original_indices]
        valid_subset = df.loc[valid_original_indices]

        # Check for asset leakage
        train_assets = set(labelled.iloc[train_idx][group_column])
        valid_assets = set(labelled.iloc[valid_idx][group_column])
        overlap = train_assets & valid_assets
        if overlap:
            raise AssertionError(
                f"Asset leakage in {target} fold {fold_num}: {sorted(overlap)[:5]}"
            )

        # Make k-NN predictions
        preds = _knn_one_attribute(
            asset_type_id="",
            attribute_column=target,
            train_subset=train_subset,
            test_subset=valid_subset,
            schema=schema,
            cache=cache,
            k=k,
        )

        if preds is None:
            continue

        # Compute metrics
        y_pred = preds.dropna()
        y_valid = labelled.iloc[valid_idx][target].loc[y_pred.index]

        if len(y_valid) == 0:
            continue

        metrics = (
            _compute_classification_metrics(y_valid, y_pred)
            if is_classification
            else _compute_regression_metrics(y_valid, y_pred)
        )

        rows.append(
            {
                "attribute": target,
                "target_column": target,
                "target_file": target_file,
                "task_type": "classification" if is_classification else "regression",
                "strategy": "knn_group_cv",
                "splitter": splitter_name,
                "fold": fold_num,
                "n_folds": target_splits,
                "knn_k": k,
                "n_train_labels": int(labelled.iloc[train_idx][target].notna().sum()),
                "n_valid_labels": len(y_valid),
                "n_train_assets": len(train_assets),
                "n_valid_assets": len(valid_assets),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def summarize_knn_cv_folds(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-fold k-NN metrics with mean and std."""
    if fold_results.empty:
        return pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    group_columns = [
        "attribute",
        "target_column",
        "target_file",
        "task_type",
        "strategy",
        "splitter",
        "knn_k",
    ]
    for keys, group in fold_results.groupby(group_columns, dropna=False):
        values = dict(zip(group_columns, keys, strict=True))
        values.update(
            {
                "n_folds": int(group["fold"].max()),
                "n_labels": int(group["n_valid_labels"].sum()),
                "n_assets": int(group["n_valid_assets"].sum()),
            }
        )

        # Aggregate metrics based on task type
        task_type = group["task_type"].iloc[0]
        if task_type == "classification":
            values.update(
                {
                    "accuracy_mean": float(group["accuracy"].mean()),
                    "accuracy_std": float(group["accuracy"].std(ddof=0)),
                    "weighted_f1_mean": float(group["weighted_f1"].mean()),
                    "weighted_f1_std": float(group["weighted_f1"].std(ddof=0)),
                    "macro_f1_mean": float(group["macro_f1"].mean()),
                    "macro_f1_std": float(group["macro_f1"].std(ddof=0)),
                }
            )
        else:
            values.update(
                {
                    "mae_mean": float(group["mae"].mean()),
                    "mae_std": float(group["mae"].std(ddof=0)),
                    "rmse_mean": float(group["rmse"].mean()),
                    "rmse_std": float(group["rmse"].std(ddof=0)),
                    "r2_mean": float(group["r2"].mean()),
                    "r2_std": float(group["r2"].std(ddof=0)),
                }
            )

        summary_rows.append(values)

    return pd.DataFrame(summary_rows).sort_values("attribute").reset_index(drop=True)


def cross_validate_knn_folder(
    feature_dir: str | Path = "data/features",
    train_dir: str | Path = "data/processed/train",
    *,
    knn_k: int = DEFAULT_K,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run grouped k-NN CV for all dinov3 feature CSVs."""
    feature_path = Path(feature_dir)
    train_path = Path(train_dir)
    schema = load_schema()
    fold_tables: list[pd.DataFrame] = []

    # Find all dinov3_attr_*_train_images.csv files
    feature_files = sorted(feature_path.glob("dinov3_attr_*_train_images.csv"))

    for feature_csv in feature_files:
        # Infer the train CSV path from feature file name
        stem = feature_csv.stem  # e.g., "dinov3_attr_decking_material_train_images"
        attr_name = stem.replace("dinov3_", "").replace("_images", "")  # e.g., "attr_decking_material_train"
        train_csv = train_path / f"{attr_name}.csv"

        # Extract the actual target column name (remove '_train' suffix for column lookup)
        target_column = attr_name.replace("_train", "")  # e.g., "attr_decking_material"
        if target_column == "attr_material_frame_tank_body":
            target_column = "attr_material_frame,_tank,_body"

        if not train_csv.exists():
            logger.warning(f"Train file not found: {train_csv}, skipping {feature_csv.name}")
            continue

        logger.info(f"Processing {target_column}...")

        try:
            # Load feature cache
            feature_df = pd.read_csv(feature_csv)
            cache = FeatureCache(df=feature_df, model_id="dinov3_vitb16")

            # Load train data
            train_df = pd.read_csv(train_csv)

            # Run CV for this attribute
            fold_table = cross_validate_knn_frame(
                train_df,
                target_column,
                target_file=str(train_csv),
                cache=cache,
                schema=schema,
                k=knn_k,
                n_splits=n_splits,
                random_state=random_state,
            )

            if not fold_table.empty:
                fold_tables.append(fold_table)
        except Exception as e:
            logger.error(f"Error processing {target_column}: {e}")
            continue
        
    # --- Second pass: bin targets that reuse a parent feature file ---
    for bin_target, feature_stem in BIN_TARGET_TO_FEATURE_STEM.items():
        feature_csv = feature_path / f"{feature_stem}.csv"
        train_csv = train_path / f"{bin_target}_train.csv"

        if not feature_csv.exists():
            logger.warning(f"Feature file not found for bin target {bin_target!r}: {feature_csv}")
            continue
        if not train_csv.exists():
            logger.warning(f"Train file not found for bin target {bin_target!r}: {train_csv}")
            continue

        logger.info(f"Processing bin target {bin_target!r} ...")
        try:
            feature_df = pd.read_csv(feature_csv)
            cache = FeatureCache(df=feature_df, model_id="dinov3_vitb16")
            train_df = pd.read_csv(train_csv)

            fold_table = cross_validate_knn_frame(
                train_df,
                bin_target,
                target_file=str(train_csv),
                cache=cache,
                schema=schema,
                k=knn_k,
                n_splits=n_splits,
                random_state=random_state,
            )
            if not fold_table.empty:
                fold_tables.append(fold_table)
        except Exception as e:
            logger.error(f"Error processing bin target {bin_target!r}: {e}")
            continue

    if not fold_tables:
        return pd.DataFrame(), pd.DataFrame()

    fold_results = pd.concat(fold_tables, ignore_index=True)
    summary = summarize_knn_cv_folds(fold_results)
    return summary, fold_results
