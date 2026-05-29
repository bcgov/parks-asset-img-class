"""Classifiers trained on frozen DINOv3 asset embeddings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.baseline import first_mode, infer_target_column
from src.dinov3_features import feature_columns


def make_classifier(random_state: int = 42) -> object:
    """Return the first classifier used for DINOv3 experiments.

    Logistic regression is deliberately boring here: the signal should come
    from DINOv3 embeddings, and a simple linear classifier is easy to compare
    against the majority-class baseline.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
    )


def join_labels_and_features(
    labels: pd.DataFrame,
    asset_features: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Join a task CSV with asset-level DINOv3 features."""
    if "asset_id" not in labels.columns:
        raise ValueError("labels must contain an 'asset_id' column.")
    if "asset_id" not in asset_features.columns:
        raise ValueError("asset_features must contain an 'asset_id' column.")

    target_column = infer_target_column(labels, target)
    features = feature_columns(asset_features.columns)
    if not features:
        raise ValueError("asset_features does not contain f_* feature columns.")

    labelled = labels[["asset_id", target_column]].dropna(subset=[target_column])
    labelled = labelled.drop_duplicates(["asset_id", target_column])
    joined = labelled.merge(asset_features[["asset_id", *features]], on="asset_id", how="inner")
    joined = joined.drop_duplicates("asset_id").reset_index(drop=True)
    return joined, target_column, features


def _make_group_splitter(
    labelled: pd.DataFrame,
    target_column: str,
    group_column: str,
    n_splits: int,
    random_state: int,
) -> tuple[object, str]:
    class_group_counts = labelled.groupby(target_column)[group_column].nunique()
    if int(class_group_counts.min()) >= n_splits:
        return (
            StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            ),
            "StratifiedGroupKFold",
        )
    return (
        GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        "GroupKFold",
    )


def cross_validate_dinov3_classifier(
    labels: pd.DataFrame,
    asset_features: pd.DataFrame,
    target: str,
    *,
    target_file: str | None = None,
    feature_file: str | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    group_column: str = "asset_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate DINOv3 embeddings with grouped cross-validation."""
    joined, target_column, features = join_labels_and_features(
        labels,
        asset_features,
        target,
    )
    if len(joined) < 2:
        return pd.DataFrame(), pd.DataFrame()

    n_asset_groups = joined[group_column].nunique()
    target_splits = min(n_splits, n_asset_groups)
    if target_splits < 2:
        return pd.DataFrame(), pd.DataFrame()

    splitter, splitter_name = _make_group_splitter(
        joined,
        target_column,
        group_column,
        target_splits,
        random_state,
    )

    X = joined[features]
    y = joined[target_column]
    groups = joined[group_column]
    fold_rows: list[dict[str, object]] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(joined, y, groups), start=1):
        train_assets = set(joined.iloc[train_idx][group_column])
        valid_assets = set(joined.iloc[valid_idx][group_column])
        overlap = train_assets & valid_assets
        if overlap:
            raise AssertionError(
                f"Asset leakage in {target} fold {fold}: {sorted(overlap)[:5]}"
            )

        model = make_classifier(random_state=random_state)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = model.predict(X.iloc[valid_idx])
        y_valid = y.iloc[valid_idx]

        fold_rows.append(
            {
                "attribute": target,
                "target_column": target_column,
                "target_file": target_file,
                "feature_file": feature_file,
                "task_type": "classification",
                "strategy": "dinov3_frozen_embeddings_logistic_regression",
                "splitter": splitter_name,
                "fold": fold,
                "n_folds": target_splits,
                "n_features": len(features),
                "n_train_labels": int(len(train_idx)),
                "n_valid_labels": int(len(valid_idx)),
                "n_train_assets": len(train_assets),
                "n_valid_assets": len(valid_assets),
                "accuracy": accuracy_score(y_valid, predictions),
                "weighted_f1": f1_score(y_valid, predictions, average="weighted", zero_division=0),
                "macro_f1": f1_score(y_valid, predictions, average="macro", zero_division=0),
            }
        )

    folds = pd.DataFrame(fold_rows)
    summary = summarize_dinov3_folds(folds)
    return summary, folds


def summarize_dinov3_folds(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-fold DINOv3 classifier metrics."""
    if fold_results.empty:
        return pd.DataFrame()

    group_columns = [
        "attribute",
        "target_column",
        "target_file",
        "feature_file",
        "task_type",
        "strategy",
        "splitter",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in fold_results.groupby(group_columns, dropna=False):
        values = dict(zip(group_columns, keys, strict=True))
        values.update(
            {
                "n_folds": int(group["fold"].max()),
                "n_labels": int(group["n_valid_labels"].sum()),
                "n_assets": int(group["n_valid_assets"].sum()),
                "n_features": int(group["n_features"].iloc[0]),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=0),
                "weighted_f1_mean": group["weighted_f1"].mean(),
                "weighted_f1_std": group["weighted_f1"].std(ddof=0),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=0),
            }
        )
        rows.append(values)

    return pd.DataFrame(rows).sort_values("attribute").reset_index(drop=True)


def run_task_from_files(
    *,
    labels_path: str | Path,
    features_path: str | Path,
    target: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read files and run one DINOv3 classification task."""
    labels = pd.read_csv(labels_path)
    features = pd.read_csv(features_path)
    return cross_validate_dinov3_classifier(
        labels,
        features,
        target,
        target_file=str(labels_path),
        feature_file=str(features_path),
        n_splits=n_splits,
        random_state=random_state,
    )


__all__ = [
    "cross_validate_dinov3_classifier",
    "join_labels_and_features",
    "make_classifier",
    "run_task_from_files",
    "summarize_dinov3_folds",
]

