"""Classifiers trained on frozen SigLIP asset embeddings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dinov3_classifier import (
    CLASSIFIER_CHOICES,
    _make_group_splitter,
    join_labels_and_features,
    make_classifier,
)
from sklearn.metrics import accuracy_score, f1_score


def cross_validate_siglip_classifier(
    labels: pd.DataFrame,
    asset_features: pd.DataFrame,
    target: str,
    *,
    target_file: str | None = None,
    feature_file: str | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    group_column: str = "asset_id",
    classifier: str = "logistic_regression",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate SigLIP embeddings with grouped cross-validation.

    Returns ``(summary, folds, predictions)``.
    """
    joined, target_column, features = join_labels_and_features(
        labels,
        asset_features,
        target,
    )
    if len(joined) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if joined[target_column].nunique(dropna=True) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    n_asset_groups = joined[group_column].nunique()
    target_splits = min(n_splits, n_asset_groups)
    if target_splits < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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
    prediction_rows: list[dict[str, object]] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(joined, y, groups), start=1):
        train_assets = set(joined.iloc[train_idx][group_column])
        valid_assets = set(joined.iloc[valid_idx][group_column])
        overlap = train_assets & valid_assets
        if overlap:
            raise AssertionError(
                f"Asset leakage in {target} fold {fold}: {sorted(overlap)[:5]}"
            )

        model = make_classifier(classifier=classifier, random_state=random_state)
        y_train = y.iloc[train_idx]
        if y_train.nunique(dropna=True) < 2:
            continue

        model.fit(X.iloc[train_idx], y_train)
        predictions = model.predict(X.iloc[valid_idx])
        y_valid = y.iloc[valid_idx]

        # Save per-asset predictions for error analysis
        for asset_id, true_label, pred_label in zip(
            joined.iloc[valid_idx][group_column], y_valid, predictions
        ):
            prediction_rows.append(
                {
                    "attribute": target,
                    "fold": fold,
                    "asset_id": asset_id,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "correct": true_label == pred_label,
                }
            )

        fold_rows.append(
            {
                "attribute": target,
                "target_column": target_column,
                "target_file": target_file,
                "feature_file": feature_file,
                "task_type": "classification",
                "strategy": f"siglip_frozen_embeddings_{classifier}",
                "classifier": classifier,
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
    predictions_df = pd.DataFrame(prediction_rows)
    summary = summarize_siglip_folds(folds)
    return summary, folds, predictions_df


def summarize_siglip_folds(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-fold SigLIP classifier metrics."""
    if fold_results.empty:
        return pd.DataFrame()

    group_columns = [
        "attribute",
        "target_column",
        "target_file",
        "feature_file",
        "task_type",
        "strategy",
        "classifier",
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
    classifier: str = "logistic_regression",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read files and run one SigLIP classification task."""
    labels = pd.read_csv(labels_path)
    features = pd.read_csv(features_path)
    return cross_validate_siglip_classifier(
        labels,
        features,
        target,
        target_file=str(labels_path),
        feature_file=str(features_path),
        n_splits=n_splits,
        random_state=random_state,
        classifier=classifier,
    )


__all__ = [
    "CLASSIFIER_CHOICES",
    "cross_validate_siglip_classifier",
    "join_labels_and_features",
    "make_classifier",
    "run_task_from_files",
    "summarize_siglip_folds",
]
