"""Classifiers trained on frozen OpenCLIP asset embeddings.

The cross-validation logic (per-asset-type training, pooled-prediction summary,
confidence scores, single clean prediction rows) lives in dinov3_classifier.py
and is model-agnostic — it only cares about f_* feature columns. This module
delegates to it and relabels the ``strategy`` output so results are tagged as
OpenCLIP. Single source of truth for CV logic, per-model entry point preserved.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dinov3_classifier import (
    CLASSIFIER_CHOICES,
    cross_validate_dinov3_classifier,
    join_labels_and_features,
    make_classifier,
    summarize_dinov3_folds,
)

MODEL_FAMILY = "openclip"


def _relabel_strategy(frame: pd.DataFrame) -> pd.DataFrame:
    """Rewrite the dinov3_* strategy label to openclip_* in a result frame."""
    if frame.empty or "strategy" not in frame.columns:
        return frame
    frame = frame.copy()
    frame["strategy"] = frame["strategy"].str.replace(
        "dinov3_frozen_embeddings_", f"{MODEL_FAMILY}_frozen_embeddings_", regex=False
    )
    return frame


def cross_validate_openclip_classifier(
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
    per_asset_type: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate OpenCLIP embeddings with grouped cross-validation.

    Delegates to the shared DINOv3 implementation (model-agnostic CV) and
    relabels the strategy column to mark these as OpenCLIP results.

    Returns ``(summary, folds, predictions)``.
    """
    summary, folds, predictions = cross_validate_dinov3_classifier(
        labels,
        asset_features,
        target,
        target_file=target_file,
        feature_file=feature_file,
        n_splits=n_splits,
        random_state=random_state,
        group_column=group_column,
        classifier=classifier,
        per_asset_type=per_asset_type,
    )
    return _relabel_strategy(summary), _relabel_strategy(folds), _relabel_strategy(predictions)


def summarize_openclip_folds(
    fold_results: pd.DataFrame,
    *,
    predictions: pd.DataFrame | None = None,
    per_asset_type: bool = False,
) -> pd.DataFrame:
    """Summarize OpenCLIP fold metrics (delegates to the shared summarizer)."""
    return _relabel_strategy(
        summarize_dinov3_folds(
            fold_results, predictions=predictions, per_asset_type=per_asset_type
        )
    )


def run_task_from_files(
    *,
    labels_path: str | Path,
    features_path: str | Path,
    target: str,
    n_splits: int = 5,
    random_state: int = 42,
    classifier: str = "logistic_regression",
    per_asset_type: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read files and run one OpenCLIP classification task."""
    labels = pd.read_csv(labels_path)
    features = pd.read_csv(features_path)
    return cross_validate_openclip_classifier(
        labels,
        features,
        target,
        target_file=str(labels_path),
        feature_file=str(features_path),
        n_splits=n_splits,
        random_state=random_state,
        classifier=classifier,
        per_asset_type=per_asset_type,
    )


__all__ = [
    "CLASSIFIER_CHOICES",
    "cross_validate_openclip_classifier",
    "join_labels_and_features",
    "make_classifier",
    "run_task_from_files",
    "summarize_openclip_folds",
]