"""Tests for DINOv3 classifier helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dinov3_classifier import (  # noqa: E402
    CLASSIFIER_CHOICES,
    cross_validate_dinov3_classifier,
    join_labels_and_features,
    make_classifier,
)


def test_join_labels_and_features_matches_asset_id_and_target() -> None:
    labels = pd.DataFrame(
        {
            "asset_id": [1, 2, 3],
            "attr_decking_material": ["Timber", "Concrete", None],
        }
    )
    features = pd.DataFrame(
        {
            "asset_id": [1, 2, 4],
            "f_0000": [0.1, 0.2, 0.4],
            "f_0001": [1.1, 1.2, 1.4],
        }
    )

    joined, target_column, feature_cols = join_labels_and_features(
        labels,
        features,
        "attr_decking_material",
    )

    assert target_column == "attr_decking_material"
    assert feature_cols == ["f_0000", "f_0001"]
    assert joined["asset_id"].tolist() == [1, 2]


def test_classifier_choices_are_constructible() -> None:
    for classifier in CLASSIFIER_CHOICES:
        assert make_classifier(classifier=classifier) is not None


def test_cross_validate_dinov3_classifier_returns_predictions() -> None:
    labels = pd.DataFrame(
        {
            "asset_id": [1, 2, 3, 4],
            "attr_decking_material": ["Timber", "Concrete", "Timber", "Concrete"],
        }
    )
    features = pd.DataFrame(
        {
            "asset_id": [1, 2, 3, 4],
            "f_0000": [0.1, 1.0, 0.2, 1.1],
            "f_0001": [0.1, 1.0, 0.2, 1.1],
        }
    )

    summary, folds, predictions = cross_validate_dinov3_classifier(
        labels,
        features,
        "attr_decking_material",
        n_splits=2,
    )

    assert not summary.empty
    assert len(folds) == 2
    assert len(predictions) == 4
    assert {
        "asset_id",
        "fold",
        "y_true",
        "y_pred",
        "is_correct",
    }.issubset(predictions.columns)


def test_tuned_logistic_regression_records_selected_hyperparameters() -> None:
    labels = pd.DataFrame(
        {
            "asset_id": list(range(1, 13)),
            "attr_decking_material": ["Timber", "Concrete"] * 6,
        }
    )
    features = pd.DataFrame(
        {
            "asset_id": list(range(1, 13)),
            "f_0000": [0.1, 1.0, 0.2, 1.1, 0.15, 1.05, 0.3, 1.2, 0.25, 1.15, 0.4, 1.3],
            "f_0001": [0.1, 1.0, 0.2, 1.1, 0.15, 1.05, 0.3, 1.2, 0.25, 1.15, 0.4, 1.3],
        }
    )

    _, folds, predictions = cross_validate_dinov3_classifier(
        labels,
        features,
        "attr_decking_material",
        n_splits=3,
        classifier="logistic_regression_tuned",
    )

    assert "tuned_C" in folds.columns
    assert "tuned_class_weight" in folds.columns
    assert "inner_macro_f1_mean" in folds.columns
    assert folds["tuned_C"].notna().all()
    assert predictions["tuned_C"].notna().all()
