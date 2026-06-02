"""Tests for DINOv3 classifier helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dinov3_classifier import CLASSIFIER_CHOICES, make_classifier, join_labels_and_features  # noqa: E402


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
