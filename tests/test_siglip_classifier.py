"""Tests for SigLIP classifier helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.siglip_classifier import cross_validate_siglip_classifier  # noqa: E402


def test_cross_validate_siglip_classifier_handles_one_class_sample() -> None:
    labels = pd.DataFrame(
        {
            "asset_id": [1, 2, 3],
            "attr_decking_material": ["Timber", "Timber", "Timber"],
        }
    )
    features = pd.DataFrame(
        {
            "asset_id": [1, 2, 3],
            "f_0000": [0.1, 0.2, 0.3],
            "f_0001": [1.1, 1.2, 1.3],
        }
    )

    summary, folds, predictions = cross_validate_siglip_classifier(
        labels,
        features,
        "attr_decking_material",
    )

    assert summary.empty
    assert folds.empty
    assert predictions.empty
