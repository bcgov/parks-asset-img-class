"""Tests for SigLIP comparison path defaults."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_siglip_to_baseline import (  # noqa: E402
    default_comparison_output,
    default_result_glob,
    make_comparison,
)


def test_default_result_glob_uses_logistic_folder() -> None:
    assert default_result_glob("logistic_regression") == (
        "results/siglip_results/siglip_logistic_reg/"
        "siglip_*_classification_results.csv"
    )


def test_default_result_glob_uses_linear_svm_suffix() -> None:
    assert default_result_glob("linear_svm") == (
        "results/siglip_results/siglip_linear_svm/"
        "siglip_*_linear_svm_classification_results.csv"
    )


def test_default_comparison_output_uses_classifier_folder() -> None:
    assert default_comparison_output("random_forest") == Path(
        "results/siglip_results/siglip_random_forest/"
        "siglip_random_forest_vs_baseline_comparison.csv"
    )


def test_make_comparison_filters_majority_baseline_strategy() -> None:
    baseline = pd.DataFrame(
        {
            "attribute": ["attr_a", "attr_a"],
            "strategy": ["majority_class_group_cv", "uniform_random_group_cv"],
            "prediction": ["A", "B"],
            "n_labels": [10, 10],
            "n_assets": [5, 5],
            "accuracy_mean": [0.7, 0.1],
            "weighted_f1_mean": [0.6, 0.1],
            "macro_f1_mean": [0.5, 0.1],
        }
    )
    siglip = pd.DataFrame(
        {
            "attribute": ["attr_a"],
            "n_labels": [10],
            "n_assets": [5],
            "n_features": [768],
            "feature_file": ["features.csv"],
            "siglip_result_file": ["results.csv"],
            "accuracy_mean": [0.8],
            "weighted_f1_mean": [0.7],
            "macro_f1_mean": [0.65],
        }
    )

    comparison = make_comparison(baseline, siglip)

    assert len(comparison) == 1
    assert comparison.loc[0, "macro_f1_mean_baseline"] == 0.5
    assert comparison.loc[0, "macro_f1_mean_delta"] == 0.15000000000000002
