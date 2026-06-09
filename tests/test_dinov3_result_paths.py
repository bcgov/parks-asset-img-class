"""Tests for standard DINOv3 result and prediction folders."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_dinov3_classifier import default_output_dir, default_prediction_dir  # noqa: E402


def test_default_output_dir_uses_classifier_folder() -> None:
    assert default_output_dir("logistic_regression") == Path(
        "results/dinov3_results/dinov3_logistic"
    )
    assert default_output_dir("linear_svm") == Path(
        "results/dinov3_results/dinov3_linear_svm"
    )
    assert default_output_dir("logistic_regression_tuned") == Path(
        "results/dinov3_results/dinov3_logistic_tuned"
    )


def test_default_prediction_dir_uses_classifier_folder() -> None:
    assert default_prediction_dir("random_forest") == Path(
        "data/predictions/dinov3_predictions/dinov3_random_forest"
    )
    assert default_prediction_dir("hist_gradient_boosting") == Path(
        "data/predictions/dinov3_predictions/dinov3_gradient_boost"
    )
    assert default_prediction_dir("logistic_regression_tuned") == Path(
        "data/predictions/dinov3_predictions/dinov3_logistic_tuned"
    )
