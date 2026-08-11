"""Tests for standard SigLIP result folders."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_siglip_classifier import default_output_dir  # noqa: E402


def test_default_output_dir_uses_existing_logistic_folder_name() -> None:
    assert default_output_dir("logistic_regression") == Path(
        "results/siglip_results/siglip_logistic_reg"
    )


def test_default_output_dir_uses_linear_svm_folder() -> None:
    assert default_output_dir("linear_svm") == Path(
        "results/siglip_results/siglip_linear_svm"
    )


def test_default_output_dir_uses_tree_model_folders() -> None:
    assert default_output_dir("random_forest") == Path(
        "results/siglip_results/siglip_random_forest"
    )
    assert default_output_dir("hist_gradient_boosting") == Path(
        "results/siglip_results/siglip_gradient_boost"
    )
