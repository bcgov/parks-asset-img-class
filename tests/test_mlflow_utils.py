"""Smoke tests for the MLflow scaffolding (issue #10)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlflow  # noqa: E402

from src.mlflow_utils import (  # noqa: E402
    make_run_name,
    make_standard_tags,
    setup_mlflow,
)


def test_make_run_name() -> None:
    assert (
        make_run_name("T2_decking_material", "majority_class")
        == "T2_decking_material__majority_class"
    )


def test_make_standard_tags_includes_required_keys() -> None:
    tags = make_standard_tags(
        task="T2_decking_material",
        model_family="baseline",
        model_name="majority_class",
        data_version="2026-05-05",
        split_seed=7,
        extra={"smoke_test": "true"},
    )
    assert tags["task"] == "T2_decking_material"
    assert tags["model_family"] == "baseline"
    assert tags["model_name"] == "majority_class"
    assert tags["data_version"] == "2026-05-05"
    assert tags["split_seed"] == "7"
    assert tags["smoke_test"] == "true"


def test_setup_and_log_round_trip(tmp_path: Path) -> None:
    """Open a run against an isolated tmp store, log a metric, read it back."""
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    setup_mlflow(experiment_name="test-exp", tracking_uri=tracking_uri)
    with mlflow.start_run(
        run_name="round_trip",
        tags=make_standard_tags(
            task="t_test", model_family="baseline", model_name="majority_class"
        ),
    ) as run:
        mlflow.log_metric("accuracy", 0.42)

    fetched = mlflow.get_run(run.info.run_id)
    assert fetched.data.metrics["accuracy"] == 0.42
    assert fetched.data.tags["task"] == "t_test"
