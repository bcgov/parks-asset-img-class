"""MLflow tracking helpers (issue #10).

Tiny set of helpers so every model run logs to a consistent local store
and carries the same identifying tags. Each model owner decides what to
log themselves with the regular ``mlflow.log_param`` / ``mlflow.log_metric``
API; this module only standardises *where* runs go and the *naming /
tagging* convention.

Default store
-------------
A file directory at ``./mlruns`` (gitignored), so no server is required
and nothing leaves the machine. Override with ``MLFLOW_TRACKING_URI`` if
you ever need a different backend.

Standard tags every run should carry
------------------------------------
* ``task``           - e.g. ``T1_relevance``, ``T2_decking_material``, ``T2_length_m``
* ``pipeline``       - e.g. ``baseline``, ``vectordb_knn``, ``dinov3_mlp``
* ``model_family``   - e.g. ``baseline``, ``catboost``, ``dinov3``, ``vlm``
* ``model_name``     - e.g. ``majority_class``, ``knn_k10__dinov2_large``
* ``data_version``   - free-form string, e.g. the date of the data drop
* ``split_seed``     - int seed used for the train/val/test split
* ``weights_name``   - name of the attribute-weight config in use

The ``log_pipeline_run`` / ``log_attribute_run`` helpers below take a
:class:`src.eval.metrics.PerAttributeReport` and write one parent run
per pipeline with one nested child run per attribute, so the MLflow UI
groups them cleanly.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import mlflow

if TYPE_CHECKING:
    from src.eval.metrics import AttributeScore, PerAttributeReport

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "parks-asset-img-class"
DEFAULT_TRACKING_URI = "file:./mlruns"


def setup_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: str | None = None,
) -> str:
    """Configure MLflow to use the project store and return the experiment id.

    Honours ``MLFLOW_TRACKING_URI`` when ``tracking_uri`` is not given.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking_uri=%s experiment=%s", uri, experiment_name)
    return experiment.experiment_id


def make_run_name(task: str, model_name: str) -> str:
    """Run names look like ``T2_decking_material__majority_class``."""
    return f"{task}__{model_name}"


def make_standard_tags(
    *,
    task: str,
    model_family: str,
    model_name: str,
    data_version: str | None = None,
    split_seed: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Build the standard tag dict every run should carry."""
    tags: dict[str, str] = {
        "task": task,
        "model_family": model_family,
        "model_name": model_name,
    }
    if data_version is not None:
        tags["data_version"] = str(data_version)
    if split_seed is not None:
        tags["split_seed"] = str(split_seed)
    if extra:
        for k, v in extra.items():
            tags[str(k)] = str(v)
    return tags


def log_attribute_run(
    *,
    score: "AttributeScore",
    pipeline: str,
    model_family: str,
    model_name: str,
    weights_name: str,
    data_version: str | None = None,
    split_seed: int | None = None,
    params: Mapping[str, Any] | None = None,
    extra_tags: Mapping[str, Any] | None = None,
    nested: bool = True,
) -> str:
    """Log one attribute's score as an MLflow (child) run."""
    task = f"T3_{score.attribute}"
    tags = make_standard_tags(
        task=task,
        model_family=model_family,
        model_name=model_name,
        data_version=data_version,
        split_seed=split_seed,
        extra={
            "pipeline": pipeline,
            "weights_name": weights_name,
            "attribute_kind": score.kind,
            **(extra_tags or {}),
        },
    )

    with mlflow.start_run(
        run_name=make_run_name(task, model_name),
        tags=tags,
        nested=nested,
    ) as run:
        if params:
            mlflow.log_params({str(k): v for k, v in params.items()})
        mlflow.log_metric("n", float(score.n))
        for k, v in score.metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except (TypeError, ValueError):
                continue
        if score.extras:
            mlflow.log_dict(score.extras, f"{score.attribute}_extras.json")
        return run.info.run_id


def log_pipeline_run(
    *,
    report: "PerAttributeReport",
    model_family: str,
    model_name: str,
    data_version: str | None = None,
    split_seed: int | None = None,
    params: Mapping[str, Any] | None = None,
    extra_tags: Mapping[str, Any] | None = None,
) -> str:
    """Log a pipeline-level run plus one nested run per attribute."""
    tags = make_standard_tags(
        task="T3_pipeline",
        model_family=model_family,
        model_name=model_name,
        data_version=data_version,
        split_seed=split_seed,
        extra={
            "pipeline": report.pipeline,
            "weights_name": report.weights_name,
            **(extra_tags or {}),
        },
    )

    with mlflow.start_run(
        run_name=f"pipeline__{report.pipeline}__{model_name}",
        tags=tags,
    ) as parent:
        if params:
            mlflow.log_params({str(k): v for k, v in params.items()})

        for k, v in report.aggregate().items():
            try:
                mlflow.log_metric(k, float(v))
            except (TypeError, ValueError):
                continue

        mlflow.log_dict(report.as_dict(), "per_attribute_report.json")
        mlflow.log_dict(
            {"name": report.weights_name, "weights": report.weights},
            "attribute_weights.json",
        )
        mlflow.log_text(
            report.per_attribute_table().to_csv(index=False),
            "per_attribute_report.csv",
        )

        for score in report.scores.values():
            log_attribute_run(
                score=score,
                pipeline=report.pipeline,
                model_family=model_family,
                model_name=model_name,
                weights_name=report.weights_name,
                data_version=data_version,
                split_seed=split_seed,
                params=None,
                extra_tags=extra_tags,
                nested=True,
            )

        return parent.info.run_id


__all__ = [
    "DEFAULT_EXPERIMENT_NAME",
    "DEFAULT_TRACKING_URI",
    "log_attribute_run",
    "log_pipeline_run",
    "make_run_name",
    "make_standard_tags",
    "setup_mlflow",
]
