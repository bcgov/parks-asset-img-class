"""MLflow tracking helpers (issue #10).

Tiny set of helpers so every model run logs to a consistent local store
and carries the same identifying tags. Each model owner decides what to
log themselves with the regular ``mlflow.log_param`` / ``mlflow.log_metric``
API; this module only standardises *where* runs go and the *naming /
tagging* convention.

Default store
-------------
A local SQLite database at ``./mlflow.db`` (gitignored), so no server is
required and nothing leaves the machine. Override with ``MLFLOW_TRACKING_URI``
if you ever need a different backend.

Standard tags every run should carry
------------------------------------
* ``task``           - e.g. ``T1_relevance``, ``T2_decking_material``, ``T2_length_m``
* ``model_family``   - e.g. ``baseline``, ``catboost``, ``dinov3``
* ``model_name``     - e.g. ``majority_class``, ``median_regressor``
* ``data_version``   - free-form string, e.g. the date of the data drop
* ``split_seed``     - int seed used for the train/val/test split
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

import mlflow

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "parks-asset-img-class"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


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


__all__ = [
    "DEFAULT_EXPERIMENT_NAME",
    "DEFAULT_TRACKING_URI",
    "make_run_name",
    "make_standard_tags",
    "setup_mlflow",
]
