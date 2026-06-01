"""Shared pipeline scaffolding.

Every model pipeline (baseline, CLIP, DINOv3+head, k-NN, VLM, YOLO,
stack) returns the same shape of object — a ``PipelineResult`` — so the
leaderboard can compare them apples-to-apples and so MLflow logging is
identical across pipelines.

Concretely a pipeline is anything that, given the project's train/test
DataFrames, produces a predictions DataFrame keyed by ``image_path``
with one column per attribute it covers.  We split that contract into
two small helpers:

- :func:`predict_per_asset_type` — the common "loop over asset types and
  call a per-attribute predictor" loop.  Use this when your pipeline's
  per-attribute logic is naturally implemented per (asset_type,
  attribute) pair, like the majority baseline or a CLIP zero-shot
  classifier.
- :func:`run_pipeline` — the outer wrapper that calls a prediction
  function once, scores the result via
  :func:`src.eval.metrics.per_attribute_report`, and writes the
  predictions CSV plus the MLflow pipeline run.  Returns a
  :class:`PipelineResult` so callers can chain into the leaderboard
  renderer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import AttributeKind, Schema, load_schema
from src.eval.metrics import PerAttributeReport, per_attribute_report
from src.mlflow_utils import log_pipeline_run, setup_mlflow
from src.models.oof import run_train_side

PredictFn = Callable[[pd.DataFrame, pd.DataFrame, Schema], pd.DataFrame]
"""Signature: ``predict_fn(train_df, test_df, schema) -> predictions_df``."""

PerAttributeFn = Callable[
    [str, str, pd.DataFrame, pd.DataFrame], pd.Series | None
]
"""Signature for the per-(asset_type, attribute) predictor used by
:func:`predict_per_asset_type`.

``fn(asset_type_id, attribute_column, train_subset_df, test_subset_df)``
returns a ``pd.Series`` aligned with ``test_subset_df.index`` (or None
to skip the attribute for that asset type)."""


@dataclass
class PipelineResult:
    """What every pipeline returns."""

    pipeline: str
    model_name: str
    model_family: str
    report: PerAttributeReport
    predictions: pd.DataFrame
    mlflow_run_id: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def aggregate(self) -> dict[str, float]:
        return self.report.aggregate()


def predict_per_asset_type(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: Schema,
    per_attribute_fn: PerAttributeFn,
) -> pd.DataFrame:
    """Loop over (asset_type, attribute) and run ``per_attribute_fn`` for each.

    Result is a single test-set-aligned DataFrame keyed by
    ``image_path`` with one column per attribute.  Asset types that
    don't apply to a given attribute get NaN — same as the truth CSV.
    """
    out = pd.DataFrame({"image_path": test_df["image_path"].values})

    for attr_col in schema.attribute_columns():
        attr = schema.attributes[attr_col]
        col = pd.Series([np.nan] * len(test_df), index=test_df.index, dtype=object)
        for asset_type_id in attr.asset_types:
            profile_name = schema.asset_types[asset_type_id].profile_name
            train_mask = train_df["profile_name"] == profile_name
            test_mask = test_df["profile_name"] == profile_name
            if not test_mask.any():
                continue
            try:
                preds = per_attribute_fn(
                    asset_type_id,
                    attr_col,
                    train_df.loc[train_mask],
                    test_df.loc[test_mask],
                )
            except ValueError:
                continue
            if preds is None:
                continue
            preds = pd.Series(preds, index=test_df.loc[test_mask].index)
            col.loc[preds.index] = preds.values
        out[attr_col] = col.values

    return out


def _coerce_predictions(predictions: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Coerce numeric/count attributes to floats so metrics work."""
    out = predictions.copy()
    for attr_col in schema.attribute_columns():
        if attr_col not in out.columns:
            continue
        kind = schema.kind_of(attr_col)
        if kind in {AttributeKind.NUMERIC, AttributeKind.COUNT}:
            out[attr_col] = pd.to_numeric(out[attr_col], errors="coerce")
    return out


def _truth_frame(test_df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    cols = ["image_path"] + [
        c for c in schema.attribute_columns() if c in test_df.columns
    ]
    return test_df[cols].copy()


def run_pipeline(
    *,
    pipeline: str,
    model_family: str,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predict_fn: PredictFn,
    weights_path: str | Path | None = None,
    schema: Schema | None = None,
    data_version: str | None = None,
    split_seed: int | None = None,
    params: dict[str, Any] | None = None,
    extra_tags: dict[str, Any] | None = None,
    log_to_mlflow: bool = True,
    predictions_dir: str | Path = "data/predictions",
    produce_train_predictions: bool = False,
    train_prediction_mode: str = "oof",
    train_n_folds: int = 5,
) -> PipelineResult:
    """One-call entry point for every pipeline.

    The caller provides a ``predict_fn`` that takes the project's
    train/test DataFrames and returns a per-image predictions
    DataFrame.  ``run_pipeline`` then:

    1. Coerces numeric columns to floats.
    2. Builds a :class:`PerAttributeReport` against the test labels.
    3. Writes the predictions CSV to ``predictions_dir`` (gitignored).
    4. Optionally also produces a ``<pipeline>__<model>__train.csv`` of
       train-set predictions (OOF via asset-grouped K-fold for trained
       pipelines, direct prediction for zero-shot ones) so the stacker
       can use them as training features.
    5. Logs the run + nested per-attribute runs to MLflow.

    Returns a :class:`PipelineResult` with the report, predictions and
    the parent MLflow run id.
    """
    schema = schema or load_schema()
    predictions = predict_fn(train_df, test_df, schema)
    predictions = _coerce_predictions(predictions, schema)

    if "image_path" not in predictions.columns:
        raise ValueError("predict_fn must return a DataFrame with an 'image_path' column.")

    truth = _truth_frame(test_df, schema)
    report = per_attribute_report(
        pipeline=pipeline,
        predictions=predictions,
        truth=truth,
        schema=schema,
        weights_path=weights_path,
    )

    out_dir = Path(predictions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{pipeline}__{model_name}.csv"
    predictions.to_csv(csv_path, index=False)

    train_csv_path: Path | None = None
    if produce_train_predictions:
        train_preds = run_train_side(
            predict_fn,
            train_df,
            schema,
            mode=train_prediction_mode,
            n_folds=train_n_folds,
        )
        train_preds = _coerce_predictions(train_preds, schema)
        train_csv_path = out_dir / f"{pipeline}__{model_name}__train.csv"
        train_preds.to_csv(train_csv_path, index=False)

    run_id: str | None = None
    if log_to_mlflow:
        setup_mlflow()
        run_id = log_pipeline_run(
            report=report,
            model_family=model_family,
            model_name=model_name,
            data_version=data_version,
            split_seed=split_seed,
            params={"predictions_csv": str(csv_path), **(params or {})},
            extra_tags=extra_tags,
        )

    extras: dict[str, Any] = {"predictions_csv": str(csv_path)}
    if train_csv_path is not None:
        extras["train_predictions_csv"] = str(train_csv_path)

    return PipelineResult(
        pipeline=pipeline,
        model_name=model_name,
        model_family=model_family,
        report=report,
        predictions=predictions,
        mlflow_run_id=run_id,
        extras=extras,
    )
