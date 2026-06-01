"""Per-attribute and cross-attribute (un)weighted metrics.

Implements the partner ask (Henry, 2026-05): report **both** unweighted
and partner-weighted macro-F1 / MAE / RMSE, plus a per-attribute report
card so partners can see whether different pipelines win on different
attributes (Model A / Model B / Model C example from BC Parks).

The key entry point is :func:`per_attribute_report`, which a pipeline
calls once after producing predictions for every attribute it predicts,
and which returns a fully-populated ``PerAttributeReport`` ready to log
to MLflow.

Two distinct weighting layers are exposed:

1. **Within an attribute**: we always report ``macro_f1`` (class-balanced)
   because rare classes matter for the partner.  Standard
   ``weighted_f1`` is also reported alongside for context.
2. **Across attributes**: the leaderboard aggregates per-attribute
   macro-F1 (or RMSE / MAE) into a single number using the weights in
   ``configs/attribute_weights.yaml``.  Default weights are all-equal,
   which makes the weighted aggregate equal to the unweighted aggregate
   on day one and lets the partner change priorities without re-running
   models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.data.schema import AttributeKind, Schema, load_schema

DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "attribute_weights.yaml"
)


@dataclass
class AttributeScore:
    """One row in the per-attribute report card."""

    attribute: str
    kind: str
    n: int
    metrics: dict[str, float]
    extras: dict[str, Any] = field(default_factory=dict)

    def primary(self) -> float:
        """Return the single number used for the cross-attribute aggregate.

        macro_f1 for classification / boolean / ordinal_bin; RMSE for
        numeric and count (lower is better; the leaderboard inverts).
        """
        if self.kind in {
            AttributeKind.CATEGORICAL.value,
            AttributeKind.BOOLEAN.value,
            AttributeKind.ORDINAL_BIN.value,
        }:
            return self.metrics.get("macro_f1", float("nan"))
        return self.metrics.get("rmse", float("nan"))


@dataclass
class PerAttributeReport:
    """Result of evaluating a single pipeline against the test set."""

    scores: dict[str, AttributeScore]
    weights: dict[str, float]
    pipeline: str
    weights_name: str = "equal_weights"

    def attributes(self) -> list[str]:
        return list(self.scores.keys())

    def per_attribute_table(self) -> pd.DataFrame:
        rows = []
        for attr, score in self.scores.items():
            row: dict[str, Any] = {
                "attribute": attr,
                "kind": score.kind,
                "n": score.n,
                "weight": float(self.weights.get(attr, 1.0)),
                **score.metrics,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def aggregate(self) -> dict[str, float]:
        """Return both unweighted and weighted cross-attribute aggregates."""
        cls_attrs = [
            a
            for a, s in self.scores.items()
            if s.kind
            in {
                AttributeKind.CATEGORICAL.value,
                AttributeKind.BOOLEAN.value,
                AttributeKind.ORDINAL_BIN.value,
            }
        ]
        num_attrs = [
            a
            for a, s in self.scores.items()
            if s.kind in {AttributeKind.NUMERIC.value, AttributeKind.COUNT.value}
        ]

        cls_scores = {a: self.scores[a].metrics.get("macro_f1", np.nan) for a in cls_attrs}
        cls_weights = {a: self.weights.get(a, 1.0) for a in cls_attrs}
        num_rmse = {a: self.scores[a].metrics.get("rmse", np.nan) for a in num_attrs}
        num_mae = {a: self.scores[a].metrics.get("mae", np.nan) for a in num_attrs}
        num_weights = {a: self.weights.get(a, 1.0) for a in num_attrs}

        agg = {
            "unweighted_macro_f1_classification": _mean(list(cls_scores.values())),
            "weighted_macro_f1_classification": weighted_macro(cls_scores, cls_weights),
            "unweighted_macro_rmse_numeric": _mean(list(num_rmse.values())),
            "weighted_macro_rmse_numeric": weighted_macro(num_rmse, num_weights),
            "unweighted_macro_mae_numeric": _mean(list(num_mae.values())),
            "weighted_macro_mae_numeric": weighted_macro(num_mae, num_weights),
        }
        return {k: float(v) for k, v in agg.items() if not np.isnan(v)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "weights_name": self.weights_name,
            "weights": dict(self.weights),
            "aggregate": self.aggregate(),
            "scores": {a: asdict(s) for a, s in self.scores.items()},
        }


def _mean(values: list[float]) -> float:
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def weighted_macro(
    per_attr_scores: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """Cross-attribute weighted average. Skips NaN attributes.

    ``weighted = sum(w[a] * s[a]) / sum(w[a])`` over attributes where
    ``s[a]`` is finite and ``w[a] > 0``.
    """
    num = 0.0
    den = 0.0
    for attr, score in per_attr_scores.items():
        if np.isnan(score):
            continue
        w = float(weights.get(attr, 1.0))
        if w <= 0:
            continue
        num += w * float(score)
        den += w
    return float(num / den) if den > 0 else float("nan")


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: list[Any] | None = None,
) -> dict[str, float]:
    """Standard classification metrics (accuracy + both F1 averages)."""
    if labels is None:
        labels = sorted(set(map(str, list(y_true) + list(y_pred))))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            )
        ),
    }


def numeric_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Standard regression metrics (MAE / RMSE / R^2)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "r2": float(r2_score(yt, yp)),
    }


def classification_attribute_score(
    attribute: str,
    kind: AttributeKind,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> AttributeScore:
    """Build an ``AttributeScore`` row for a classification attribute."""
    labels = sorted(set(map(str, list(y_true) + list(y_pred))))
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    return AttributeScore(
        attribute=attribute,
        kind=kind.value,
        n=int(len(y_true)),
        metrics=classification_metrics(y_true, y_pred, labels=labels),
        extras={
            "labels": labels,
            "confusion_matrix": matrix,
            "classification_report": report,
        },
    )


def numeric_attribute_score(
    attribute: str,
    kind: AttributeKind,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> AttributeScore:
    """Build an ``AttributeScore`` row for a numeric or count attribute."""
    metrics = numeric_metrics(y_true, y_pred)
    return AttributeScore(
        attribute=attribute,
        kind=kind.value,
        n=int(np.isfinite(np.asarray(y_true, dtype=float)).sum()),
        metrics=metrics,
    )


def load_attribute_weights(
    path: str | Path | None = None,
) -> tuple[dict[str, float], str]:
    """Read the partner-weight YAML, return (weights, name)."""
    weight_path = Path(path) if path is not None else DEFAULT_WEIGHTS_PATH
    if not weight_path.exists():
        return {}, "missing"
    with weight_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    weights = {str(k): float(v) for k, v in (raw.get("weights") or {}).items()}
    name = str(raw.get("name", weight_path.stem))
    return weights, name


def per_attribute_report(
    *,
    pipeline: str,
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    schema: Schema | None = None,
    weights_path: str | Path | None = None,
    attributes: list[str] | None = None,
) -> PerAttributeReport:
    """Score every shared attribute and return a ``PerAttributeReport``.

    Parameters
    ----------
    pipeline
        Identifier for the row in the leaderboard (e.g. ``"clip_zeroshot"``,
        ``"dinov3_mlp"``, ``"qwen25_vl_72b"``).
    predictions, truth
        DataFrames keyed by ``image_path`` (or any common index) with
        one column per attribute.  Rows present in ``predictions`` but
        not in ``truth`` (or vice versa) are inner-joined.
    schema
        Loaded schema; if None, uses ``load_schema()``.
    weights_path
        Path to a partner-weight YAML; defaults to ``configs/attribute_weights.yaml``.
    attributes
        Subset of attribute columns to score.  Defaults to every shared
        attribute that exists in both frames.
    """
    schema = schema or load_schema()
    weights, weights_name = load_attribute_weights(weights_path)

    common_cols = [c for c in predictions.columns if c in truth.columns]
    if attributes is None:
        attributes = [c for c in schema.attribute_columns() if c in common_cols]
    else:
        missing = [c for c in attributes if c not in common_cols]
        if missing:
            raise KeyError(f"Predictions/truth missing attributes: {missing}")

    key_col = "image_path" if "image_path" in predictions.columns else None
    if key_col is not None:
        joined = predictions[[key_col, *attributes]].merge(
            truth[[key_col, *attributes]],
            on=key_col,
            suffixes=("_pred", "_true"),
            how="inner",
        )
    else:
        joined = predictions[attributes].join(truth[attributes], rsuffix="_true")
        joined.columns = [
            f"{c}_pred" if not c.endswith("_true") else c for c in joined.columns
        ]

    scores: dict[str, AttributeScore] = {}
    for col in attributes:
        attr = schema.attributes[col]
        pred_col = f"{col}_pred"
        true_col = f"{col}_true"
        mask = joined[true_col].notna() & joined[pred_col].notna()
        if mask.sum() == 0:
            continue
        y_true = joined.loc[mask, true_col]
        y_pred = joined.loc[mask, pred_col]
        if attr.kind in {
            AttributeKind.CATEGORICAL,
            AttributeKind.BOOLEAN,
            AttributeKind.ORDINAL_BIN,
        }:
            scores[col] = classification_attribute_score(
                attribute=col,
                kind=attr.kind,
                y_true=y_true.astype(str),
                y_pred=y_pred.astype(str),
            )
        else:
            scores[col] = numeric_attribute_score(
                attribute=col,
                kind=attr.kind,
                y_true=pd.to_numeric(y_true, errors="coerce"),
                y_pred=pd.to_numeric(y_pred, errors="coerce"),
            )

    return PerAttributeReport(
        scores=scores,
        weights=weights,
        weights_name=weights_name,
        pipeline=pipeline,
    )
