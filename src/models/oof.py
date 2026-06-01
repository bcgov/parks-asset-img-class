"""Out-of-fold prediction helpers for stacking.

The CatBoost meta-learner (:mod:`src.models.stacking`) needs base-pipeline
predictions on the **training** set to learn how to combine them.  For a
trained pipeline (DINOv2 + head, anything with state) we cannot just
predict on train because that leaks the labels the head was fit on; we
have to use **out-of-fold** predictions produced by asset-grouped
K-fold.  For a zero-shot pipeline (CLIP, VLM, YOLO, baseline-as-zero-shot)
there is no training, so calling the same ``predict_fn`` on train rows
once is sufficient and ~K-times cheaper.

This module exposes two entry points used by every
``scripts/run_*.py --predict-train`` runner:

- :func:`oof_predict` runs asset-grouped K-fold and concatenates the
  per-fold predictions into a single train-aligned DataFrame.
- :func:`direct_predict` just calls ``predict_fn(train_df, train_df, ...)``
  on the full set (zero-shot path, no leakage).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from src.data.schema import Schema
from src.data.splits import asset_grouped_kfold

PredictFn = Callable[[pd.DataFrame, pd.DataFrame, Schema], pd.DataFrame]


def oof_predict(
    predict_fn: PredictFn,
    train_df: pd.DataFrame,
    schema: Schema,
    *,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Run asset-grouped K-fold to produce OOF predictions on ``train_df``.

    For each fold, ``predict_fn`` is called with the in-fold rows as
    the train side and the out-of-fold rows as the test side.  The
    returned per-fold prediction frames are concatenated into one frame
    keyed by ``image_path`` covering every row in ``train_df``.
    """
    pieces: list[pd.DataFrame] = []
    for train_idx, val_idx in asset_grouped_kfold(train_df, n_splits=n_folds):
        fold_train = train_df.loc[train_idx].reset_index(drop=True)
        fold_val = train_df.loc[val_idx].reset_index(drop=True)
        if fold_val.empty:
            continue
        preds = predict_fn(fold_train, fold_val, schema)
        pieces.append(preds)
    if not pieces:
        return pd.DataFrame({"image_path": []})
    return pd.concat(pieces, ignore_index=True)


def direct_predict(
    predict_fn: PredictFn,
    train_df: pd.DataFrame,
    schema: Schema,
) -> pd.DataFrame:
    """Predict on the full train set in one shot (zero-shot pipelines).

    Some pipelines (CLIP zero-shot, VLM, YOLO/Grounding-DINO,
    majority/median baseline) do not actually train on the training
    rows.  For them OOF and direct prediction give the same result
    but direct is K× cheaper, so the caller chooses.
    """
    return predict_fn(train_df, train_df, schema)


def run_train_side(
    predict_fn: PredictFn,
    train_df: pd.DataFrame,
    schema: Schema,
    *,
    mode: str = "oof",
    n_folds: int = 5,
) -> pd.DataFrame:
    """Convenience wrapper. ``mode`` is ``"oof"`` or ``"direct"``."""
    if mode == "oof":
        return oof_predict(predict_fn, train_df, schema, n_folds=n_folds)
    if mode == "direct":
        return direct_predict(predict_fn, train_df, schema)
    raise ValueError(f"Unknown train-side mode: {mode!r}")
