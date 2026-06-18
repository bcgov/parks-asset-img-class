"""Classifiers trained on frozen DINOv3 asset embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.baseline import first_mode, infer_target_column
from src.dinov3_features import feature_columns


CLASSIFIER_CHOICES = (
    "logistic_regression",
    "logistic_regression_tuned",
    "linear_svm",
    "random_forest",
    "hist_gradient_boosting",
)

# Column used to identify an asset's type (for per-asset-type training).
ASSET_TYPE_COLUMN = "profile_name"

LOGISTIC_TUNING_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "class_weight": ["balanced", None],
}


def make_classifier(
    classifier: str = "logistic_regression",
    random_state: int = 42,
    *,
    logistic_c: float = 1.0,
    logistic_class_weight: str | None = "balanced",
) -> object:
    """Return a small classifier for frozen DINOv3 embeddings."""
    if classifier in {"logistic_regression", "logistic_regression_tuned"}:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=logistic_c,
                max_iter=2000,
                class_weight=logistic_class_weight,
                random_state=random_state,
            ),
        )

    if classifier == "linear_svm":
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="hinge",
                alpha=0.0001,
                class_weight="balanced",
                random_state=random_state,
                max_iter=2000,
                tol=1e-3,
            ),
        )

    if classifier == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )

    if classifier == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            class_weight="balanced",
            random_state=random_state,
        )

    raise ValueError(
        f"Unknown classifier {classifier!r}. Expected one of {CLASSIFIER_CHOICES}."
    )


def _format_class_weight(class_weight: str | None) -> str:
    return "none" if class_weight is None else class_weight


def _predict_with_confidence(model: object, X: pd.DataFrame):
    """Return (predictions, confidence) where confidence is the max class
    probability for each row. Classifiers without predict_proba get NaN.
    """
    predictions = model.predict(X)
    confidences = [float("nan")] * len(predictions)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            confidences = proba.max(axis=1).tolist()
        except Exception:
            pass
    return predictions, confidences


def join_labels_and_features(
    labels: pd.DataFrame,
    asset_features: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Join a task CSV with asset-level DINOv3 features.

    Carries the asset-type column (``profile_name``) through when present so
    callers can train one model per asset type.
    """
    if "asset_id" not in labels.columns:
        raise ValueError("labels must contain an 'asset_id' column.")
    if "asset_id" not in asset_features.columns:
        raise ValueError("asset_features must contain an 'asset_id' column.")

    target_column = infer_target_column(labels, target)
    features = feature_columns(asset_features.columns)
    if not features:
        raise ValueError("asset_features does not contain f_* feature columns.")

    keep_cols = ["asset_id", target_column]
    if ASSET_TYPE_COLUMN in labels.columns:
        keep_cols.append(ASSET_TYPE_COLUMN)

    labelled = labels[keep_cols].dropna(subset=[target_column])
    labelled = labelled.drop_duplicates(["asset_id", target_column])
    joined = labelled.merge(asset_features[["asset_id", *features]], on="asset_id", how="inner")
    joined = joined.drop_duplicates("asset_id").reset_index(drop=True)
    return joined, target_column, features


def _make_group_splitter(
    labelled: pd.DataFrame,
    target_column: str,
    group_column: str,
    n_splits: int,
    random_state: int,
) -> tuple[object, str]:
    class_group_counts = labelled.groupby(target_column)[group_column].nunique()
    if int(class_group_counts.min()) >= n_splits:
        return (
            StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            ),
            "StratifiedGroupKFold",
        )
    return (
        GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        "GroupKFold",
    )


def _tune_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    random_state: int,
) -> dict[str, object]:
    """Select logistic regression hyperparameters using grouped inner CV."""
    n_groups = groups.nunique()
    inner_splits = min(3, n_groups)
    default_params: dict[str, object] = {
        "C": 1.0,
        "class_weight": "balanced",
        "inner_macro_f1_mean": pd.NA,
    }
    if inner_splits < 2 or y.nunique() < 2:
        return default_params

    labelled = pd.DataFrame({"target": y, "asset_id": groups})
    splitter, _ = _make_group_splitter(
        labelled,
        "target",
        "asset_id",
        inner_splits,
        random_state,
    )

    candidate_scores: list[dict[str, object]] = []
    for c_value in LOGISTIC_TUNING_GRID["C"]:
        for class_weight in LOGISTIC_TUNING_GRID["class_weight"]:
            scores: list[float] = []
            for train_idx, valid_idx in splitter.split(X, y, groups):
                y_train = y.iloc[train_idx]
                y_valid = y.iloc[valid_idx]
                if y_train.nunique() < 2 or y_valid.nunique() < 2:
                    continue

                model = make_classifier(
                    classifier="logistic_regression",
                    random_state=random_state,
                    logistic_c=float(c_value),
                    logistic_class_weight=class_weight,
                )
                model.fit(X.iloc[train_idx], y_train)
                predictions = model.predict(X.iloc[valid_idx])
                scores.append(
                    f1_score(y_valid, predictions, average="macro", zero_division=0)
                )

            if scores:
                candidate_scores.append(
                    {
                        "C": float(c_value),
                        "class_weight": class_weight,
                        "inner_macro_f1_mean": float(sum(scores) / len(scores)),
                    }
                )

    if not candidate_scores:
        return default_params

    return max(
        candidate_scores,
        key=lambda row: (
            float(row["inner_macro_f1_mean"]),
            -abs(float(row["C"]) - 1.0),
            row["class_weight"] == "balanced",
        ),
    )


def _run_cv_on_subset(
    joined: pd.DataFrame,
    features: list[str],
    target: str,
    target_column: str,
    *,
    asset_type: str | None,
    target_file: str | None,
    feature_file: str | None,
    n_splits: int,
    random_state: int,
    group_column: str,
    classifier: str,
) -> tuple[list[dict], list[dict]]:
    """Run grouped CV on one subset of assets (optionally a single asset type).

    Returns (fold_rows, prediction_rows). Each asset produces exactly ONE
    prediction row with clean column names plus confidence and asset_type.
    """
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    n_asset_groups = joined[group_column].nunique()
    target_splits = min(n_splits, n_asset_groups)
    if len(joined) < 2 or target_splits < 2 or joined[target_column].nunique() < 2:
        # Not enough data to cross-validate this subset.
        return fold_rows, prediction_rows

    splitter, splitter_name = _make_group_splitter(
        joined,
        target_column,
        group_column,
        target_splits,
        random_state,
    )

    X = joined[features]
    y = joined[target_column]
    groups = joined[group_column]

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(joined, y, groups), start=1):
        train_assets = set(joined.iloc[train_idx][group_column])
        valid_assets = set(joined.iloc[valid_idx][group_column])
        overlap = train_assets & valid_assets
        if overlap:
            raise AssertionError(
                f"Asset leakage in {target} fold {fold}: {sorted(overlap)[:5]}"
            )

        # Skip folds whose TRAINING split has fewer than 2 classes — a
        # classifier can't be fit on a single class. This can happen when a
        # rare class has so few assets that they all land in one fold's
        # validation set, leaving the training set single-class.
        if y.iloc[train_idx].nunique() < 2:
            print(
                f"  [skip fold] {target}"
                + (f" / {asset_type}" if asset_type else "")
                + f" fold {fold}: training split has only one class "
                f"({y.iloc[train_idx].unique().tolist()}); skipping this fold."
            )
            continue

        tuned_params: dict[str, object] = {}
        if classifier == "logistic_regression_tuned":
            tuned_params = _tune_logistic_regression(
                X.iloc[train_idx],
                y.iloc[train_idx],
                groups.iloc[train_idx],
                random_state=random_state,
            )

        model = make_classifier(
            classifier=classifier,
            random_state=random_state,
            logistic_c=float(tuned_params.get("C", 1.0)),
            logistic_class_weight=tuned_params.get("class_weight", "balanced"),
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions, confidences = _predict_with_confidence(model, X.iloc[valid_idx])
        y_valid = y.iloc[valid_idx]
        valid_asset_ids = joined.iloc[valid_idx][group_column].tolist()

        if classifier == "logistic_regression_tuned":
            tuned_c = tuned_params.get("C")
            tuned_class_weight = tuned_params.get("class_weight")
            inner_macro_f1_mean = tuned_params.get("inner_macro_f1_mean")
        else:
            tuned_c = pd.NA
            tuned_class_weight = pd.NA
            inner_macro_f1_mean = pd.NA

        # ONE clean row per asset (no duplicate long-format row).
        for asset_id, true_label, pred_label, conf in zip(
            valid_asset_ids, y_valid, predictions, confidences, strict=True
        ):
            prediction_rows.append(
                {
                    "attribute": target,
                    "asset_type": asset_type,
                    "fold": fold,
                    "asset_id": asset_id,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "correct": true_label == pred_label,
                    "confidence": conf,
                    "target_column": target_column,
                    "classifier": classifier,
                    "strategy": f"dinov3_frozen_embeddings_{classifier}",
                    "tuned_C": tuned_c,
                    "tuned_class_weight": _format_class_weight(tuned_class_weight),
                    "inner_macro_f1_mean": inner_macro_f1_mean,
                }
            )

        fold_rows.append(
            {
                "attribute": target,
                "asset_type": asset_type,
                "target_column": target_column,
                "target_file": target_file,
                "feature_file": feature_file,
                "task_type": "classification",
                "strategy": f"dinov3_frozen_embeddings_{classifier}",
                "classifier": classifier,
                "splitter": splitter_name,
                "fold": fold,
                "n_folds": target_splits,
                "n_features": len(features),
                "n_train_labels": int(len(train_idx)),
                "n_valid_labels": int(len(valid_idx)),
                "n_train_assets": len(train_assets),
                "n_valid_assets": len(valid_assets),
                "accuracy": accuracy_score(y_valid, predictions),
                "weighted_f1": f1_score(y_valid, predictions, average="weighted", zero_division=0),
                "macro_f1": f1_score(y_valid, predictions, average="macro", zero_division=0),
                "tuned_C": tuned_c,
                "tuned_class_weight": _format_class_weight(tuned_class_weight),
                "inner_macro_f1_mean": inner_macro_f1_mean,
            }
        )

    return fold_rows, prediction_rows


def cross_validate_dinov3_classifier(
    labels: pd.DataFrame,
    asset_features: pd.DataFrame,
    target: str,
    *,
    target_file: str | None = None,
    feature_file: str | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    group_column: str = "asset_id",
    classifier: str = "logistic_regression",
    per_asset_type: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate DINOv3 embeddings with grouped cross-validation.

    When ``per_asset_type`` is True, a separate model is trained and evaluated
    for each asset type (``profile_name``), so binned numeric attributes whose
    bin ranges differ by asset type are not mixed into one incoherent label
    space. Predictions from all asset types are concatenated; the summary is a
    single combined row per attribute computed over the pooled folds.
    """
    joined, target_column, features = join_labels_and_features(
        labels,
        asset_features,
        target,
    )
    if len(joined) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_fold_rows: list[dict[str, object]] = []
    all_prediction_rows: list[dict[str, object]] = []

    if per_asset_type:
        if ASSET_TYPE_COLUMN not in joined.columns:
            raise ValueError(
                f"per_asset_type=True requires a '{ASSET_TYPE_COLUMN}' column in the "
                f"labels CSV, but it was not found. Columns: {labels.columns.tolist()}"
            )
        asset_types = sorted(a for a in joined[ASSET_TYPE_COLUMN].dropna().unique())
        for atype in asset_types:
            subset = joined[joined[ASSET_TYPE_COLUMN] == atype].reset_index(drop=True)
            fold_rows, pred_rows = _run_cv_on_subset(
                subset,
                features,
                target,
                target_column,
                asset_type=atype,
                target_file=target_file,
                feature_file=feature_file,
                n_splits=n_splits,
                random_state=random_state,
                group_column=group_column,
                classifier=classifier,
            )
            if not pred_rows:
                n_assets = subset[group_column].nunique()
                n_classes = subset[target_column].nunique()
                print(
                    f"  [skip] asset type '{atype}' for {target}: not enough data "
                    f"(assets={n_assets}, classes={n_classes}, need >=2 of each and "
                    f">=2 assets per class for {n_splits}-fold CV)."
                )
                continue
            all_fold_rows.extend(fold_rows)
            all_prediction_rows.extend(pred_rows)
    else:
        fold_rows, pred_rows = _run_cv_on_subset(
            joined,
            features,
            target,
            target_column,
            asset_type=None,
            target_file=target_file,
            feature_file=feature_file,
            n_splits=n_splits,
            random_state=random_state,
            group_column=group_column,
            classifier=classifier,
        )
        all_fold_rows.extend(fold_rows)
        all_prediction_rows.extend(pred_rows)

    if not all_prediction_rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    folds = pd.DataFrame(all_fold_rows)
    predictions = pd.DataFrame(all_prediction_rows)
    summary = summarize_dinov3_folds(folds, predictions=predictions,
                                     per_asset_type=per_asset_type)
    return summary, folds, predictions


def summarize_dinov3_folds(
    fold_results: pd.DataFrame,
    *,
    predictions: pd.DataFrame | None = None,
    per_asset_type: bool = False,
) -> pd.DataFrame:
    """Summarize DINOv3 classifier metrics into one row per attribute.

    The headline F1 metrics (weighted_f1_mean, macro_f1_mean, accuracy_mean) are
    computed by POOLING all out-of-fold predictions for the attribute and scoring
    them once, rather than averaging per-fold scores.

    Why pooling, not fold-averaging: with grouped CV — especially per-asset-type
    training — some folds (or tiny asset-type subsets) contain only a handful of
    assets. A 1-2 asset fold scores 0.0 or 1.0 with no middle ground, and
    averaging those degenerate fold scores equally with large folds badly
    distorts the result. Pooling all out-of-fold predictions and scoring once
    weights each asset equally and gives the honest aggregate. Every asset has
    exactly one out-of-fold prediction, so pooling is well-defined.

    The *_std columns are still the spread across folds (a dispersion measure),
    computed from fold_results.
    """
    if fold_results.empty:
        return pd.DataFrame()

    # Metadata grouping keys — one summary row per attribute.
    group_columns = [
        "attribute",
        "target_column",
        "target_file",
        "feature_file",
        "task_type",
        "strategy",
        "classifier",
        "splitter",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in fold_results.groupby(group_columns, dropna=False):
        values = dict(zip(group_columns, keys, strict=True))
        attribute = values["attribute"]

        # --- Pooled F1 from out-of-fold predictions (the honest aggregate) ---
        pooled_weighted_f1 = float("nan")
        pooled_macro_f1 = float("nan")
        pooled_accuracy = float("nan")
        if predictions is not None and not predictions.empty:
            pred_sub = predictions[predictions["attribute"] == attribute].dropna(
                subset=["true_label", "predicted_label"]
            )
            if not pred_sub.empty:
                y_true = pred_sub["true_label"]
                y_pred = pred_sub["predicted_label"]
                pooled_weighted_f1 = f1_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )
                pooled_macro_f1 = f1_score(
                    y_true, y_pred, average="macro", zero_division=0
                )
                pooled_accuracy = accuracy_score(y_true, y_pred)

        values.update(
            {
                "n_folds": int(group["fold"].max()),
                "n_labels": int(group["n_valid_labels"].sum()),
                "n_assets": int(group["n_valid_assets"].sum()),
                "n_features": int(group["n_features"].iloc[0]),
                # Headline metrics: pooled over all out-of-fold predictions.
                "accuracy_mean": pooled_accuracy,
                "accuracy_std": group["accuracy"].std(ddof=0),
                "weighted_f1_mean": pooled_weighted_f1,
                "weighted_f1_std": group["weighted_f1"].std(ddof=0),
                "macro_f1_mean": pooled_macro_f1,
                "macro_f1_std": group["macro_f1"].std(ddof=0),
            }
        )
        rows.append(values)

    return pd.DataFrame(rows).sort_values("attribute").reset_index(drop=True)


def run_task_from_files(
    *,
    labels_path: str | Path,
    features_path: str | Path,
    target: str,
    n_splits: int = 5,
    random_state: int = 42,
    classifier: str = "logistic_regression",
    per_asset_type: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read files and run one DINOv3 classification task."""
    labels = pd.read_csv(labels_path)
    features = pd.read_csv(features_path)
    return cross_validate_dinov3_classifier(
        labels,
        features,
        target,
        target_file=str(labels_path),
        feature_file=str(features_path),
        n_splits=n_splits,
        random_state=random_state,
        classifier=classifier,
        per_asset_type=per_asset_type,
    )


__all__ = [
    "CLASSIFIER_CHOICES",
    "cross_validate_dinov3_classifier",
    "join_labels_and_features",
    "make_classifier",
    "run_task_from_files",
    "summarize_dinov3_folds",
]
