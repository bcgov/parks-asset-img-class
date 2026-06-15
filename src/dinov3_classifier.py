"""Classifiers trained on frozen DINOv3 asset embeddings."""

from __future__ import annotations

from pathlib import Path

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


def join_labels_and_features(
    labels: pd.DataFrame,
    asset_features: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Join a task CSV with asset-level DINOv3 features."""
    if "asset_id" not in labels.columns:
        raise ValueError("labels must contain an 'asset_id' column.")
    if "asset_id" not in asset_features.columns:
        raise ValueError("asset_features must contain an 'asset_id' column.")

    target_column = infer_target_column(labels, target)
    features = feature_columns(asset_features.columns)
    if not features:
        raise ValueError("asset_features does not contain f_* feature columns.")

    labelled = labels[["asset_id", target_column]].dropna(subset=[target_column])
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate DINOv3 embeddings with grouped cross-validation."""
    joined, target_column, features = join_labels_and_features(
        labels,
        asset_features,
        target,
    )
    if len(joined) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    n_asset_groups = joined[group_column].nunique()
    target_splits = min(n_splits, n_asset_groups)
    if target_splits < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(joined, y, groups), start=1):
        train_assets = set(joined.iloc[train_idx][group_column])
        valid_assets = set(joined.iloc[valid_idx][group_column])
        overlap = train_assets & valid_assets
        if overlap:
            raise AssertionError(
                f"Asset leakage in {target} fold {fold}: {sorted(overlap)[:5]}"
            )

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
        predictions = model.predict(X.iloc[valid_idx])
        y_valid = y.iloc[valid_idx]
        valid_rows = joined.iloc[valid_idx]
        if classifier == "logistic_regression_tuned":
            tuned_c = tuned_params.get("C")
            tuned_class_weight = tuned_params.get("class_weight")
            inner_macro_f1_mean = tuned_params.get("inner_macro_f1_mean")
        else:
            tuned_c = pd.NA
            tuned_class_weight = pd.NA
            inner_macro_f1_mean = pd.NA

        for row, true_label, predicted_label in zip(
            valid_rows.itertuples(index=False),
            y_valid,
            predictions,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "attribute": target,
                    "target_column": target_column,
                    "target_file": target_file,
                    "feature_file": feature_file,
                    "task_type": "classification",
                    "strategy": f"dinov3_frozen_embeddings_{classifier}",
                    "classifier": classifier,
                    "splitter": splitter_name,
                    "fold": fold,
                    "n_folds": target_splits,
                    "asset_id": getattr(row, group_column),
                    "y_true": true_label,
                    "y_pred": predicted_label,
                    "is_correct": true_label == predicted_label,
                    "tuned_C": tuned_c,
                    "tuned_class_weight": _format_class_weight(tuned_class_weight),
                    "inner_macro_f1_mean": inner_macro_f1_mean,
                }
            )

        fold_rows.append(
            {
                "attribute": target,
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

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = summarize_dinov3_folds(folds)
    return summary, folds, predictions


def summarize_dinov3_folds(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-fold DINOv3 classifier metrics."""
    if fold_results.empty:
        return pd.DataFrame()

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
        values.update(
            {
                "n_folds": int(group["fold"].max()),
                "n_labels": int(group["n_valid_labels"].sum()),
                "n_assets": int(group["n_valid_assets"].sum()),
                "n_features": int(group["n_features"].iloc[0]),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=0),
                "weighted_f1_mean": group["weighted_f1"].mean(),
                "weighted_f1_std": group["weighted_f1"].std(ddof=0),
                "macro_f1_mean": group["macro_f1"].mean(),
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
    )


__all__ = [
    "CLASSIFIER_CHOICES",
    "cross_validate_dinov3_classifier",
    "join_labels_and_features",
    "make_classifier",
    "run_task_from_files",
    "summarize_dinov3_folds",
]
