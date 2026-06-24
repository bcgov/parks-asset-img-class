"""Baselines for park asset attribute files.
 
The baseline intentionally learns only one thing from each training fold: the
most common label. Cross-validation is grouped by ``asset_id`` so images or
records from the same asset never appear in both the training and validation
parts of a fold.
 
Three strategies are supported:
- ``majority_class_group_cv``: always predict the most common training label.
- ``uniform_random_group_cv``: pick a label uniformly at random from observed classes.
- ``stratified_random_group_cv``: pick a label at random weighted by empirical
  class frequencies in the training fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


DEFAULT_CLASSIFICATION_TARGETS = [
    "attr_abutment_material",
    "attr_bridge_type",
    "attr_decking_material",
    "attr_has_edge_guard",
    "attr_has_pedestrian_railing",
    "attr_material_frame_tank_body",
    "attr_structure_material",
    "attr_structure_position",
    "length_bin",
    "width_bin",
    "fall_height_bin",
    "steps_bin",
]

DEFAULT_TRAIN_DIR = Path("data/processed/train")
STRATEGIES = (
    "majority_class_group_cv",
    "uniform_random_group_cv",
    "stratified_random_group_cv",
)


@dataclass(frozen=True)
class ConstantPredictor:
    """Predict the same fitted value for every row."""

    value: object

    def predict(self, n_rows: int) -> list[object]:
        """Return ``n_rows`` copies of the fitted constant."""
        return [self.value] * n_rows

@dataclass(frozen=True)
class UniformRandomPredictor:
    """Pick a label uniformly at random from the observed classes."""
    classes: list
    random_state: int = 42

    def predict(self, n_rows: int) -> list:
        """Return ``n_rows`` labels sampled uniformly from observed classes."""
        rng = np.random.default_rng(self.random_state)
        return rng.choice(self.classes, size=n_rows).tolist()


@dataclass(frozen=True)
class StratifiedRandomPredictor:
    """Pick a label at random weighted by empirical class frequencies."""
    classes: list
    probabilities: list
    random_state: int = 42

    def predict(self, n_rows: int) -> list:
        """Return ``n_rows`` labels sampled by observed class frequency."""
        rng = np.random.default_rng(self.random_state)
        return rng.choice(self.classes, size=n_rows, p=self.probabilities).tolist()


def first_mode(series: pd.Series) -> object:
    """Return the first modal value after dropping missing labels."""
    modes = series.dropna().mode()
    if modes.empty:
        raise ValueError("Cannot compute a mode for an empty target.")
    return modes.iloc[0]


def normalize_name(name: str) -> str:
    """Normalize column/file target names for loose matching."""
    return "".join(char for char in name.lower() if char.isalnum())


def target_from_train_path(path: str | Path) -> str:
    """Infer the target name from a ``*_train.csv`` path."""
    stem = Path(path).stem
    return stem.removesuffix("_train")


def infer_target_column(frame: pd.DataFrame, target: str) -> str:
    """Find the actual column for a target inferred from a file name.

    Most train files use the same target name in the file and column. One
    source column contains punctuation, though:
    ``attr_material_frame,_tank,_body``. This helper lets the cleaner file name
    ``attr_material_frame_tank_body`` still match it.
    """
    if target in frame.columns:
        return target

    normalized_target = normalize_name(target)
    matches = [
        column for column in frame.columns if normalize_name(column) == normalized_target
    ]
    if len(matches) == 1:
        return matches[0]

    raise ValueError(f"Could not infer a target column for {target!r}.")


def existing_targets(train_dir: str | Path, targets: Iterable[str]) -> list[str]:
    """Return requested targets with matching ``*_train.csv`` files."""
    train_path = Path(train_dir)
    return [target for target in targets if (train_path / f"{target}_train.csv").exists()]


def _make_group_splitter(
    labelled: pd.DataFrame,
    target_column: str,
    group_column: str,
    n_splits: int,
    random_state: int,
) -> tuple[object, str]:
    """Prefer stratified grouped folds when every class has enough assets."""
    class_group_counts = labelled.groupby(target_column)[group_column].nunique()
    if int(class_group_counts.min()) >= n_splits:
        return (
            StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            ),
            "StratifiedGroupKFold",
        )
    return (
        GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        "GroupKFold",
    )


def cross_validate_majority_class_frame(
    df: pd.DataFrame,
    target: str,
    *,
    target_file: str | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    group_column: str = "asset_id",
) -> pd.DataFrame:
    """Cross-validate one train-file data frame with grouped asset folds."""
    if group_column not in df.columns:
        raise ValueError(f"Missing required group column {group_column!r}.")

    target_column = infer_target_column(df, target)
    labelled = df.loc[df[target_column].notna(), [target_column, group_column]].copy()
    labelled = labelled.reset_index(drop=True)
    if len(labelled) < 2:
        return pd.DataFrame()

    n_asset_groups = labelled[group_column].nunique()
    target_splits = min(n_splits, n_asset_groups)
    if target_splits < 2:
        return pd.DataFrame()

    splitter, splitter_name = _make_group_splitter(
        labelled, target_column, group_column, target_splits, random_state
    )
    y = labelled[target_column]
    groups = labelled[group_column]

    rows: list[dict[str, object]] = []
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(labelled, y, groups), start=1
    ):
        train_fold = labelled.iloc[train_idx]
        valid_fold = labelled.iloc[valid_idx]
        train_assets = set(train_fold[group_column])
        valid_assets = set(valid_fold[group_column])
        overlap = train_assets & valid_assets
        if overlap:
            raise AssertionError(
                f"Asset leakage in {target} fold {fold}: {sorted(overlap)[:5]}"
            )

        y_train = train_fold[target_column]
        y_valid = valid_fold[target_column]

        class_counts = y_train.value_counts(dropna=True)
        majority_class = first_mode(y_train)
        majority_count = int(class_counts.loc[majority_class])
        train_label_count = int(class_counts.sum())
        classes = class_counts.index.tolist()
        probabilities = (class_counts / train_label_count).tolist()
 
        predictors = {
            "majority_class_group_cv": ConstantPredictor(majority_class),
            "uniform_random_group_cv": UniformRandomPredictor(
                classes, random_state=random_state
            ),
            "stratified_random_group_cv": StratifiedRandomPredictor(
                classes, probabilities, random_state=random_state
            ),
        }
 
        for strategy_name, predictor in predictors.items():
            y_pred = predictor.predict(len(y_valid))
            rows.append(
                {
                    "attribute": target,
                    "target_column": target_column,
                    "target_file": target_file,
                    "task_type": "classification",
                    "strategy": strategy_name,
                    "splitter": splitter_name,
                    "fold": fold,
                    "n_folds": target_splits,
                    "prediction": majority_class if strategy_name == "majority_class_group_cv" else None,
                    "n_train_labels": train_label_count,
                    "n_valid_labels": int(len(y_valid)),
                    "n_train_assets": len(train_assets),
                    "n_valid_assets": len(valid_assets),
                    "train_majority_count": majority_count,
                    "train_majority_share": majority_count / train_label_count,
                    "accuracy": accuracy_score(y_valid, y_pred),
                    "weighted_f1": f1_score(
                        y_valid, y_pred, average="weighted", zero_division=0
                    ),
                    "macro_f1": f1_score(
                        y_valid, y_pred, average="macro", zero_division=0
                    ),
                }
            )
 
    return pd.DataFrame(rows)
 
 
def summarize_cv_folds(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-fold metrics across all strategies."""
    if fold_results.empty:
        return pd.DataFrame()
 
    summary_rows: list[dict[str, object]] = []
    group_columns = [
        "attribute",
        "target_column",
        "target_file",
        "task_type",
        "strategy",
        "splitter",
    ]
    for keys, group in fold_results.groupby(group_columns, dropna=False):
        values = dict(zip(group_columns, keys, strict=True))
        values.update(
            {
                "n_folds": int(group["fold"].max()),
                "n_labels": int(group["n_valid_labels"].sum()),
                "n_assets": int(group["n_valid_assets"].sum()),
                "prediction": first_mode(group["prediction"].dropna()) if group["prediction"].notna().any() else None,
                "train_majority_share_mean": group["train_majority_share"].mean(),
                "train_majority_share_std": group["train_majority_share"].std(ddof=0),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=0),
                "weighted_f1_mean": group["weighted_f1"].mean(),
                "weighted_f1_std": group["weighted_f1"].std(ddof=0),
                "macro_f1_mean": group["macro_f1"].mean(),
                "macro_f1_std": group["macro_f1"].std(ddof=0),
            }
        )
        summary_rows.append(values)
 
    return pd.DataFrame(summary_rows).sort_values(["attribute", "strategy"]).reset_index(drop=True)
 
 
def cross_validate_train_folder(
    train_dir: str | Path = DEFAULT_TRAIN_DIR,
    targets: Iterable[str] = DEFAULT_CLASSIFICATION_TARGETS,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    group_column: str = "asset_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all three baseline strategies for train CSVs in ``train_dir``."""
    train_path = Path(train_dir)
    fold_tables: list[pd.DataFrame] = []
 
    for target in existing_targets(train_path, targets):
        csv_path = train_path / f"{target}_train.csv"
        df = pd.read_csv(csv_path)
        fold_table = cross_validate_majority_class_frame(
            df,
            target,
            target_file=str(csv_path),
            n_splits=n_splits,
            random_state=random_state,
            group_column=group_column,
        )
        if not fold_table.empty:
            fold_tables.append(fold_table)
 
    if not fold_tables:
        return pd.DataFrame(), pd.DataFrame()
 
    fold_results = pd.concat(fold_tables, ignore_index=True)
    summary = summarize_cv_folds(fold_results)
    return summary, fold_results


def evaluate_all_baselines(
    train_dir: str | Path = DEFAULT_TRAIN_DIR,
    classification_targets: Iterable[str] = DEFAULT_CLASSIFICATION_TARGETS,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return the grouped cross-validated majority-class baseline summary."""
    summary, _ = cross_validate_train_folder(
        train_dir,
        targets=classification_targets,
        n_splits=n_splits,
        random_state=random_state,
    )
    return summary


__all__ = [
    "ConstantPredictor",
    "DEFAULT_CLASSIFICATION_TARGETS",
    "DEFAULT_TRAIN_DIR",
    "cross_validate_majority_class_frame",
    "cross_validate_train_folder",
    "evaluate_all_baselines",
    "existing_targets",
    "first_mode",
    "infer_target_column",
    "normalize_name",
    "summarize_cv_folds",
    "target_from_train_path",
]
