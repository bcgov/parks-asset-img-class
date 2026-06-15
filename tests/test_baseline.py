"""Tests for grouped majority-class baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline import (  # noqa: E402
    ConstantPredictor,
    STRATEGIES,
    cross_validate_majority_class_frame,
    cross_validate_train_folder,
    first_mode,
    infer_target_column,
)


def test_constant_predictor_repeats_value() -> None:
    assert ConstantPredictor("Timber").predict(3) == ["Timber", "Timber", "Timber"]


def test_first_mode_drops_missing_values() -> None:
    values = pd.Series([None, "Beam", "Timber", "Timber"])
    assert first_mode(values) == "Timber"


def test_infer_target_column_matches_punctuation_variant() -> None:
    frame = pd.DataFrame(columns=["asset_id", "attr_material_frame,_tank,_body"])
    assert (
        infer_target_column(frame, "attr_material_frame_tank_body")
        == "attr_material_frame,_tank,_body"
    )


def test_grouped_cv_keeps_asset_ids_out_of_validation_fold() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "attr_bridge_type": [
                "Beam",
                "Beam",
                "Beam",
                "Beam",
                "Suspension",
                "Suspension",
                "Truss",
                "Truss",
            ],
        }
    )

    folds = cross_validate_majority_class_frame(
        frame, "attr_bridge_type", n_splits=2, random_state=7
    )

    assert len(folds) == 2 * len(STRATEGIES)
    assert set(folds["strategy"]) == set(STRATEGIES)
    per_strategy_assets = folds.groupby("strategy")["n_valid_assets"].sum()
    assert (per_strategy_assets == frame["asset_id"].nunique()).all()


def test_cross_validate_train_folder_reads_only_train_csvs(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    pd.DataFrame(
        {
            "asset_id": [1, 1, 2, 3, 4, 5],
            "attr_bridge_type": ["Beam", "Beam", "Beam", "Truss", "Beam", "Truss"],
        }
    ).to_csv(train_dir / "attr_bridge_type_train.csv", index=False)

    summary, folds = cross_validate_train_folder(
        train_dir,
        targets=["attr_bridge_type"],
        n_splits=2,
        random_state=11,
    )

    assert summary.loc[0, "attribute"] == "attr_bridge_type"
    assert summary.loc[0, "n_folds"] == 2
    assert len(folds) == 2 * len(STRATEGIES)
    assert len(summary) == len(STRATEGIES)
