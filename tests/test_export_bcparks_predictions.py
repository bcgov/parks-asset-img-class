"""Tests for partner-facing prediction export helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_bcparks_predictions import (  # noqa: E402
    _applicable_profiles_for_target,
    _filter_applicable_assets,
    load_applicability,
)
from src.final_model_artifacts import (  # noqa: E402
    load_final_model_bundle,
    predict_with_final_model_bundle,
    save_final_model_bundle,
    train_final_model_bundle,
)


def test_load_applicability_maps_targets_to_asset_types(tmp_path: Path) -> None:
    matrix_path = tmp_path / "attribute_applicability.csv"
    pd.DataFrame(
        {
            "Attribute": ["attr_bridge_type", "steps_bin", "load_capacity_kg"],
            "Want AI to Determine": ["yes", "yes", "no"],
            "Trail Bridge": ["X", "", "X"],
            "Stairs": ["", "X", ""],
        }
    ).to_csv(matrix_path, index=False)

    applicability = load_applicability(matrix_path)

    assert applicability == {
        "Trail Bridge": {"attr_bridge_type"},
        "Stairs": {"steps_bin"},
    }
    assert _applicable_profiles_for_target(applicability, "steps_bin") == ["Stairs"]


def test_filter_applicable_assets_uses_explicit_profiles() -> None:
    assets = pd.DataFrame(
        {
            "asset_id": [1, 2, 3],
            "profile_name": ["Trail Bridge", "Stairs", "Viewing Platform"],
        }
    )
    labels = pd.DataFrame(
        {
            "asset_id": [1, 2, 3],
            "profile_name": ["Trail Bridge", "Stairs", "Viewing Platform"],
        }
    )

    filtered, profiles = _filter_applicable_assets(
        assets,
        labels,
        predict_all_assets=False,
        explicit_profiles=["Trail Bridge"],
    )

    assert filtered["asset_id"].tolist() == [1]
    assert profiles == "Trail Bridge"


def test_filter_applicable_assets_can_still_predict_all_assets() -> None:
    assets = pd.DataFrame(
        {
            "asset_id": [1, 2],
            "profile_name": ["Trail Bridge", "Stairs"],
        }
    )
    labels = pd.DataFrame({"asset_id": [1], "profile_name": ["Trail Bridge"]})

    filtered, profiles = _filter_applicable_assets(
        assets,
        labels,
        predict_all_assets=True,
        explicit_profiles=["Trail Bridge"],
    )

    assert filtered["asset_id"].tolist() == [1, 2]
    assert profiles == "all_profiles"


def test_filter_applicable_assets_with_empty_explicit_profiles_returns_no_assets() -> None:
    assets = pd.DataFrame(
        {
            "asset_id": [1, 2],
            "profile_name": ["Trail Bridge", "Stairs"],
        }
    )
    labels = pd.DataFrame({"asset_id": [1], "profile_name": ["Trail Bridge"]})

    filtered, profiles = _filter_applicable_assets(
        assets,
        labels,
        predict_all_assets=False,
        explicit_profiles=[],
    )

    assert filtered.empty
    assert profiles == "no_applicable_profiles"


def test_saved_final_model_bundle_can_be_loaded_and_used_for_prediction(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    labels = pd.DataFrame(
        {
            "asset_id": [1, 2, 3, 4],
            "profile_name": ["Trail Bridge"] * 4,
            "attr_bridge_type": ["Beam", "Beam", "Suspension", "Suspension"],
        }
    )
    labels.to_csv(train_dir / "attr_bridge_type_train.csv", index=False)

    asset_features = pd.DataFrame(
        {
            "asset_id": [1, 2, 3, 4],
            "f_0000": [0.0, 0.1, 1.0, 1.1],
            "f_0001": [0.0, 0.1, 1.0, 1.1],
        }
    )
    applicability = {"Trail Bridge": {"attr_bridge_type"}}
    bundle = train_final_model_bundle(
        train_dir=train_dir,
        asset_features=asset_features,
        targets=["attr_bridge_type"],
        classifier="logistic_regression",
        model_family="dinov3",
        model_name="dinov3_vitb16",
        random_state=42,
        applicability=applicability,
    )

    model_dir = tmp_path / "models" / "final"
    save_final_model_bundle(bundle, model_dir)
    loaded = load_final_model_bundle(model_dir)
    predictions = predict_with_final_model_bundle(
        bundle=loaded,
        asset_features=asset_features,
        asset_metadata=pd.DataFrame(
            {
                "asset_id": [1, 2, 3, 4],
                "profile_name": ["Trail Bridge"] * 4,
                "image_count": [1, 1, 1, 1],
            }
        ),
    )

    assert predictions["attribute"].unique().tolist() == ["attr_bridge_type"]
    assert predictions["predicted_value"].notna().all()
    assert predictions["model_name"].unique().tolist() == ["dinov3_vitb16"]
