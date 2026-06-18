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
