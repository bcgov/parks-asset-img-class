"""Tests for reusable DINOv3 feature helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dinov3_features import aggregate_asset_features, feature_columns, resolve_image_path  # noqa: E402


def test_feature_columns_sort_numerically() -> None:
    assert feature_columns(["asset_id", "f_0010", "f_0002", "f_0001"]) == [
        "f_0001",
        "f_0002",
        "f_0010",
    ]


def test_aggregate_asset_features_averages_images() -> None:
    features = pd.DataFrame(
        {
            "asset_id": [1, 1, 2],
            "image_path": ["a.jpg", "b.jpg", "c.jpg"],
            "f_0000": [1.0, 3.0, 10.0],
            "f_0001": [2.0, 4.0, 20.0],
        }
    )

    aggregated = aggregate_asset_features(features)

    assert aggregated.to_dict("records") == [
        {"asset_id": 1, "f_0000": 2.0, "f_0001": 3.0},
        {"asset_id": 2, "f_0000": 10.0, "f_0001": 20.0},
    ]


def test_resolve_image_path_prefers_clean_image_root(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "citywide" / "images" / "1.jpg"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    image = tmp_path / "data" / "processed" / "images_clean" / "citywide" / "images" / "1.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"clean")

    resolved = resolve_image_path(
        "data/citywide/images/1.jpg",
        image_root="data/processed/images_clean",
        repo_root=tmp_path,
    )

    assert resolved == image
