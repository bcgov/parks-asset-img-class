"""Tests for clear DINOv3 feature-pipeline failure messages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import build_asset_features_from_image_features as build_assets  # noqa: E402
from scripts import extract_dinov3_features  # noqa: E402


def test_read_features_rejects_empty_csv(tmp_path: Path) -> None:
    empty_features = tmp_path / "dinov3_vitb16_master_images.csv"
    empty_features.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="make clean-dinov3"):
        build_assets.read_features(empty_features)


def test_extract_dinov3_does_not_write_empty_feature_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_csv = tmp_path / "master_dataset.csv"
    output_csv = tmp_path / "features.csv"
    asset_output_csv = tmp_path / "asset_features.csv"
    skipped_csv = tmp_path / "skipped.csv"
    input_csv.write_text("asset_id,image_path\n1,missing.jpg\n", encoding="utf-8")

    def fake_extract_image_features(*args: object, **kwargs: object) -> tuple[pd.DataFrame, pd.DataFrame]:
        skipped = pd.DataFrame(
            {
                "asset_id": [1],
                "image_path": ["missing.jpg"],
                "resolved_path": ["missing.jpg"],
                "reason": ["missing_file"],
            }
        )
        return pd.DataFrame(), skipped

    monkeypatch.setattr(
        extract_dinov3_features,
        "extract_image_features",
        fake_extract_image_features,
    )

    args = argparse.Namespace(
        limit_assets=None,
        image_root=tmp_path / "images_clean",
        model="dinov3_vitb16",
        model_source="facebookresearch/dinov3",
        weights=None,
        device=None,
        image_size=224,
    )

    with pytest.raises(SystemExit, match="No image features extracted"):
        extract_dinov3_features.process_one(
            input_csv,
            output_csv,
            asset_output_csv,
            skipped_csv,
            args,
        )

    assert not output_csv.exists()
    assert not asset_output_csv.exists()
    assert skipped_csv.exists()
