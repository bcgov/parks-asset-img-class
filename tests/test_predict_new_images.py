"""Tests for new-image prediction helpers that do not load DINOv3."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.predict_new_images import limit_asset_rows  # noqa: E402


def test_limit_asset_rows_keeps_all_images_for_first_assets() -> None:
    rows = pd.DataFrame(
        {
            "asset_id": ["a", "a", "b", "c"],
            "image_path": ["a1.jpg", "a2.jpg", "b1.jpg", "c1.jpg"],
            "profile_name": ["Stairs", "Stairs", "Stairs", "Trail Bridge"],
        }
    )

    limited = limit_asset_rows(rows, 2)

    assert limited["asset_id"].tolist() == ["a", "a", "b"]


def test_limit_asset_rows_rejects_non_positive_limit() -> None:
    rows = pd.DataFrame({"asset_id": ["a"], "image_path": ["a.jpg"]})

    with pytest.raises(ValueError, match="positive integer"):
        limit_asset_rows(rows, 0)
