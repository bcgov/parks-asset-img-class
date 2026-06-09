"""Tests for the DINOv3 multi-attribute runner."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_dinov3_remaining_attributes import parse_args  # noqa: E402


def test_default_image_root_uses_clean_images(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_dinov3_remaining_attributes.py"])

    args = parse_args()

    assert args.image_root == Path("data/processed/images_clean")
