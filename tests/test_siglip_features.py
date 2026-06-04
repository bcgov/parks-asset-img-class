"""Tests for reusable SigLIP feature helpers."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.siglip_features import model_slug  # noqa: E402


def test_model_slug_is_filesystem_friendly() -> None:
    assert model_slug("google/siglip2-base-patch16-224") == "google_siglip2_base_patch16_224"
