"""Tests for the SigLIP multi-attribute runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_dinov3_remaining_attributes import DEFAULT_TARGETS  # noqa: E402
from scripts.run_siglip_attributes import parse_args, selected_targets, target_set_slug  # noqa: E402


def test_target_set_slug_names_explicit_targets() -> None:
    args = argparse.Namespace(targets=["attr_decking_material"])

    assert target_set_slug(args, ["attr_decking_material"]) == "attr_decking_material"


def test_selected_targets_defaults_to_all_attributes() -> None:
    args = argparse.Namespace(targets=None)

    assert selected_targets(args) == DEFAULT_TARGETS
    assert "attr_decking_material" in selected_targets(args)


def test_target_set_slug_names_all_attributes_by_default() -> None:
    args = argparse.Namespace(targets=None)

    assert target_set_slug(args, ["attr_a", "attr_b"]) == "all_attributes"


def test_include_decking_flag_is_backward_compatible(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_siglip_attributes.py", "--include-decking"])

    args = parse_args()

    assert selected_targets(args) == DEFAULT_TARGETS


def test_default_image_root_uses_clean_images(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_siglip_attributes.py"])

    args = parse_args()

    assert args.image_root == Path("data/processed/images_clean")
