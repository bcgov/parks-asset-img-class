"""Tests for final pipeline input validation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import check_pipeline_inputs  # noqa: E402


def test_require_images_rejects_non_matching_clean_folder(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "environment.yml").write_text("name: test\n", encoding="utf-8")
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "attribute_applicability.csv").write_text("profile_name,attribute\n", encoding="utf-8")
    (processed / "train").mkdir()
    (processed / "master_dataset.csv").write_text(
        "asset_id,image_path\n1,data/citywide/images/337/1/photo.jpg\n",
        encoding="utf-8",
    )
    (processed / "images_clean").mkdir()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(check_pipeline_inputs, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_pipeline_inputs.py", "--require-images"],
    )

    assert check_pipeline_inputs.main() == 1
    captured = capsys.readouterr()
    assert "no sampled master image paths were found" in captured.out
    assert "data/processed/images_clean/citywide/images" in captured.out
