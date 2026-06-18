"""Tests for CityWide downloader helpers that do not call the API."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_citywide_images import image_destination, safe_filename, selected_profiles  # noqa: E402
from src.citywide_client import CitywideClient  # noqa: E402


def test_selected_profiles_returns_requested_profiles() -> None:
    profiles = selected_profiles([337, 356])

    assert profiles == {
        337: "Boardwalk < 1.2m High",
        356: "Stairs",
    }


def test_selected_profiles_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unknown CityWide profile"):
        selected_profiles([999])


def test_image_destination_uses_profile_asset_and_file_ids() -> None:
    file_record = {
        "asset_id": 123,
        "profile_id": 337,
        "id": 456,
        "filename": "photo 01!.jpg",
    }

    assert image_destination(Path("data/raw/citywide"), file_record) == Path(
        "data/raw/citywide/images/337/123/456__photo_01_.jpg"
    )


def test_safe_filename_limits_problem_characters() -> None:
    assert safe_filename("bridge photo (north).jpg") == "bridge_photo__north_.jpg"


def test_citywide_client_reports_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for credential in ("CITYWIDE_API_KEY", "CITYWIDE_DB", "CITYWIDE_USER"):
        monkeypatch.delenv(credential, raising=False)

    with pytest.raises(RuntimeError, match="CITYWIDE_API_KEY.*CITYWIDE_DB.*CITYWIDE_USER"):
        CitywideClient()
