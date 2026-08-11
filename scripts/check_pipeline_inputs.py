"""Validate required inputs for the final Makefile pipeline.

Pipeline role:
- runs near the start of ``make all`` and related Makefile targets;
- checks that committed configuration, processed training data, cleaned-image
  markers, and optional credentials are present before expensive work starts;
- fails early with actionable messages instead of letting downstream scripts
  fail halfway through the pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.dinov3_features import resolve_image_path  # noqa: E402
from src.vlm.config import detect_provider, missing_credentials_for_provider  # noqa: E402


REQUIRED_PATHS = [
    Path("environment.yml"),
    Path("data/processed/attribute_applicability.csv"),
    Path("data/processed/master_dataset.csv"),
    Path("data/processed/train")
]

CITYWIDE_CREDENTIALS = [
    "CITYWIDE_API_KEY",
    "CITYWIDE_DB",
    "CITYWIDE_USER",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Require the cleaned image directory used by model/VLM pipelines.",
    )
    parser.add_argument(
        "--require-dinov3-weights",
        action="store_true",
        help="Require a local DINOv3 checkpoint for feature extraction.",
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        action="append",
        default=[],
        help="Require a precomputed feature file. May be supplied multiple times.",
    )
    parser.add_argument(
        "--require-vlm-credentials",
        action="store_true",
        help="Require API credentials for the selected cloud VLM model.",
    )
    parser.add_argument(
        "--require-citywide-credentials",
        action="store_true",
        help="Require CityWide API credentials for raw asset/image downloads.",
    )
    parser.add_argument(
        "--vlm-model",
        default="gemini-3-flash-preview",
        help="Cloud VLM model name used to determine required credential variables.",
    )
    parser.add_argument(
        "--vlm-provider",
        default="auto",
        help="Cloud VLM provider: auto, gemini, openai, grok, claude, or github.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/processed/images_clean"),
    )
    parser.add_argument(
        "--master-data",
        type=Path,
        default=Path("data/processed/master_dataset.csv"),
        help="Master dataset used to verify cleaned image paths.",
    )
    parser.add_argument(
        "--image-check-limit",
        type=int,
        default=50,
        help="Number of master image paths to sample when --require-images is set.",
    )
    parser.add_argument(
        "--dinov3-weights",
        type=Path,
        default=Path("models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    )
    return parser.parse_args()


def main() -> int:
    """Run the script from parsed command-line arguments."""
    args = parse_args()
    required = list(REQUIRED_PATHS)
    if args.require_images:
        required.append(args.image_root)
    if args.require_dinov3_weights:
        required.append(args.dinov3_weights)
    required.extend(args.feature_file)

    missing = [path for path in required if not path.exists()]
    if missing:
        print("Missing required pipeline input(s):")
        for path in missing:
            print(f"  - {path}")
        return 1

    if args.require_images:
        try:
            master_images = (
                pd.read_csv(args.master_data, usecols=["image_path"])
                ["image_path"]
                .dropna()
                .drop_duplicates()
                .head(args.image_check_limit)
            )
        except ValueError:
            print(f"Cannot verify cleaned images because {args.master_data} has no image_path column.")
            return 1

        if master_images.empty:
            print(f"Cannot verify cleaned images because {args.master_data} has no image paths.")
            return 1

        resolved = [
            resolve_image_path(
                image_path,
                image_root=args.image_root,
                repo_root=REPO_ROOT,
            )
            for image_path in master_images
        ]
        matches = [path for path in resolved if path.exists()]
        if not matches:
            print("Cleaned image directory exists, but no sampled master image paths were found there.")
            print(f"  Image root checked: {args.image_root}")
            print(f"  Master data checked: {args.master_data}")
            print("  Expected paths like:")
            for path in resolved[:3]:
                print(f"    - {path}")
            print("Run `make pii` after adding raw images, or check that the SharePoint/API data")
            print("was copied so cleaned images are under data/processed/images_clean/citywide/images/.")
            return 1

    if args.require_vlm_credentials:
        provider = detect_provider(args.vlm_model, args.vlm_provider)
        missing_credentials = missing_credentials_for_provider(provider, args.vlm_model)
        if missing_credentials:
            print(f"Missing VLM credential(s) for provider/model {provider}/{args.vlm_model}:")
            for credential in missing_credentials:
                print(f"  - {credential}")
            return 1

    if args.require_citywide_credentials:
        missing_citywide_credentials = [
            credential for credential in CITYWIDE_CREDENTIALS if not os.getenv(credential)
        ]
        if missing_citywide_credentials:
            print("Missing CityWide credential(s):")
            for credential in missing_citywide_credentials:
                print(f"  - {credential}")
            print("Add them to .env or export them before running CityWide download targets.")
            return 1

    print("Pipeline inputs look ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
