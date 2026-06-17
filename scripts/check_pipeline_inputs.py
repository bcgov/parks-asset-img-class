"""Validate required inputs for the final Makefile pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vlm.config import detect_provider, missing_credentials_for_provider  # noqa: E402


REQUIRED_PATHS = [
    Path("environment.yml"),
    Path("data/processed/master_dataset.csv"),
    Path("data/processed/train")
]

CITYWIDE_CREDENTIALS = [
    "CITYWIDE_API_KEY",
    "CITYWIDE_DB",
    "CITYWIDE_USER",
]


def parse_args() -> argparse.Namespace:
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
        "--dinov3-weights",
        type=Path,
        default=Path("models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    )
    return parser.parse_args()


def main() -> int:
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
