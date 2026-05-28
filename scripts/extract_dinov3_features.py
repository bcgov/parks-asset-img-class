"""Extract frozen DINOv3 features for project images.

First experiment:
    python scripts/extract_dinov3_features.py \
        --input data/processed/train/attr_decking_material_train.csv \
        --output data/features/dinov3_vitb16_attr_decking_material_images.csv \
        --asset-output data/features/dinov3_vitb16_attr_decking_material_assets.csv \
        --model dinov3_vitb16

If the machine cannot download from torch.hub, clone the official DINOv3 repo
elsewhere and pass ``--model-source /path/to/dinov3``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dinov3_features import (  # noqa: E402
    DEFAULT_DINOV3_MODEL,
    DEFAULT_IMAGE_ROOT,
    aggregate_asset_features,
    extract_image_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 image embeddings and aggregate them by asset_id."
    )
    parser.add_argument("--input", type=Path, required=True, help="CSV with asset_id and image_path columns.")
    parser.add_argument("--output", type=Path, required=True, help="Image-level feature CSV.")
    parser.add_argument("--asset-output", type=Path, required=True, help="Asset-level averaged feature CSV.")
    parser.add_argument("--skipped-output", type=Path, default=None, help="Optional CSV for missing/unreadable images.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--model", default=DEFAULT_DINOV3_MODEL)
    parser.add_argument("--model-source", default="facebookresearch/dinov3")
    parser.add_argument(
        "--weights",
        default=None,
        help="DINOv3 checkpoint URL or local .pth path from approved model-weight access.",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, mps, or omitted for auto.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--limit-assets", type=int, default=None, help="Optional smoke-test limit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = pd.read_csv(args.input)
    if args.limit_assets is not None:
        keep_assets = rows["asset_id"].drop_duplicates().head(args.limit_assets)
        rows = rows[rows["asset_id"].isin(keep_assets)]

    features, skipped = extract_image_features(
        rows,
        image_root=args.image_root,
        model_name=args.model,
        model_source=args.model_source,
        weights=args.weights,
        device=args.device,
        image_size=args.image_size,
        repo_root=REPO_ROOT,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.asset_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)

    if features.empty:
        print("No image features were extracted. Check --image-root and image availability.")
    else:
        asset_features = aggregate_asset_features(features)
        asset_features.to_csv(args.asset_output, index=False)
        print(f"Wrote {len(features)} image feature rows to {args.output}")
        print(f"Wrote {len(asset_features)} asset feature rows to {args.asset_output}")

    skipped_path = args.skipped_output or args.output.with_name(f"{args.output.stem}_skipped.csv")
    if not skipped.empty:
        skipped_path.parent.mkdir(parents=True, exist_ok=True)
        skipped.to_csv(skipped_path, index=False)
        print(f"Wrote {len(skipped)} skipped image rows to {skipped_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
