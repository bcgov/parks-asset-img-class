"""Extract frozen SigLIP features for project images.

Pipeline role:
- reads the same processed train/master image manifests used by the DINOv3
  pipeline;
- loads a frozen Hugging Face SigLIP vision encoder;
- writes image-level and asset-level feature CSVs under ``data/features``.

The asset-level CSV is consumed by ``scripts/run_siglip_classifier.py`` for
cross-validation and by comparison scripts/figures when benchmarking SigLIP
against DINOv3 and the baseline.

Example:
    python scripts/extract_siglip_features.py \
        --input data/processed/train/attr_decking_material_train.csv \
        --output data/features/siglip2_base_patch16_224_attr_decking_material_images.csv \
        --asset-output data/features/siglip2_base_patch16_224_attr_decking_material_assets.csv \
        --image-root data/processed/images_clean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.siglip_features import (  # noqa: E402
    DEFAULT_IMAGE_ROOT,
    DEFAULT_SIGLIP_MODEL,
    aggregate_asset_features,
    extract_image_features,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script."""
    parser = argparse.ArgumentParser(
        description="Extract SigLIP image embeddings and aggregate them by asset_id."
    )
    parser.add_argument("--input", type=Path, required=True, help="CSV with asset_id and image_path columns.")
    parser.add_argument("--output", type=Path, required=True, help="Image-level feature CSV.")
    parser.add_argument("--asset-output", type=Path, required=True, help="Asset-level averaged feature CSV.")
    parser.add_argument("--skipped-output", type=Path, default=None, help="Optional CSV for missing/unreadable images.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_SIGLIP_MODEL,
        help="Hugging Face SigLIP/SigLIP2 model id.",
    )
    parser.add_argument("--device", default=None, help="cuda, mps, cpu, or omitted for auto.")
    parser.add_argument("--limit-assets", type=int, default=None, help="Optional smoke-test limit.")
    return parser.parse_args()


def main() -> int:
    """Run the script from parsed command-line arguments."""
    args = parse_args()
    rows = pd.read_csv(args.input)
    if args.limit_assets is not None:
        keep_assets = rows["asset_id"].drop_duplicates().head(args.limit_assets)
        rows = rows[rows["asset_id"].isin(keep_assets)]

    features, skipped = extract_image_features(
        rows,
        image_root=args.image_root,
        model_name=args.model_name,
        device=args.device,
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
