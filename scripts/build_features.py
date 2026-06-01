"""Build the DINOv3 (or DINOv2) feature cache for the train+test images.

Heavy step — run once on the DGX Spark, then every downstream pipeline
(DINOv3 heads, k-NN, YOLO crop classifier, stacking) reads the parquet.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --model facebook/dinov2-large
    python scripts/build_features.py --max-images 64  # smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.data.splits import DEFAULT_SPLIT_SEED, load_split  # noqa: E402
from src.embed import (  # noqa: E402
    DEFAULT_DINOV3_MODEL,
    extract_features_for_split,
    save_features,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_DINOV3_MODEL)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features"),
        help="Parquet output directory (gitignored).",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, choices=[None, "fp16", "bf16"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    train_df, test_df = load_split(processed_dir=args.data_dir, split_seed=args.split_seed)
    combined = pd.concat([train_df, test_df], ignore_index=True).drop_duplicates(
        subset="image_path"
    )
    print(
        f"Will encode {len(combined)} images "
        f"({len(train_df)} train + {len(test_df)} test, deduped)"
    )

    cache = extract_features_for_split(
        combined,
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_images=args.max_images,
    )

    print(f"Encoded {len(cache.df)} images into {cache.dim}-d embeddings.")
    p = save_features(cache, out_dir=args.output_dir)
    print(f"Wrote feature cache to {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
