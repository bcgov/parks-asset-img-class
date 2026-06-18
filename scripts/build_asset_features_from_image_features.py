"""Aggregate precomputed image-level features to asset-level features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dinov3_features import feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master",
        type=Path,
        default=Path("data/processed/master_dataset.csv"),
        help="Master CSV containing asset_id and image_path.",
    )
    parser.add_argument(
        "--image-features",
        type=Path,
        required=True,
        help="Image-level feature file (.parquet or .csv) with image_path and f_* columns.",
    )
    parser.add_argument(
        "--asset-output",
        type=Path,
        required=True,
        help="Output asset-level feature CSV.",
    )
    return parser.parse_args()


def read_features(path: Path) -> pd.DataFrame:
    """Read a CSV or parquet feature file."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature file extension: {path.suffix}")


def build_asset_features(master: pd.DataFrame, image_features: pd.DataFrame) -> pd.DataFrame:
    """Join image features to asset_id and average them per asset."""
    if "asset_id" not in master.columns or "image_path" not in master.columns:
        raise ValueError("master must contain asset_id and image_path columns.")
    if "image_path" not in image_features.columns:
        raise ValueError("image_features must contain an image_path column.")

    features = feature_columns(image_features.columns)
    if not features:
        raise ValueError("image_features must contain f_* feature columns.")

    image_to_asset = master[["asset_id", "image_path"]].dropna().drop_duplicates()
    joined = image_to_asset.merge(image_features[["image_path", *features]], on="image_path", how="inner")
    if joined.empty:
        raise ValueError("No image features matched master image_path values.")

    return (
        joined.groupby("asset_id", as_index=False)[features]
        .mean()
        .sort_values("asset_id")
        .reset_index(drop=True)
    )


def main() -> int:
    args = parse_args()
    master = pd.read_csv(args.master)
    image_features = read_features(args.image_features)
    asset_features = build_asset_features(master, image_features)

    args.asset_output.parent.mkdir(parents=True, exist_ok=True)
    asset_features.to_csv(args.asset_output, index=False)
    print(
        f"Wrote {len(asset_features)} asset feature rows and "
        f"{len(feature_columns(asset_features.columns))} features to {args.asset_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
