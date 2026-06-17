"""Predict asset attributes for a folder of NEW images (BC Parks deployment).

Point this at a folder of images, and it encodes them with DINOv3, trains the
final per-attribute classifiers on the labelled training data, and writes
predictions + confidence scores to CSV. Only attributes that apply to each
asset type are predicted (per the applicability matrix), and binned numeric
attributes (length, width, fall height) are trained per asset type so each
asset is labelled in its own bin scheme.

Two input layouts are supported:

1. Flat folder of one declared asset type (use --asset-type):
       folder/
         <asset_id>/ img1.jpg img2.jpg ...
         <asset_id>/ img1.jpg ...
   Example:
       python scripts/predict_new_images.py \
           --image-folder data/new_stairs \
           --asset-type "Stairs" \
           --weights models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
           --output results/final/new_stairs_predictions.csv

2. CityWide-style profile structure (asset type inferred per profile_id folder):
       folder/
         356/                 (profile_id -> Stairs)
           <asset_id>/ imgs
         253/                 (profile_id -> Trail Bridge)
           <asset_id>/ imgs
   Example:
       python scripts/predict_new_images.py \
           --image-folder data/raw/citywide/images \
           --weights models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
           --output results/final/new_predictions.csv

Multiple images in one asset_id folder are averaged into a single embedding,
matching how the training features were built.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_citywide_images import PROFILES  # noqa: E402
from src.baseline import infer_target_column  # noqa: E402
from src.dinov3_classifier import make_classifier  # noqa: E402
from src.dinov3_features import (  # noqa: E402
    aggregate_asset_features,
    extract_image_features,
    feature_columns,
)

# Binned numeric attributes whose bin ranges differ across asset types — trained
# per asset type. steps_bin is Stairs-only (single scheme) so it is pooled.
PER_ASSET_TYPE_TARGETS = {"length_bin", "width_bin", "fall_height_bin"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

VALID_ASSET_TYPES = set(PROFILES.values())
PROFILE_ID_TO_NAME = {str(profile_id): name for profile_id, name in PROFILES.items()}


# ---------------------------------------------------------------------
# Confidence bucketing (kept local to avoid importing the export module)
# ---------------------------------------------------------------------

def confidence_bucket(
    score: object,
    *,
    high_threshold: float = 0.80,
    medium_threshold: float = 0.60,
) -> str:
    if pd.isna(score):
        return "unavailable"
    value = float(score)
    if value >= high_threshold:
        return "high"
    if value >= medium_threshold:
        return "medium"
    return "low"


def _predict_confidence(model: object, X: pd.DataFrame) -> list[object]:
    if not hasattr(model, "predict_proba"):
        return [pd.NA] * len(X)
    return model.predict_proba(X).max(axis=1).tolist()


# ---------------------------------------------------------------------
# Folder walking -> (asset_id, image_path, profile_name) table
# ---------------------------------------------------------------------

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def build_input_table(image_folder: Path, asset_type: str | None) -> pd.DataFrame:
    """Walk the image folder and return rows of asset_id, image_path, profile_name.

    Flat mode (asset_type given): folder/<asset_id>/<images>.
    Profile mode (no asset_type): folder/<profile_id>/<asset_id>/<images>.
    """
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    rows: list[dict[str, object]] = []

    if asset_type is not None:
        # Flat mode: each immediate subfolder is one asset of the declared type.
        if asset_type not in VALID_ASSET_TYPES:
            raise ValueError(
                f"--asset-type {asset_type!r} is not a known asset type. "
                f"Valid: {sorted(VALID_ASSET_TYPES)}"
            )
        asset_dirs = [p for p in sorted(image_folder.iterdir()) if p.is_dir()]
        if not asset_dirs:
            # No subfolders: treat each image in the folder as its own asset.
            for img in sorted(image_folder.iterdir()):
                if _is_image(img):
                    rows.append({
                        "asset_id": img.stem,
                        "image_path": str(img),
                        "profile_name": asset_type,
                    })
        else:
            for asset_dir in asset_dirs:
                for img in sorted(asset_dir.iterdir()):
                    if _is_image(img):
                        rows.append({
                            "asset_id": asset_dir.name,
                            "image_path": str(img),
                            "profile_name": asset_type,
                        })
    else:
        # Profile mode: folder/<profile_id>/<asset_id>/<images>.
        profile_dirs = [p for p in sorted(image_folder.iterdir()) if p.is_dir()]
        if not profile_dirs:
            raise ValueError(
                "No profile_id subfolders found and no --asset-type given. "
                "Either pass --asset-type for a flat folder, or use the "
                "<profile_id>/<asset_id>/<images> layout."
            )
        for profile_dir in profile_dirs:
            profile_name = PROFILE_ID_TO_NAME.get(profile_dir.name)
            if profile_name is None:
                print(
                    f"  [skip] folder '{profile_dir.name}' is not a known profile_id "
                    f"({sorted(PROFILE_ID_TO_NAME)}); skipping."
                )
                continue
            for asset_dir in sorted(profile_dir.iterdir()):
                if not asset_dir.is_dir():
                    continue
                for img in sorted(asset_dir.iterdir()):
                    if _is_image(img):
                        rows.append({
                            "asset_id": asset_dir.name,
                            "image_path": str(img),
                            "profile_name": profile_name,
                        })

    if not rows:
        raise ValueError(f"No images found under {image_folder}.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Applicability matrix
# ---------------------------------------------------------------------

def load_applicability(path: Path) -> dict[str, set[str]]:
    """Return {asset_type: set(applicable target names)} from the matrix CSV.

    Expects a column 'Attribute' (internal target names) plus one column per
    asset type whose header matches a PROFILES value, with non-empty cells
    (e.g. 'X') marking applicability.
    """
    matrix = pd.read_csv(path)
    if "Attribute" not in matrix.columns:
        raise ValueError(f"Applicability CSV must have an 'Attribute' column. Got {matrix.columns.tolist()}")

    asset_type_columns = [c for c in matrix.columns if c in VALID_ASSET_TYPES]
    if not asset_type_columns:
        raise ValueError(
            "Applicability CSV has no asset-type columns matching PROFILES. "
            f"Expected some of {sorted(VALID_ASSET_TYPES)}; got {matrix.columns.tolist()}"
        )

    applicable: dict[str, set[str]] = {atype: set() for atype in asset_type_columns}
    for _, row in matrix.iterrows():
        target = str(row["Attribute"]).strip()
        for atype in asset_type_columns:
            cell = row[atype]
            if pd.notna(cell) and str(cell).strip() != "":
                applicable[atype].add(target)
    return applicable


# ---------------------------------------------------------------------
# Train + predict one attribute for one asset type
# ---------------------------------------------------------------------

def _train_subset(labels: pd.DataFrame, target_column: str, features: list[str],
                  asset_features: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    """Build the training frame for one asset type (per-asset-type targets) or
    all asset types (pooled targets)."""
    keep = ["asset_id", target_column]
    if "profile_name" in labels.columns:
        keep.append("profile_name")
    labelled = labels[keep].dropna(subset=[target_column]).drop_duplicates("asset_id")
    train = labelled.merge(asset_features[["asset_id", *features]], on="asset_id", how="inner")
    return train


def predict_attribute_for_type(
    *,
    target: str,
    asset_type: str,
    train_dir: Path,
    new_assets: pd.DataFrame,
    train_asset_features: pd.DataFrame,
    features: list[str],
    classifier: str,
    random_state: int,
    high_threshold: float,
    medium_threshold: float,
) -> pd.DataFrame:
    """Train on labelled data and predict the new assets of one asset type."""
    train_path = train_dir / f"{target}_train.csv"
    if not train_path.exists():
        print(f"  [skip] {target}: no train CSV at {train_path}")
        return pd.DataFrame()

    labels = pd.read_csv(train_path)
    target_column = infer_target_column(labels, target)
    train = _train_subset(labels, target_column, features, train_asset_features, asset_type)

    # For per-asset-type targets, restrict training to this asset type so the
    # bin scheme matches. Pooled targets train on all asset types.
    if target in PER_ASSET_TYPE_TARGETS and "profile_name" in train.columns:
        train = train[train["profile_name"] == asset_type]

    if train[target_column].nunique(dropna=True) < 2:
        print(f"  [skip] {target} / {asset_type}: fewer than two training classes; left unpredicted.")
        return pd.DataFrame()

    model = make_classifier(classifier=classifier, random_state=random_state)
    model.fit(train[features], train[target_column])

    X = new_assets[features]
    predictions = model.predict(X)
    confidences = _predict_confidence(model, X)

    out = new_assets[["asset_id", "profile_name"]].copy()
    out["attribute"] = target
    out["predicted_value"] = predictions
    out["confidence_score"] = confidences
    out["confidence_level"] = [
        confidence_bucket(c, high_threshold=high_threshold, medium_threshold=medium_threshold)
        for c in confidences
    ]
    return out


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-folder", type=Path, required=True,
                        help="Folder of new images (flat with --asset-type, or profile_id structure).")
    parser.add_argument("--asset-type", default=None,
                        help='Declare the asset type for a flat folder, e.g. "Stairs". '
                             "Omit to infer per-folder from a profile_id structure.")
    parser.add_argument("--applicability", type=Path,
                        default=Path("data/processed/attribute_applicability.csv"))
    parser.add_argument("--train-dir", type=Path, default=Path("data/processed/train"))
    parser.add_argument("--weights", type=Path,
                        default=Path("models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"))
    parser.add_argument("--model", default="dinov3_vitb16")
    parser.add_argument("--model-source", default="facebookresearch/dinov3")
    parser.add_argument("--image-root", type=Path, default=Path("."),
                        help="Root for resolving image paths (image_path values are already absolute here).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--classifier", default="logistic_regression")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--high-threshold", type=float, default=0.80)
    parser.add_argument("--medium-threshold", type=float, default=0.60)
    parser.add_argument("--output", type=Path,
                        default=Path("results/final/new_image_predictions_long.csv"))
    parser.add_argument("--output-wide", type=Path, default=None,
                        help="Optional wide CSV (one row per asset, columns per attribute).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 1. Build the input table from the folder.
    rows = build_input_table(args.image_folder, args.asset_type)
    asset_types_present = sorted(rows["profile_name"].unique())
    print(f"Found {rows['asset_id'].nunique()} asset(s) across {len(rows)} image(s).")
    print(f"Asset type(s): {asset_types_present}")

    # 2. Extract DINOv3 embeddings for the new images, aggregate per asset.
    print("Extracting DINOv3 embeddings for new images...")
    image_features, skipped = extract_image_features(
        rows,
        image_root=args.image_root,
        model_name=args.model,
        model_source=args.model_source,
        weights=str(args.weights),
        device=args.device,
        repo_root=REPO_ROOT,
    )
    if image_features.empty:
        print("No embeddings extracted — check image paths and weights.", file=sys.stderr)
        return 1
    if not skipped.empty:
        print(f"  ({len(skipped)} image(s) skipped as missing/unreadable)")

    asset_features = aggregate_asset_features(image_features)
    # carry profile_name onto the aggregated assets
    asset_profile = rows[["asset_id", "profile_name"]].drop_duplicates("asset_id")
    asset_features = asset_features.merge(asset_profile, on="asset_id", how="left")
    features = feature_columns(asset_features.columns)

    # 3. Applicability matrix.
    applicable = load_applicability(args.applicability)

    # 4. Predict each applicable attribute, per asset type present.
    all_predictions: list[pd.DataFrame] = []
    for asset_type in asset_types_present:
        type_assets = asset_features[asset_features["profile_name"] == asset_type].reset_index(drop=True)
        if type_assets.empty:
            continue
        targets = sorted(applicable.get(asset_type, set()))
        if not targets:
            print(f"[{asset_type}] no applicable attributes in matrix; skipping.")
            continue
        print(f"[{asset_type}] predicting {len(targets)} applicable attribute(s): {targets}")
        for target in targets:
            preds = predict_attribute_for_type(
                target=target,
                asset_type=asset_type,
                train_dir=args.train_dir,
                new_assets=type_assets,
                train_asset_features=asset_features,
                features=features,
                classifier=args.classifier,
                random_state=args.seed,
                high_threshold=args.high_threshold,
                medium_threshold=args.medium_threshold,
            )
            if not preds.empty:
                all_predictions.append(preds)

    if not all_predictions:
        print("No predictions produced.", file=sys.stderr)
        return 1

    long_df = pd.concat(all_predictions, ignore_index=True)
    long_df["model_name"] = args.model
    long_df["classifier"] = args.classifier
    long_df["generated_at"] = datetime.now(UTC).isoformat()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.output, index=False)
    print(f"\nWrote {len(long_df)} prediction rows to {args.output}")

    if args.output_wide is not None:
        wide = long_df.pivot_table(
            index=["asset_id", "profile_name"],
            columns="attribute",
            values=["predicted_value", "confidence_score", "confidence_level"],
            aggfunc="first",
        )
        wide.columns = [f"{attr}_{field}" for field, attr in wide.columns]
        wide = wide.reset_index()
        args.output_wide.parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(args.output_wide, index=False)
        print(f"Wrote {wide.shape[0]} wide asset rows to {args.output_wide}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())