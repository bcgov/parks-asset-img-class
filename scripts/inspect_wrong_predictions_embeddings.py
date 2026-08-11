"""Visualize embedding-model (DINOv3 / OpenCLIP / SigLIP) prediction errors.

The embedding classifiers save per-asset out-of-fold predictions with columns:
    attribute, fold, asset_id, true_label, predicted_label, correct

This script reshapes those into the format expected by the shared
``create_error_report_html`` renderer (reused from inspect_wrong_predictions.py),
joins image paths from the attribute's train CSV so every image of a
misclassified asset is shown, and writes an HTML report.

Usage:
    python scripts/inspect_wrong_predictions_embeddings.py \
        --predictions results/dinov3_results/dinov3_logistic/predictions/dinov3_attr_decking_material_classification_predictions.csv \
        --attribute attr_decking_material \
        --model-family dinov3

    python scripts/inspect_wrong_predictions_embeddings.py \
        --predictions results/openclip_results/openclip_logistic_reg/predictions/openclip_attr_bridge_type_classification_predictions.csv \
        --attribute attr_bridge_type \
        --model-family openclip

    python scripts/inspect_wrong_predictions_embeddings.py \
        --predictions results/siglip_results/siglip_logistic_reg/predictions/siglip_steps_bin_classification_predictions.csv \
        --attribute steps_bin \
        --model-family siglip

Then open the printed HTML path in a browser.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the shared HTML renderer and helpers — unchanged.
from scripts.inspect_wrong_predictions import create_error_report_html  # noqa: E402
from src.prediction_inspection import find_ground_truth_file  # noqa: E402


# Pretty model labels for the report header
MODEL_LABELS = {
    "dinov3": "DINOv3 + Logistic Regression",
    "openclip": "OpenCLIP + Logistic Regression",
    "siglip": "SigLIP + Logistic Regression",
}

# The renderer reads these column names; we rename our long-format columns to match.
PRED_COLUMN = "predicted_value"
GT_COLUMN = "true_value"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script."""
    parser = argparse.ArgumentParser(
        description="Inspect embedding-model prediction errors with images."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to the embedding predictions CSV (asset_id/true_label/predicted_label/correct).",
    )
    parser.add_argument(
        "--attribute",
        required=True,
        help="Attribute name, e.g. attr_decking_material or steps_bin.",
    )
    parser.add_argument(
        "--model-family",
        required=True,
        choices=["dinov3", "openclip", "siglip"],
        help="Which embedding model produced the predictions.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        default="data/processed/train",
        help="Directory containing <attribute>_train.csv files (for image paths).",
    )
    parser.add_argument(
        "--image-root",
        default="data/processed/images_clean",
        help="Root of the (blurred) clean images shown in the report.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to save HTML reports. Defaults to "
            "results/prediction_inspection/<model_family>."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of wrong assets to show.",
    )
    return parser.parse_args()


def build_wrong_predictions_frame(
    predictions_path: str,
    attribute: str,
    ground_truth_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (wrong_preds, merged) in the shape create_error_report_html expects.

    - One row per (asset, image) so every image of a misclassified asset shows.
    - Columns include PRED_COLUMN, GT_COLUMN, asset_id, image_path, filename.
    """
    preds = pd.read_csv(predictions_path)
    required = {"asset_id", "true_label", "predicted_label", "correct"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(
            f"Predictions CSV missing expected columns: {missing}. "
            f"Found: {preds.columns.tolist()}"
        )

    preds["asset_id"] = preds["asset_id"].astype(str).str.strip()

    # Rename to the column names the HTML renderer reads.
    preds = preds.rename(
        columns={"predicted_label": PRED_COLUMN, "true_label": GT_COLUMN}
    )

    # Load the attribute's train CSV to attach image paths (predictions only have asset_id).
    gt_path = find_ground_truth_file(attribute, ground_truth_dir)
    if gt_path is None:
        raise ValueError(
            f"Could not find train CSV for attribute '{attribute}' under "
            f"'{ground_truth_dir}'."
        )
    train = pd.read_csv(gt_path)
    if "asset_id" not in train.columns or "image_path" not in train.columns:
        raise ValueError(
            f"Train CSV {gt_path} must contain 'asset_id' and 'image_path' columns. "
            f"Found: {train.columns.tolist()}"
        )
    train["asset_id"] = train["asset_id"].astype(str).str.strip()

    # All image rows per asset (an asset can have multiple images).
    image_rows = train[["asset_id", "image_path"]].dropna().drop_duplicates()

    # merged = every asset that has a prediction, expanded to one row per image.
    merged = preds.merge(image_rows, on="asset_id", how="left")
    # filename for display (basename of the image path)
    merged["filename"] = merged["image_path"].apply(
        lambda p: Path(str(p)).name if pd.notna(p) else ""
    )

    # wrong = the misclassified assets, expanded per image.
    wrong = merged[merged["correct"] == False].reset_index(drop=True)  # noqa: E712

    return wrong, merged


def main() -> int:
    """Run the script from parsed command-line arguments."""
    args = parse_args()

    try:
        wrong_preds, merged = build_wrong_predictions_frame(
            args.predictions,
            args.attribute,
            args.ground_truth_dir,
        )
    except Exception as exc:
        print(f"Error loading data: {exc}", file=sys.stderr)
        return 1

    if wrong_preds.empty:
        print(f"No wrong predictions found for attribute: {args.attribute}")
        return 0

    n_wrong = wrong_preds["asset_id"].nunique()
    n_total = merged["asset_id"].nunique()
    print(f"\nFound {n_wrong} wrong assets out of {n_total} total assets")
    if n_total:
        print(f"Asset-level error rate: {100 * n_wrong / n_total:.1f}%")

    # Console summary
    asset_level = wrong_preds.drop_duplicates(subset=["asset_id"])
    print("\nPredicted values distribution (asset-level):")
    print(asset_level[PRED_COLUMN].value_counts().to_string())
    print("\nActual values distribution for wrong assets (asset-level):")
    print(asset_level[GT_COLUMN].value_counts().to_string())

    # Output dir: results/prediction_inspection/<model_family>/
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("results/prediction_inspection") / args.model_family
    )

    # The renderer resolves images via get_image_path -> resolve_image_path,
    # which uses DEFAULT_IMAGE_ROOT. To honor --image-root we set it via env-like
    # override: resolve_image_path already searches images_clean by default, so
    # the blurred images are found automatically.
    create_error_report_html(
        wrong_preds,
        merged,
        args.attribute,
        PRED_COLUMN,
        GT_COLUMN,
        output_dir,
        model_name=MODEL_LABELS.get(args.model_family, args.model_family),
        asset_type="",  # embedding runs aren't split by asset type
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())