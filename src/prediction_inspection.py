"""Visualize VLM prediction errors with images for a specified attribute.

This script loads VLM predictions and ground truth, filters for wrong predictions,
and displays them with images to help identify patterns in VLM failures.

Images are embedded directly into the HTML as base64 data URIs so the report
renders in any browser, regardless of how the file is opened (no file:// issues).

Usage:
    python scripts/inspect_wrong_predictions.py \
        --predictions results/vlm_stairs_attributes/vlm_stairs_gemini-3-flash_complete.csv \
        --attribute steps_bin \
        --asset_type Stairs \
        --group_by_prediction
    
To render the HTML report:
    start results/prediction_inspection/stairs/wrong_predictions_steps_bin.html
    
    python scripts/inspect_wrong_predictions.py \
        --predictions results/vlm_bridge_attributes/vlm_trail_bridge_gemini-3-flash_complete.csv \
        --attribute has_pedestrian_railing \
        --asset_type "Trail Bridge" \
        --group_by_prediction
        
To render the HTML report:
    start results/prediction_inspection/trail_bridge/wrong_predictions_has_pedestrian_railing.html
"""

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional
from collections import defaultdict

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction_inspection import (
    load_predictions_and_ground_truth,
    get_wrong_predictions,
    extract_response_value,
    get_image_path,
    normalize_attribute_name,
)


def load_image_safe(image_path: Optional[Path], max_width: int = 300) -> Optional[Image.Image]:
    """Load an image, returning None if it doesn't exist or fails to load."""
    if image_path is None or not image_path.exists():
        return None

    try:
        img = Image.open(image_path)
        img.thumbnail((max_width, max_width), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Warning: Failed to load image {image_path}: {e}", file=sys.stderr)
        return None


def image_to_data_uri(image_path: Optional[Path], max_width: int = 400) -> Optional[str]:
    """Read an image and return a base64 data URI.

    Embedding the image directly in the HTML means it renders in any browser
    without relying on file:// access (which browsers often block).
    """
    if image_path is None or not Path(image_path).exists():
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((max_width, max_width), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"Warning: Failed to encode image {image_path}: {e}", file=sys.stderr)
        return None


def create_error_report_html(
    wrong_preds: pd.DataFrame,
    merged: pd.DataFrame,
    attribute: str,
    pred_column: str,
    gt_column: str,
    output_dir: Path,
    model_name: str = "",
    asset_type: str = "",
    limit: int = None,
) -> None:
    """Create an interactive HTML report of wrong predictions, grouped by asset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Work at the asset level: get unique wrong asset IDs
    wrong_asset_ids = wrong_preds["asset_id"].unique()
    total_asset_ids = merged["asset_id"].unique()

    if limit:
        wrong_asset_ids = wrong_asset_ids[:limit]

    asset_cards_html = []

    for asset_id in wrong_asset_ids:
        # All rows for this asset from the wrong_preds dataframe
        asset_rows = wrong_preds[wrong_preds["asset_id"] == asset_id]

        # Prediction and ground truth are per-asset, so take the first row's values
        first_row = asset_rows.iloc[0]
        predicted = first_row.get(pred_column, "N/A")
        actual = first_row.get(gt_column, "N/A")
        confidence = first_row.get(pred_column.replace("_value", "_confidence"), "N/A")
        timestamp = first_row.get("timestamp", "")

        # Try to extract prompt from response (use first row)
        prompt_text = ""
        if "raw_response" in first_row and pd.notna(first_row["raw_response"]):
            try:
                resp_data = json.loads(first_row["raw_response"])
                if "prompt" in resp_data:
                    prompt_text = resp_data["prompt"]
            except Exception:
                pass

        # Build one image element per row/image for this asset
        images_html_parts = []
        for _, row in asset_rows.iterrows():
            filename = row.get("filename", "")
            resolved_image = get_image_path(row)
            data_uri = image_to_data_uri(resolved_image)

            if data_uri:
                img_html = f"""
                <div class="image-wrapper">
                    <img src="{data_uri}"
                         alt="Asset {asset_id} image">
                    <div class="image-filename">{filename}</div>
                </div>
                """
            else:
                img_html = f"""
                <div class="image-wrapper image-missing">
                    <div>Image not found</div>
                    <div class="image-filename">{filename}</div>
                </div>
                """
            images_html_parts.append(img_html)

        images_html = "\n".join(images_html_parts)
        n_images = len(asset_rows)
        image_label = f"{n_images} image{'s' if n_images != 1 else ''}"

        asset_cards_html.append(f"""
        <div class="asset-card">
            <div class="card-header">
                <span class="asset-id">Asset {asset_id}</span>
                <span class="image-count">{image_label}</span>
                <span class="timestamp">{timestamp}</span>
            </div>
            <div class="card-body">
                <div class="prediction-row">
                    <span class="label">Predicted:</span>
                    <span class="value predicted">{predicted}</span>
                    <span class="confidence">(conf: {confidence})</span>
                </div>
                <div class="prediction-row">
                    <span class="label">Actual:</span>
                    <span class="value actual">{actual}</span>
                </div>
                {f'<div class="meta-row"><span class="label">Prompt:</span><span class="value meta-text prompt-text">{prompt_text}</span></div>' if prompt_text else ''}
                <div class="images-strip">
                    {images_html}
                </div>
            </div>
        </div>
        """)

    n_wrong_assets = len(wrong_asset_ids)
    n_total_assets = len(total_asset_ids)
    error_rate = 100 * n_wrong_assets / n_total_assets if n_total_assets > 0 else 0
    asset_type_line = f"<strong>Asset type:</strong> {asset_type}<br>" if asset_type else ""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VLM Wrong Predictions - {attribute}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: #f5f5f5;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            h1 {{
                color: #333;
                margin-bottom: 10px;
            }}
            .summary {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .asset-cards {{
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            .asset-card {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .card-header {{
                background: #f0f0f0;
                padding: 12px 16px;
                border-bottom: 1px solid #ddd;
                display: flex;
                align-items: center;
                gap: 16px;
            }}
            .asset-id {{
                font-weight: bold;
                color: #222;
                font-size: 1.05em;
            }}
            .image-count {{
                background: #e3eaf7;
                color: #3a5a9e;
                font-size: 0.82em;
                font-weight: 600;
                padding: 2px 10px;
                border-radius: 12px;
            }}
            .timestamp {{
                margin-left: auto;
                color: #888;
                font-size: 0.82em;
            }}
            .card-body {{
                padding: 16px;
            }}
            .prediction-row {{
                display: flex;
                margin-bottom: 10px;
                align-items: center;
                gap: 10px;
            }}
            .meta-row {{
                margin-bottom: 10px;
                font-size: 0.9em;
                display: flex;
                gap: 10px;
                align-items: flex-start;
            }}
            .label {{
                font-weight: 600;
                color: #555;
                min-width: 100px;
                flex-shrink: 0;
            }}
            .value {{
                flex: 1;
                padding: 5px 10px;
                background: #f5f5f5;
                border-radius: 4px;
                word-break: break-word;
            }}
            .predicted {{
                background: #ffe0e0;
                color: #c62828;
                font-weight: 600;
            }}
            .actual {{
                background: #e0f5e0;
                color: #2e7d32;
                font-weight: 600;
            }}
            .confidence {{
                font-size: 0.85em;
                color: #777;
                flex-shrink: 0;
            }}
            .meta-text {{
                background: #f9f9f9;
                color: #666;
                font-family: monospace;
                font-size: 0.82em;
            }}
            .prompt-text {{
                display: block;
                max-height: 80px;
                overflow-y: auto;
            }}
            /* Horizontal image strip */
            .images-strip {{
                display: flex;
                flex-direction: row;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 12px;
                padding-top: 12px;
                border-top: 1px solid #eee;
            }}
            .image-wrapper {{
                flex: 0 0 auto;
                width: 400px;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 4px;
            }}
            .image-wrapper img {{
                width: 400px;
                height: 300px;
                object-fit: contain;
                border-radius: 5px;
                border: 1px solid #ddd;
                background: #fafafa;
            }}
            .image-filename {{
                font-size: 0.72em;
                color: #888;
                word-break: break-all;
                text-align: center;
                max-width: 400px;
            }}
            .image-missing {{
                width: 400px;
                height: 300px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                border: 1px dashed #ccc;
                border-radius: 5px;
                color: #aaa;
                font-size: 0.85em;
                background: #fafafa;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>VLM Wrong Predictions: {attribute}</h1>
            <div class="summary">
                {asset_type_line}
                <strong>Model:</strong> {model_name or 'unknown'}<br>
                <strong>Attribute:</strong> {attribute}<br>
                <strong>Wrong assets:</strong> {n_wrong_assets} of {n_total_assets} ({error_rate:.1f}% asset-level error rate)
            </div>
            <div class="asset-cards">
                {"".join(asset_cards_html)}
            </div>
        </div>
    </body>
    </html>
    """

    output_file = output_dir / f"wrong_predictions_{normalize_attribute_name(attribute)}.html"
    output_file.write_text(html_content)
    print(f"\nHTML report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize VLM prediction errors with images"
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions CSV file",
    )
    parser.add_argument(
        "--attribute",
        required=True,
        help="Attribute to inspect (e.g., 'steps_bin', 'attr_material_frame,_tank,_body')",
    )
    parser.add_argument(
        "--asset_type",
        default="",
        help="Asset type label to display in the report (e.g., 'Stairs', 'Trail Bridge')",
    )
    parser.add_argument(
        "--ground_truth_dir",
        default="data/processed/train",
        help="Directory containing ground truth CSVs",
    )
    parser.add_argument(
        "--output_dir",
        default="results/prediction_inspection",
        help="Directory to save HTML reports",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of wrong assets to show",
    )
    parser.add_argument(
        "--group_by_prediction",
        action="store_true",
        help="Group errors by predicted value",
    )
    parser.add_argument(
        "--html_only",
        action="store_true",
        default=True,
        help="Output HTML report (default: True)",
    )
    args = parser.parse_args()

    # Load data
    try:
        merged, preds_df, pred_col, gt_col = load_predictions_and_ground_truth(
            args.predictions,
            args.attribute,
            args.ground_truth_dir,
        )
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    # Get wrong predictions (row-level, but we'll aggregate to asset-level for display)
    wrong_preds = get_wrong_predictions(merged, pred_col, gt_col)

    if len(wrong_preds) == 0:
        print(f"No wrong predictions found for attribute: {args.attribute}")
        sys.exit(0)

    # Report counts at asset level
    unique_wrong_assets = wrong_preds["asset_id"].nunique()
    unique_total_assets = merged["asset_id"].nunique()

    print(
        f"\nFound {unique_wrong_assets} wrong assets "
        f"out of {unique_total_assets} total assets"
    )
    print(
        f"Asset-level error rate: "
        f"{100 * unique_wrong_assets / unique_total_assets:.1f}%"
    )

    # Get model name from predictions CSV path or filename
    model_name = Path(args.predictions).stem
    if "gemini" in model_name.lower():
        model_name = "gemini-3-flash"
    elif "gemma" in model_name.lower():
        model_name = "gemma-2-27b"

    # Summary statistics (asset-level)
    # One row per asset for distribution summaries
    asset_level = wrong_preds.drop_duplicates(subset=["asset_id"])
    print(f"\nPredicted values distribution (asset-level):")
    print(asset_level[pred_col].value_counts().to_string())
    print(f"\nActual values distribution for wrong assets (asset-level):")
    print(asset_level[gt_col].value_counts().to_string())

    # Group by prediction if requested
    if args.group_by_prediction:
        print(f"\n=== Errors grouped by prediction (asset-level) ===")
        for pred_val, group in asset_level.groupby(pred_col):
            actual_vals = group[gt_col].value_counts()
            print(f"\nPredicted '{pred_val}' (n={len(group)} assets):")
            for actual_val, count in actual_vals.items():
                print(f"  → Actually '{actual_val}': {count}")

    # Build output dir: append asset_type slug as a subfolder when provided
    output_dir = Path(args.output_dir)
    if args.asset_type:
        asset_type_slug = args.asset_type.lower().replace(" ", "_")
        output_dir = output_dir / asset_type_slug

    # Create HTML report
    create_error_report_html(
        wrong_preds,
        merged,
        args.attribute,
        pred_col,
        gt_col,
        output_dir,
        model_name=model_name,
        asset_type=args.asset_type,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()