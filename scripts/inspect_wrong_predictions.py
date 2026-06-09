#!/usr/bin/env python3
"""Visualize VLM prediction errors with images for a specified attribute.

This script loads VLM predictions and ground truth, filters for wrong predictions,
and displays them with images to help identify patterns in VLM failures.

Usage:
    python scripts/inspect_wrong_predictions.py \
        --predictions results/vlm_stairs_attributes/vlm_stairs_gemini-3-flash_complete.csv \
        --attribute steps_bin \
        --group_by_prediction
    
    python scripts/inspect_wrong_predictions.py \
        --predictions results/vlm_bridge_attributes/vlm_trail_bridge_gemini-3-flash_complete.csv \
        --attribute has_pedestrian_railing \
        --group_by_prediction
        
To render the HTML file:
    start results/prediction_inspection/wrong_predictions_steps_bin.html
"""

import argparse
import json
import sys
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
        # Resize to reasonable size for display
        img.thumbnail((max_width, max_width), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"Warning: Failed to load image {image_path}: {e}", file=sys.stderr)
        return None


def create_error_report_html(
    wrong_preds: pd.DataFrame,
    attribute: str,
    pred_column: str,
    gt_column: str,
    output_dir: Path,
    model_name: str = "",
    limit: int = None,
) -> None:
    """Create an interactive HTML report of wrong predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if limit:
        wrong_preds = wrong_preds.head(limit)

    rows_html = []

    for idx, row in wrong_preds.iterrows():
        asset_id = row.get("asset_id", "unknown")
        predicted = row.get(pred_column, "N/A")
        actual = row.get(gt_column, "N/A")
        confidence = row.get(pred_column.replace("_value", "_confidence"), "N/A")
        filename = row.get("filename", "")
        resolved_image = get_image_path(row)

        img_html = ""
        if resolved_image and resolved_image.exists():
            img_html = f"""
            <div class="image-container">
                <img src="{resolved_image.resolve().as_uri()}"
                    alt="Asset image">
            </div>
            """
        else:
            img_html = """
            <div class="image-container image-missing">
                Image not found
            </div>
            """
        timestamp = row.get("timestamp", "")

        # Try to extract prompt from response
        prompt_text = ""
        if "raw_response" in row and pd.notna(row["raw_response"]):
            try:
                resp_data = json.loads(row["raw_response"])
                if "prompt" in resp_data:
                    prompt_text = resp_data["prompt"]
            except:
                pass

        rows_html.append(f"""
        <div class="prediction-card">
            <div class="card-header">
                <span class="asset-id">Asset {asset_id}</span>
                <span class="timestamp">{timestamp}</span>
            </div>
            <div class="card-body">
                {img_html}

                <div class="prediction-row">
                    <span class="label">Predicted:</span>
                    <span class="value predicted">{predicted}</span>
                    <span class="confidence">(conf: {confidence})</span>
                </div>
                <div class="prediction-row">
                    <span class="label">Actual:</span>
                    <span class="value actual">{actual}</span>
                </div>
                <div class="meta-row">
                    <span class="label">Image:</span>
                    <span class="value">{filename}</span>
                </div>
                {f'<div class="meta-row"><span class="label">Prompt:</span><span class="value meta-text prompt-text">{prompt_text}</span></div>' if prompt_text else ''}
            </div>
        </div>
        """)

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
                max-width: 1200px;
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
            .predictions {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 15px;
            }}
            .prediction-card {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            .card-header {{
                background: #f9f9f9;
                padding: 12px;
                border-bottom: 1px solid #e0e0e0;
                display: flex;
                justify-content: space-between;
                font-size: 0.9em;
            }}
            .asset-id {{
                font-weight: bold;
                color: #333;
            }}
            .timestamp {{
                color: #666;
                font-size: 0.85em;
            }}
            .card-body {{
                padding: 15px;
            }}
            .prediction-row {{
                display: flex;
                margin-bottom: 10px;
                align-items: center;
                gap: 10px;
            }}
            .meta-row {{
                margin-bottom: 8px;
                font-size: 0.9em;
            }}
            .label {{
                font-weight: 600;
                color: #555;
                min-width: 100px;
            }}
            .value {{
                flex: 1;
                padding: 6px 10px;
                background: #f5f5f5;
                border-radius: 4px;
                word-break: break-word;
            }}
            .predicted {{
                background: #ffe0e0;
                color: #d32f2f;
            }}
            .actual {{
                background: #e0ffe0;
                color: #388e3c;
            }}
            .confidence {{
                font-size: 0.85em;
                color: #666;
            }}
            .meta-text {{
                background: #f9f9f9;
                color: #666;
                font-family: monospace;
                font-size: 0.85em;
            }}
            .prompt-text {{
                display: block;
                max-height: 100px;
                overflow-y: auto;
            }}
            .image-container {{
                margin-bottom: 15px;
                text-align: center;
            }}

            .image-container img {{
                width: 100%;
                max-height: 350px;
                object-fit: contain;
                border-radius: 6px;
                border: 1px solid #ddd;
                background: white;
            }}
            .image-missing {{
                padding: 40px;
                color: #888;
                border: 1px dashed #ccc;
                border-radius: 6px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>VLM Wrong Predictions: {attribute}</h1>
            <div class="summary">
                <strong>Wrong assets:</strong> {wrong_preds["asset_id"].nunique()}<br>
                <strong>Total assets inspected:</strong> {wrong_preds["asset_id"].nunique() + (merged["asset_id"].nunique() - wrong_preds["asset_id"].nunique())}<br>
                <strong>Model:</strong> {model_name or 'unknown'}<br>
                <strong>Attribute:</strong> {attribute}
            </div>
            <div class="predictions">
                {"".join(rows_html)}
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
        help="Maximum number of wrong predictions to show",
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

    # Get wrong predictions
    wrong_preds = get_wrong_predictions(merged, pred_col, gt_col)

    if len(wrong_preds) == 0:
        print(f"No wrong predictions found for attribute: {args.attribute}")
        sys.exit(0)

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

    # Summary statistics
    print(f"\nPredicted values distribution:")
    print(wrong_preds[pred_col].value_counts().to_string())
    print(f"\nActual values distribution (for wrong predictions):")
    print(wrong_preds[gt_col].value_counts().to_string())

    # Group by prediction if requested
    if args.group_by_prediction:
        print(f"\n=== Errors grouped by prediction ===")
        for pred_val, group in wrong_preds.groupby(pred_col):
            actual_vals = group[gt_col].value_counts()
            print(f"\nPredicted '{pred_val}' (n={len(group)}):")
            for actual_val, count in actual_vals.items():
                print(f"  → Actually '{actual_val}': {count}")

    # Create HTML report
    create_error_report_html(
        wrong_preds,
        args.attribute,
        pred_col,
        gt_col,
        Path(args.output_dir),
        model_name=model_name,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
