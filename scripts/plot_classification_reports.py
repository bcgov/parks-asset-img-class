"""
Classification report heatmap generator.

Reads VLM predictions and ground truth CSVs, computes per-class
classification reports, and saves one image per asset type with
a heatmap subplot for each attribute.

Usage:
    python scripts/plot_classification_reports.py \
        --predictions results/vlm_stairs_gemini-3-flash_complete.csv \
        --ground_truth_dir data/processed/train \
        --attributes attr_structure_position attr_has_pedestrian_railing "attr_material_frame,_tank,_body" steps_bin fall_height_bin \
        --model gemini-3-flash-preview \
        --asset_type Stairs \
        --prompt_version stairs_v1 \
        --output_dir results/vlm_classification_reports
"""

import argparse
import os
import sys
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def ground_truth_path_for(attribute: str, ground_truth_dir: str) -> str:
    """Return the expected ground-truth CSV path for an attribute."""
    filename = (
        attribute
        .replace("attr_", "")
        .replace(",", "")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("<", "lt")
        .replace(">", "gt")
        .replace("/", "_")
    )
    if attribute.startswith("attr_"):
        return f"{ground_truth_dir}/attr_{filename}_train.csv"
    else:
        return f"{ground_truth_dir}/{filename}_train.csv"


def pred_col_for(attribute: str) -> str:
    """Return the prediction column name for an attribute."""
    attr_key = attribute.replace("attr_", "").replace(",", "")
    return f"{attr_key}_value"


def get_report_df(y_true, y_pred) -> pd.DataFrame:
    """Return classification_report as a DataFrame (classes x metrics)."""
    report = classification_report(
        y_true, y_pred,
        zero_division=0,
        output_dict=True
    )
    # drop summary rows
    drop_keys = {"accuracy", "macro avg", "weighted avg"}
    rows = {k: v for k, v in report.items() if k not in drop_keys}
    df = pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]]
    df.index.name = "class"
    return df


def load_attribute_data(attribute, predictions_df, ground_truth_dir):
    """Load predictions and ground truth for one classification-report panel."""
    gt_path = ground_truth_path_for(attribute, ground_truth_dir)
    if not os.path.exists(gt_path):
        print(f"  Warning: ground truth not found: {gt_path}", file=sys.stderr)
        return None

    gt_df = pd.read_csv(gt_path)
    if attribute not in gt_df.columns:
        print(f"  Warning: column '{attribute}' not in {gt_path}", file=sys.stderr)
        return None

    col = pred_col_for(attribute)
    if col not in predictions_df.columns:
        print(f"  Warning: prediction column '{col}' not in predictions file", file=sys.stderr)
        return None

    merged = predictions_df.merge(
        gt_df[["asset_id", attribute]],
        on="asset_id",
        how="inner"
    )
    merged = merged[merged[col].notna() & merged[attribute].notna()]
    merged = merged.drop_duplicates("asset_id")

    valid_labels = merged[attribute].dropna().unique().tolist()
    merged = merged[merged[col].isin(valid_labels)]

    if merged.empty:
        print(f"  Warning: no valid rows after merge for '{attribute}'", file=sys.stderr)
        return None

    y_true = merged[attribute].tolist()
    y_pred = merged[col].tolist()
    return y_true, y_pred


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_report_heatmap(ax, report_df, title):
    """Plot a heatmap from a sklearn classification report table."""
    metrics = ["precision", "recall", "f1-score"]
    data = report_df[metrics].values.astype(float)
    classes = report_df.index.tolist()
    support = report_df["support"].astype(int).tolist()

    n_classes = len(classes)
    n_metrics = len(metrics)

    cmap = plt.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=0, vmax=1)

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    # axis labels
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metrics, fontsize=8, fontweight="medium")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(
        [f"{c}\n(n={s})" for c, s in zip(classes, support)],
        fontsize=7,
    )

    # cell annotations
    for row in range(n_classes):
        for col_idx in range(n_metrics):
            val = data[row, col_idx]
            text_color = "black" if 0.3 < val < 0.75 else "white"
            ax.text(
                col_idx, row,
                f"{val:.2f}",
                ha="center", va="center",
                fontsize=7.5,
                color=text_color,
                fontweight="medium",
            )

    ax.set_title(title, fontsize=9, pad=6, fontweight="medium")
    ax.tick_params(axis="both", length=0)

    # thin grid lines between cells
    ax.set_xticks(np.arange(-0.5, n_metrics, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    return im


def plot_asset_type(
    attribute_data: dict,   # {attribute: (y_true, y_pred)}
    asset_type: str,
    model_name: str,
    prompt_version: str,
    output_dir: str,
    dpi: int,
    max_cols: int,
):
    """Plot all requested attribute reports for one asset type."""
    n = len(attribute_data)
    if n == 0:
        print("No valid attributes to plot.", file=sys.stderr)
        return

    n_cols = min(n, max_cols)
    n_rows = math.ceil(n / n_cols)

    # dynamically size figure based on class counts
    row_heights = []
    for attr, (y_true, _) in attribute_data.items():
        n_classes = len(set(y_true))
        row_heights.append(max(2.5, n_classes * 0.55))

    # each subplot column is ~3.8in wide
    fig_w = n_cols * 5.0
    # height per row = max subplot height in that row
    fig_h = 0
    for row_idx in range(n_rows):
        attrs_in_row = list(attribute_data.items())[row_idx * n_cols: (row_idx + 1) * n_cols]
        fig_h += max(
            max(2.5, len(set(y_true)) * 0.55)
            for _, (y_true, _) in attrs_in_row
        ) + 1.0   # padding for title

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h + 1.2), constrained_layout=True)

    # normalise axes to 2-D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    last_im = None
    for idx, (attribute, (y_true, y_pred)) in enumerate(attribute_data.items()):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row_idx, col_idx]

        report_df = get_report_df(y_true, y_pred)

        label = attribute.replace("attr_", "").replace("_", " ")
        support = report_df["support"]
        weighted_f1 = (report_df["f1-score"] * support).sum() / support.sum()
        title = f"{label}  ·  weighted F1={weighted_f1:.2f}"

        last_im = plot_report_heatmap(ax, report_df, title)

    # hide unused subplots
    for idx in range(n, n_rows * n_cols):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        axes[row_idx, col_idx].set_visible(False)

    # shared colorbar
    cbar = fig.colorbar(last_im, ax=axes, orientation="vertical", fraction=0.015, pad=0.02)
    cbar.set_label("score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # fig.suptitle(
    #     f"{asset_type}  ·  {model_name}  ·  {prompt_version}",
    #     fontsize=11,
    #     fontweight="medium",
    #     y=1.01,
    # )

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        output_dir,
        f"classification_report_{asset_type}_{model_name}_{prompt_version}_{ts}.png"
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot per-attribute classification report heatmaps for a given asset type."
    )
    parser.add_argument("--predictions", required=True,
                        help="Path to VLM predictions CSV")
    parser.add_argument("--ground_truth_dir", required=True,
                        help="Directory containing attribute ground truth CSVs")
    parser.add_argument("--attributes", required=True, nargs="+",
                        help="Attributes to evaluate (e.g. attr_structure_position steps_bin)")
    parser.add_argument("--model", required=True,
                        help="Model name (for labelling)")
    parser.add_argument("--asset_type", default="unknown",
                        help="Asset type (e.g. Stairs)")
    parser.add_argument("--prompt_version", default="v1",
                        help="Prompt version (for labelling)")
    parser.add_argument("--output_dir", default="results/vlm_classification_reports",
                        help="Directory to save output images")
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--max_cols", type=int, default=3,
                        help="Max subplot columns per row (default: 3)")

    args = parser.parse_args()

    predictions_df = pd.read_csv(args.predictions)
    if "asset_id" not in predictions_df.columns:
        print("Error: predictions CSV must contain an 'asset_id' column.", file=sys.stderr)
        sys.exit(1)

    attribute_data = {}
    for attribute in args.attributes:
        print(f"Loading: {attribute}")
        result = load_attribute_data(attribute, predictions_df, args.ground_truth_dir)
        if result is not None:
            attribute_data[attribute] = result

    plot_asset_type(
        attribute_data=attribute_data,
        asset_type=args.asset_type,
        model_name=args.model,
        prompt_version=args.prompt_version,
        output_dir=args.output_dir,
        dpi=args.dpi,
        max_cols=args.max_cols,
    )