"""
Model evaluation results bar chart generator.

Usage:
    python scripts/plot_eval_results.py \
        --input_dir results/vlm_eval_results/ \
        --asset_type Stairs \
        --output results/eval_results_plots/stairs_vlm_comparison_macro-f1.png
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

METRICS = ["macro_f1", "weighted_f1"]

COLORS = [
    "#378ADD",  # blue
    "#1D9E75",  # teal
    "#D85A30",  # coral
    "#D4537E",  # pink
    "#BA7517",  # amber
    "#639922",  # green
    "#7F77DD",  # purple
    "#888780",  # gray
]


# ---------------------------------------------------------------------
# Load and merge CSVs
# ---------------------------------------------------------------------

def load_results(input_dir: str) -> pd.DataFrame:
    pattern = os.path.join(input_dir, "*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"No CSV files found in: {input_dir}", file=sys.stderr)
        sys.exit(1)

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception as e:
            print(f"Warning: could not read {f}: {e}", file=sys.stderr)

    if not frames:
        print("No valid CSV files could be loaded.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    required = {"attribute", "model", "asset_type", "prompt_version", "macro_f1", "weighted_f1"}
    missing = required - set(combined.columns)
    if missing:
        print(f"Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # deduplicate: keep latest timestamp per (attribute, model, asset_type, prompt_version)
    if "timestamp" in combined.columns:
        combined = (
            combined
            .sort_values("timestamp")
            .drop_duplicates(
                subset=["attribute", "model", "asset_type", "prompt_version"],
                keep="last"
            )
        )

    return combined

# load baseline separately
def load_baseline(baseline_path: str) -> pd.DataFrame:
    df = pd.read_csv(baseline_path)
    # normalize attribute name to match eval results
    # e.g. "attr_material_frame_tank_body" -> "attr_material_frame,_tank,_body"
    if "target_column" in df.columns:
        df["attribute"] = df["target_column"]
    return df[["attribute", "macro_f1_mean", "weighted_f1_mean"]]

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_comparison(
    df: pd.DataFrame,
    metric: str,
    asset_type: str | None,
    output_path: str,
    dpi: int,
    figsize: tuple[float, float],
    baseline_df: pd.DataFrame
):
    if asset_type:
        df = df[df["asset_type"] == asset_type]
        if df.empty:
            print(f"No data found for asset_type='{asset_type}'", file=sys.stderr)
            sys.exit(1)

    # group key: model + prompt_version
    df = df.copy()
    df["series"] = df["model"] + "\n(" + df["prompt_version"] + ")"

    attributes = sorted(df["attribute"].unique())
    series = sorted(df["series"].unique())

    n_attrs = len(attributes)
    n_series = len(series)

    bar_width = 0.8 / n_series
    x = np.arange(n_attrs)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=figsize)

    for i, s in enumerate(series):
        color = COLORS[i % len(COLORS)]
        s_df = df[df["series"] == s].set_index("attribute")

        values = [
            s_df.loc[attr, metric] if attr in s_df.index else np.nan
            for attr in attributes
        ]

        offset = (i - (n_series - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width * 0.92,
            label=s,
            color=color,
            alpha=0.90,
            zorder=3,
        )

        # value labels on top of bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                is_high = val > 0.92
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() - 0.03 if is_high else bar.get_height() + 0.012,
                    f"{val:.2f}",
                    ha="center",
                    va="top" if is_high else "bottom",
                    fontsize=7,
                    color="white" if is_high else "#444441",
                )
                
    # ---- baseline reference lines ----
    if baseline_df is not None:
        baseline_metric_col = f"{metric}_mean"  # macro_f1_mean or weighted_f1_mean
        for j, attr in enumerate(attributes):
            row = baseline_df[baseline_df["attribute"] == attr]
            if row.empty:
                continue
            val = row[baseline_metric_col].iloc[0]
            half = 0.4  # half-width of the line segment
            ax.plot(
                [j - half, j + half],
                [val, val],
                color="#222",
                linewidth=1.5,
                linestyle="--",
                zorder=5,
            )

        # add to legend once
        ax.plot([], [], color="#222", linewidth=1.5, linestyle="--", label="baseline (majority class)")

    # ---- axes ----
    ax.set_xticks(x)
    ax.set_xticklabels(
        [a.replace("attr_", "").replace("_", "\n") for a in attributes],
        fontsize=9,
    )
    ax.set_ylabel(metric, fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))

    title_parts = [f"VLM attribute evaluation — {metric}"]
    if asset_type:
        title_parts[0] += f"  ·  {asset_type}"
    ax.set_title(title_parts[0], fontsize=12, pad=14, fontweight="medium")

    # ---- grid ----
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#D3D1C7", zorder=0)
    ax.xaxis.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)

    # ---- legend ----
    ax.legend(
        title="model  (prompt)",
        title_fontsize=8,
        fontsize=8,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor="#D3D1C7",
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved chart to: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot VLM eval results as a grouped bar chart.")

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing VLM eval CSV files (e.g. results/vlm_eval_results/)",
    )
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="macro_f1",
        help="F1 metric to plot (default: macro_f1)",
    )
    parser.add_argument(
        "--asset_type",
        default=None,
        help="Filter to a specific asset type (e.g. Stairs). Plots all if omitted.",
    )
    parser.add_argument(
        "--output",
        default="vlm_comparison.png",
        help="Output file path (default: vlm_comparison.png). Supports .png, .pdf, .svg.",
    )
    parser.add_argument("--dpi", type=int, default=120, help="Output DPI (default: 120)")
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("W", "H"),
        default=[12, 5],
        help="Figure size in inches, e.g. --figsize 14 6 (default: 12 5)",
    )
    parser.add_argument(
        "--baseline",
        default="results/baseline_classification_results.csv",
        help="Path to baseline CV results CSV (optional)",
    )

    args = parser.parse_args()
    
    baseline_df = load_baseline(args.baseline) if args.baseline else None
    df = load_results(args.input_dir)
    plot_comparison(
        df=df,
        metric=args.metric,
        asset_type=args.asset_type,
        output_path=args.output,
        dpi=args.dpi,
        figsize=tuple(args.figsize),
        baseline_df=baseline_df
    )