"""Model evaluation results bar chart generator — DINOv3 multi-classifier edition.

Supports VLM eval results (original) and DINOv3 classifier results from
subdirectories of a results root (dinov3_gradient_boost, dinov3_knn_cv,
dinov3_linear_svm, dinov3_logistic).

Usage — VLM (original):
    python scripts/plot_eval_results.py \
        --input_dir results/vlm_eval_results/ \
        --asset_type Stairs \
        --output results/eval_results_plots/stairs_vlm_comparison_macro-f1.png

Usage — DINOv3 classifiers:
    python scripts/plot_eval_results.py \
        --dinov3_dir results/dinov3_results \
        --output results/eval_results_plots/dinov3_comparison_macro_f1.png \
        --title "DINOv3 Classifiers"
    
Usage — DINOv3 vs SigLIP logistic regression only
    python scripts/plot_eval_results.py \
        --dinov3_dir results/dinov3_results \
        --siglip-dir results/siglip_results \
        --include-series "DINOv3 + Logistic Regression" "SigLIP + Logistic Regression" \
        --output results/eval_results_plots/logistic_comparison_macro_f1.png \
        --title "DINOv3 vs. SigLIP"

Usage — All DINOv3 classifiers
    python scripts/plot_eval_results.py \
        --dinov3_dir results/dinov3_results \
        --output results/eval_results_plots/dinov3_only.png
    
Usage — DINOv3 + logistic regression, gemini-3-flash-preview
    python scripts/plot_eval_results.py \
        --dinov3_dir results/dinov3_results \
        --input_dir results/vlm_eval_results/ \
        --include-series \
            "DINOv3 + Logistic Regression" \
            "gemini-3-flash-preview"'\n'"(attribute-specific prompt)"
    --output results/eval_results_plots/gemini_vs_dinov3.png
    
Usage — Aggregate all attributes into one bar per series
    python scripts/plot_eval_results.py \
        --dinov3_dir results/dinov3_results \
        --aggregate \
        --title "DINOv3 classifiers — mean across all attributes" \
        --output results/eval_results_plots/dinov3_aggregate_macro_f1.png \
        --figsize 7 7

Usage — compare gemini vs dinov3 aggregated
    python scripts/plot_eval_results.py \
        --dinov3_dir results/dinov3_results \
        --input_dir results/vlm_eval_results/ \
        --aggregate \
        --include-series "DINOv3 + Logistic Regression" "gemini-3-flash-preview"$'\n'"(attribute-specific prompt)" \
        --output results/eval_results_plots/gemini_vs_dinov3_aggregate.png \
        --figsize 4 6
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

METRICS = ["macro_f1", "weighted_f1"]

VLM_COLORS = [
    "#0B4C8D",  # blue
    "#79BCFF",  # light blue
    "#BD431A",  # coral
    "#FF9470",  # light coral
    "#BA7517",  # amber
    "#639922",  # green
    "#7F77DD",  # purple
    "#888780",  # gray
]

DINOV3_COLORS = [
    "#1D9E75",  # teal
    "#BA7517",  # amber
    "#7F77DD",  # purple
    "#D85A30",  # coral
    "#0B4C8D",  # blue
]

SIGLIP_COLORS = [
    "#C14DB3",  # magenta
    "#7B2FA8",  # violet
    "#E8853D",  # orange
    "#4A90D9",  # sky blue
    "#C2396E",  # rose
]


DINOV3_SUBDIRS = [
    "dinov3_gradient_boost",
    "dinov3_knn_cv",
    "dinov3_linear_svm",
    "dinov3_logistic",
    "dinov3_random_forest"
]

DINOV3_LABELS = {
    "dinov3_gradient_boost": "DINOv3 + Gradient Boost",
    "dinov3_knn_cv": "DINOv3 + k-NN (k=10)",
    "dinov3_linear_svm": "DINOv3 + Linear SVM",
    "dinov3_logistic": "DINOv3 + Logistic Regression",
    "dinov3_random_forest": "DINOv3 + Random Forest"
}

SIGLIP_SUBDIRS = [
    "siglip_logistic_reg",
]

SIGLIP_LABELS = {
    "siglip_gradient_boost": "SigLIP + Gradient Boost",
    "siglip_knn_cv": "SigLIP + k-NN (k=10)",
    "siglip_linear_svm": "SigLIP + Linear SVM",
    "siglip_logistic_reg": "SigLIP + Logistic Regression",
    "siglip_random_forest": "SigLIP + Random Forest",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalise_attribute(attr: str) -> str:
    """Normalise attribute names to a canonical underscore form."""
    return attr.strip().lower().replace(",", "").replace(" ", "_")


def _scalar(df: pd.DataFrame, attr: str, col: str) -> float:
    if attr not in df.index:
        return np.nan
    val = df.loc[attr, col]
    if hasattr(val, "iloc"):
        return float(val.iloc[0])
    return float(val)


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

def load_results(input_dir: str, individual_prompt_label: str = "attribute-specific prompt") -> pd.DataFrame:
    pattern = os.path.join(input_dir, "*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"No CSV files found in: {input_dir}", file=sys.stderr)
        sys.exit(1)

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["attribute"] = df["attribute"].astype(str).map(normalise_attribute)
            if df["attribute"].nunique() == 1:
                df["prompt_version"] = individual_prompt_label
            frames.append(df)
        except Exception as e:
            print(f"Warning: could not read {f}: {e}", file=sys.stderr)

    if not frames:
        print("No valid CSV files could be loaded.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    # drop any duplicate columns
    combined = combined.loc[:, ~combined.columns.duplicated()]

    required = {"attribute", "model", "asset_type", "prompt_version", "macro_f1", "weighted_f1"}
    missing = required - set(combined.columns)
    if missing:
        print(f"Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    if "timestamp" in combined.columns:
        combined = (
            combined
            .sort_values("timestamp")
            .drop_duplicates(
                subset=["attribute", "model", "asset_type", "prompt_version"],
                keep="last",
            )
        )

    # drop attributes where macro_f1 and weighted_f1 are all NaN (numeric targets)
    combined = combined[combined["macro_f1"].notna() & combined["weighted_f1"].notna()]

    return combined


def load_embedding_results(results_dir: str, metric: str, subdirs: list, labels: dict) -> pd.DataFrame:
    """Generic loader for any embedding classifier results directory."""
    mean_col = f"{metric}_mean"
    std_col  = f"{metric}_std"

    frames = []
    for subdir in subdirs:
        subdir_path = os.path.join(results_dir, subdir)
        if not os.path.isdir(subdir_path):
            print(f"  [skip] directory not found: {subdir_path}", file=sys.stderr)
            continue

        csv_files = glob.glob(os.path.join(subdir_path, "*.csv"))
        if not csv_files:
            print(f"  [skip] no CSVs in: {subdir_path}", file=sys.stderr)
            continue

        for f in csv_files:
            if os.path.basename(f).startswith("knn_summary"):
                continue
            try:
                df = pd.read_csv(f)

                if "attribute" not in df.columns:
                    if "target_column" in df.columns:
                        df["attribute"] = df["target_column"]
                    else:
                        print(f"  [skip] no attribute or target_column in {f}", file=sys.stderr)
                        continue

                df["attribute"] = df["attribute"].astype(str).map(normalise_attribute)

                if mean_col not in df.columns or std_col not in df.columns:
                    continue

                df = df[df["macro_f1_mean"].notna() & df["weighted_f1_mean"].notna()]
                if df.empty:
                    continue

                row = df[["attribute", "macro_f1_mean", "macro_f1_std",
                           "weighted_f1_mean", "weighted_f1_std"]].copy().reset_index(drop=True)
                row["series"] = labels.get(subdir, subdir)
                frames.append(row)
            except Exception as e:
                print(f"Warning: could not read {f}: {e}", file=sys.stderr)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined
        .groupby(["attribute", "series"], as_index=False)
        [["macro_f1_mean", "macro_f1_std", "weighted_f1_mean", "weighted_f1_std"]]
        .mean()
    )
    return combined


def load_dinov3_results(dinov3_dir: str, metric: str) -> pd.DataFrame:
    return load_embedding_results(dinov3_dir, metric, DINOV3_SUBDIRS, DINOV3_LABELS)


def load_siglip_results(siglip_dir: str, metric: str) -> pd.DataFrame:
    return load_embedding_results(siglip_dir, metric, SIGLIP_SUBDIRS, SIGLIP_LABELS)

def load_baseline(baseline_path: str) -> pd.DataFrame:
    df = pd.read_csv(baseline_path)
    if "target_column" in df.columns:
        df["attribute"] = df["target_column"]
    df["attribute"] = df["attribute"].astype(str).map(normalise_attribute)
    # keep only the two strategies we want to plot
    strategies = ["majority_class_group_cv", "uniform_random_group_cv"]
    df = df[df["strategy"].isin(strategies)]
    return df[["attribute", "strategy", "macro_f1_mean", "weighted_f1_mean"]]


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_comparison(
    df: pd.DataFrame | None,
    metric: str,
    asset_type: str | None,
    output_path: str,
    dpi: int,
    figsize: tuple[float, float],
    title: str | None,
    baseline_df: pd.DataFrame | None,
    dinov3_df: pd.DataFrame | None,
    siglip_df: pd.DataFrame | None,
    include_series: list[str] | None = None,
    aggregate: bool = False,
):
    mean_col = f"{metric}_mean"
    std_col  = f"{metric}_std"

    vlm_rows = pd.DataFrame()
    if df is not None:
        if asset_type:
            df = df[df["asset_type"] == asset_type]
        df = df.copy().reset_index(drop=True)
        # drop any duplicate columns that crept in during concat
        df = df.loc[:, ~df.columns.duplicated()]
        df["series"] = df["model"] + "\n(" + df["prompt_version"] + ")"
        df[mean_col] = df[metric]
        df[std_col]  = 0.0
        vlm_rows = df[["attribute", "series", mean_col, std_col]].reset_index(drop=True)

    dino_rows = pd.DataFrame()
    if dinov3_df is not None and not dinov3_df.empty:
        dino_rows = dinov3_df[["attribute", "series", mean_col, std_col]].copy().reset_index(drop=True)

    siglip_rows = pd.DataFrame()
    if siglip_df is not None and not siglip_df.empty:
        siglip_rows = siglip_df[["attribute", "series", mean_col, std_col]].copy().reset_index(drop=True)

    all_rows = pd.concat([vlm_rows, dino_rows, siglip_rows], ignore_index=True)
    
    if include_series:
        all_rows = all_rows[all_rows["series"].isin(include_series)]
        if all_rows.empty:
            print(
                f"Error: none of the requested series were found. "
                f"Available: {sorted(all_rows['series'].unique().tolist())}",
                file=sys.stderr,
            )
            sys.exit(1)

    if all_rows.empty:
        print("No data to plot.", file=sys.stderr)
        sys.exit(1)

    # drop any attributes where ALL series have NaN (numeric targets that slipped through)
    valid_attrs = (
        all_rows.groupby("attribute")[mean_col]
        .apply(lambda x: x.notna().any())
    )
    valid_attrs = valid_attrs[valid_attrs].index.tolist()
    all_rows = all_rows[all_rows["attribute"].isin(valid_attrs)]
    
    if aggregate:
        all_rows = (
            all_rows.groupby("series", as_index=False)[mean_col]
            .mean()
        )
        all_rows["attribute"] = "mean (all attributes)"

    attributes = sorted(all_rows["attribute"].unique())
    series = sorted(all_rows["series"].unique())

    n_attrs  = len(attributes)
    n_series = len(series)

    bar_width = 0.8 / n_series
    x = np.arange(n_attrs)

    fig, ax = plt.subplots(figsize=figsize)

    vlm_i    = 0
    dino_i   = 0
    siglip_i = 0

    dinov3_series = set(DINOV3_LABELS.values())
    siglip_series = set(SIGLIP_LABELS.values())

    for i, s in enumerate(series):
        if s in dinov3_series:
            color = DINOV3_COLORS[dino_i % len(DINOV3_COLORS)]
            dino_i += 1
        elif s in siglip_series:
            color = SIGLIP_COLORS[siglip_i % len(SIGLIP_COLORS)]
            siglip_i += 1
        else:
            color = VLM_COLORS[vlm_i % len(VLM_COLORS)]
            vlm_i += 1

        s_df = all_rows[all_rows["series"] == s].set_index("attribute")
        values = [_scalar(s_df, attr, mean_col) for attr in attributes]

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

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                is_high = val > 1
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() - 0.03 if is_high else bar.get_height() + 0.012,
                    f"{val:.2f}",
                    ha="center",
                    va="top" if is_high else "bottom",
                    fontsize=7,
                    color="white" if is_high else "#444441",
                )

    # --- baseline reference lines ---
    BASELINE_STYLES = {
        "majority_class_group_cv":  {"color": "#444441", "linestyle": "--",  "label": "baseline (majority class)"},
        "uniform_random_group_cv":  {"color": "#E0E0E0", "linestyle": ":",   "label": "baseline (uniform random)"},
    }

    if baseline_df is not None:
        baseline_metric_col = f"{metric}_mean"
        legend_added = set()
        if aggregate:
            # compute mean baseline value across all attributes per strategy
            for strategy, style in BASELINE_STYLES.items():
                strategy_rows = baseline_df[baseline_df["strategy"] == strategy]
                if strategy_rows.empty:
                    continue
                val = strategy_rows[baseline_metric_col].mean()
                ax.plot(
                    [-0.4, 0.4],
                    [val, val],
                    color=style["color"],
                    linewidth=2,
                    linestyle=style["linestyle"],
                    zorder=5,
                )
                ax.plot([], [],
                        color=style["color"],
                        linewidth=2,
                        linestyle=style["linestyle"],
                        label=style["label"])
        else:
            for j, attr in enumerate(attributes):
                attr_rows = baseline_df[baseline_df["attribute"] == attr]
                for _, row in attr_rows.iterrows():
                    strategy = row["strategy"]
                    style = BASELINE_STYLES.get(strategy)
                    if style is None:
                        continue
                    val = row[baseline_metric_col]
                    half = 0.4
                    ax.plot(
                        [j - half, j + half],
                        [val, val],
                        color=style["color"],
                        linewidth=2,
                        linestyle=style["linestyle"],
                        zorder=5,
                    )
                    if strategy not in legend_added:
                        ax.plot([], [],
                                color=style["color"],
                                linewidth=2,
                                linestyle=style["linestyle"],
                                label=style["label"])
                        legend_added.add(strategy)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [a.replace("attr_", "").replace("_", "\n") for a in attributes],
        fontsize=9,
    )
    ax.set_ylabel(metric, fontsize=10)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))

    if title is None:
        title = f"Attribute evaluation — {metric}"
    else:
        title = f"{title} — {metric}"
    if asset_type:
        title += f"  ·  {asset_type}"
    ax.set_title(title, fontsize=12, pad=14, fontweight="medium")

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#D3D1C7", zorder=0)
    ax.xaxis.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.5)

    ax.legend(
        title="model  (prompt)" if df is not None else "classifier",
        title_fontsize=8,
        fontsize=8,
        loc="lower right",
        bbox_to_anchor=(1, 0.75),
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
    parser = argparse.ArgumentParser(description="Plot model eval results as a grouped bar chart.")

    parser.add_argument("--input_dir", default=None,
                        help="Directory of VLM eval CSV files (optional).")
    parser.add_argument("--dinov3_dir", default=None,
                        help="Root directory containing dinov3_* subdirectories (optional).")
    parser.add_argument("--siglip-dir", default=None,
                        help="Root directory containing siglip_* subdirectories (optional).")
    parser.add_argument("--metric", choices=METRICS, default="macro_f1")
    parser.add_argument("--asset_type", default=None)
    parser.add_argument("--output", default="comparison.png")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--figsize", nargs=2, type=float, metavar=("W", "H"), default=[20, 7])
    parser.add_argument(
        "--title",
        default=None,
        help="Custom plot title. Defaults to 'Attribute evaluation — {metric}'.",
    )
    parser.add_argument("--baseline", default="results/baseline_results/baseline_classification_results.csv")
    parser.add_argument("--individual_prompt_label", default="attribute-specific prompt")
    parser.add_argument(
        "--include-series",
        nargs="+",
        default=None,
        help=(
            "Only plot these series labels. Use the exact label names, e.g.: "
            "'DINOv3 + Logistic Regression' 'SigLIP + Logistic Regression'"
        ),
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Plot mean performance across all attributes instead of per-attribute bars.",
    )

    args = parser.parse_args()

    if not args.input_dir and not args.dinov3_dir and not args.siglip_dir:
        print("Error: provide at least one of --input_dir, --dinov3_dir, or --siglip-dir", file=sys.stderr)
        sys.exit(1)

    vlm_df = load_results(args.input_dir, args.individual_prompt_label) if args.input_dir else None
    dinov3_df = load_dinov3_results(args.dinov3_dir, args.metric) if args.dinov3_dir else None
    siglip_df = load_siglip_results(args.siglip_dir, args.metric) if args.siglip_dir else None
    baseline_df = load_baseline(args.baseline) if args.baseline and os.path.exists(args.baseline) else None

    plot_comparison(
        df=vlm_df,
        metric=args.metric,
        asset_type=args.asset_type,
        output_path=args.output,
        dpi=args.dpi,
        figsize=tuple(args.figsize),
        title=args.title,
        baseline_df=baseline_df,
        dinov3_df=dinov3_df,
        siglip_df=siglip_df,
        include_series=args.include_series,
        aggregate=args.aggregate,
    )