"""Plot per-attribute test-set results.

Reads the test-set evaluation summary (one row per attribute, as written by
scripts/evaluate_test_set.py) and produces a per-attribute bar chart of
weighted F1 (and optionally macro F1 alongside it, which surfaces rare-class
behaviour). Attributes are grouped into categorical and measurement
(binned) blocks so the headline pattern -- strong on categorical attributes,
weaker on measurement attributes -- is visible at a glance.

Usage:
    python scripts/plot_test_results.py \
        --input results/final/test_set_results.csv \
        --output reports/report_images/test_set_results.png

    # weighted F1 only
    python scripts/plot_test_results.py --metric weighted_f1 --no-macro
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Binned / measurement attributes (everything else is treated as categorical).
MEASUREMENT_ATTRS = {"length_bin", "width_bin", "fall_height_bin", "steps_bin"}

# Colours kept consistent with the other report figures.
WEIGHTED_COLOR = "#0B4C8D"     # blue (categorical bars)
MEASUREMENT_COLOR = "#BD431A"  # coral/red (measurement bars)
MACRO_COLOR = "#FF9470"        # light coral (macro F1, when shown)
CATEGORICAL_BAND = "#F2F6FA"
MEASUREMENT_BAND = "#FBF1EC"


def prettify(attr: str) -> str:
    return attr.replace("attr_", "").replace("_", " ")


def normalise(attr: str) -> str:
    """Canonical form for matching attribute names across files."""
    return str(attr).strip().lower().replace(",", "").replace(" ", "_")


# Baseline reference-line styles (cross-validation baselines).
BASELINE_STYLES = {
    "majority_class_group_cv": {"color": "#444441", "linestyle": "--",
                                "label": "baseline (majority class)"},
    "uniform_random_group_cv": {"color": "#9A9A9A", "linestyle": ":",
                                "label": "baseline (uniform random)"},
}


def load_baseline(path: Path, metric: str) -> pd.DataFrame:
    """Load per-attribute baseline values, normalised for name matching."""
    df = pd.read_csv(path)
    if "target_column" in df.columns:
        df["attribute"] = df["target_column"]
    df["attribute"] = df["attribute"].map(normalise)
    df = df[df["strategy"].isin(BASELINE_STYLES)]
    col = f"{metric}_mean"
    return df[["attribute", "strategy", col]].rename(columns={col: "value"})


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"attribute", "weighted_f1", "macro_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing columns: {missing}. Found: {list(df.columns)}")
    # group: categorical first, then measurement; sort within group by weighted F1
    df["group"] = np.where(df["attribute"].isin(MEASUREMENT_ATTRS),
                           "measurement", "categorical")
    df = df.sort_values(
        ["group", "weighted_f1"], ascending=[True, False]
    ).reset_index(drop=True)
    return df


def plot(df: pd.DataFrame, args: argparse.Namespace) -> None:
    attributes = df["attribute"].tolist()
    labels = [prettify(a) for a in attributes]
    x = np.arange(len(attributes))

    show_macro = args.macro
    bar_w = 0.38 if show_macro else 0.6

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    # shade the categorical vs measurement regions
    n_categorical = int((df["group"] == "categorical").sum())
    if 0 < n_categorical < len(df):
        ax.axvspan(-0.5, n_categorical - 0.5, color=CATEGORICAL_BAND, zorder=0)
        ax.axvspan(n_categorical - 0.5, len(df) - 0.5, color=MEASUREMENT_BAND, zorder=0)

    if show_macro:
        w_bars = ax.bar(x - bar_w / 2, df["weighted_f1"], bar_w,
                        label="weighted F1", color=WEIGHTED_COLOR, zorder=3)
        m_bars = ax.bar(x + bar_w / 2, df["macro_f1"], bar_w,
                        label="macro F1", color=MACRO_COLOR, zorder=3)
        bar_groups = [w_bars, m_bars]
    else:
        bar_colors = [
            MEASUREMENT_COLOR if attr in MEASUREMENT_ATTRS else WEIGHTED_COLOR
            for attr in attributes
        ]
        w_bars = ax.bar(x, df[args.metric], bar_w,
                        color=bar_colors, zorder=3)
        bar_groups = [w_bars]

    for bars in bar_groups:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                        f"{h:.2f}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="medium", color="#333333")

    # --- baseline reference lines (per attribute, from cross-validation) ---
    if args.baseline is not None and Path(args.baseline).exists():
        baseline = load_baseline(Path(args.baseline), args.metric)
        norm_attrs = [normalise(a) for a in attributes]
        legend_added = set()
        for j, attr_norm in enumerate(norm_attrs):
            rows = baseline[baseline["attribute"] == attr_norm]
            for _, row in rows.iterrows():
                style = BASELINE_STYLES.get(row["strategy"])
                if style is None or np.isnan(row["value"]):
                    continue
                ax.plot([j - 0.45, j + 0.45], [row["value"], row["value"]],
                        color=style["color"], linewidth=1.8,
                        linestyle=style["linestyle"], zorder=4)
                if row["strategy"] not in legend_added:
                    ax.plot([], [], color=style["color"], linewidth=1.8,
                            linestyle=style["linestyle"], label=style["label"])
                    legend_added.add(row["strategy"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("F1 score", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    title = args.title or "Test-set performance by attribute (DINOv3 + Logistic Regression)"
    ax.set_title(title, fontsize=14, fontweight="semibold", pad=12)

    # group annotations
    if 0 < n_categorical < len(df):
        ax.text((n_categorical - 1) / 2, 1.04, "categorical",
                ha="center", va="bottom", fontsize=10, color="#0B4C8D",
                fontweight="semibold")
        ax.text((n_categorical + len(df) - 1) / 2, 1.04, "measurement",
                ha="center", va="bottom", fontsize=10, color="#BD431A",
                fontweight="semibold")

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#D3D1C7", zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

    # build legend: color key (categorical/measurement) + any baseline lines
    from matplotlib.patches import Patch
    if show_macro:
        handles, _ = ax.get_legend_handles_labels()
    else:
        handles = [
            Patch(color=WEIGHTED_COLOR, label="categorical"),
            Patch(color=MEASUREMENT_COLOR, label="measurement"),
        ]
        # append baseline line handles (added via ax.plot([], [], label=...))
        line_handles, line_labels = ax.get_legend_handles_labels()
        handles += [h for h, lab in zip(line_handles, line_labels)
                    if lab.startswith("baseline")]
    ax.legend(handles=handles, fontsize=9, loc="upper right",
              frameon=True, framealpha=0.9)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=Path("results/final/test_set_results.csv"))
    p.add_argument("--output", type=Path,
                   default=Path("reports/report_images/test_set_results.png"))
    p.add_argument("--metric", default="weighted_f1",
                   choices=["weighted_f1", "macro_f1"],
                   help="Metric to plot when --no-macro is set.")
    p.add_argument("--macro", action="store_true",
                   help="Also plot macro F1 alongside weighted F1 (off by default).")
    p.add_argument("--title", default=None)
    p.add_argument("--figsize", nargs=2, type=float, default=[12, 6],
                   metavar=("W", "H"))
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--baseline", type=Path,
                   default=Path("results/baseline_results/baseline_classification_results.csv"),
                   help="Baseline results CSV for reference lines (cross-validation). "
                        "Pass an empty/nonexistent path to omit baseline lines.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"Input not found: {args.input}")
        return 1
    df = load_results(args.input)
    plot(df, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())