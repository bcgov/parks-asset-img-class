"""Generate the exploratory data analysis (EDA) figures used in the final report.

Produces three figures from the raw CityWide attribute manifest:

1. attribute_fill_rate.png       - per-asset-category attribute fill rates
2. categorical_distributions.png - class distributions for categorical attributes
3. numerical_distributions.png   - histograms for numeric attributes

The raw manifest is large and is not committed to the repository, so this script
reads it from --input (default: data/raw/citywide/image_attributes_manifest.csv).
If the file is absent, the script exits with a clear message rather than a
traceback, and the report can still render against the committed PNGs.

Usage:
    python scripts/generate_eda_figures.py
    python scripts/generate_eda_figures.py \
        --input data/raw/citywide/image_attributes_manifest.csv \
        --output-dir reports/report_images
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BAR_COLOR = "#4a90d9"

# Attribute coverage per asset category (drives the fill-rate figure).
PROFILE_ATTRS: dict[str, dict[str, object]] = {
    "Boardwalk < 1.2m High": {
        "profile_id": 337,
        "attrs": [
            "attr_Decking Material",
            "attr_Fall Height",
            "attr_Has Edge Guard",
            "attr_Has Pedestrian Railing",
            "attr_Length",
            "attr_Structure Material",
            "attr_Width",
        ],
    },
    "Boardwalk > 1.2m High": {
        "profile_id": 573,
        "attrs": [
            "attr_Decking Material",
            "attr_Fall Height",
            "attr_Has Edge Guard",
            "attr_Has Pedestrian Railing",
            "attr_Length",
            "attr_Structure Material",
            "attr_Width",
        ],
    },
    "Stairs": {
        "profile_id": 356,
        "attrs": [
            "attr_Fall Height",
            "attr_Has Pedestrian Railing",
            "attr_Material (Frame, Tank, Body)",
            "attr_Number of Steps",
            "attr_Structure Position",
        ],
    },
    "Trail Bridge": {
        "profile_id": 253,
        "attrs": [
            "attr_Abutment Material",
            "attr_Bridge Type",
            "attr_Decking Material",
            "attr_Fall Height",
            "attr_Has Pedestrian Railing",
            "attr_Length",
            "attr_Structure Material",
            "attr_Width",
        ],
    },
    "Viewing Platform": {
        "profile_id": 359,
        "attrs": [
            "attr_Decking Material",
            "attr_Fall Height",
            "attr_Has Edge Guard",
            "attr_Has Pedestrian Railing",
            "attr_Length",
            "attr_Structure Material",
            "attr_Structure Position",
            "attr_Width",
        ],
    },
}

# Numeric attributes are binned for modelling but shown raw in the EDA.
NUM_ATTRS = [
    "attr_Fall Height",
    "attr_Number of Steps",
    "attr_Length",
    "attr_Width",
]

# The curated attribute set used for modelling (matches the notebook's explicit
# column selection). The raw manifest contains many additional attr_* columns
# (inspections, accessibility, capacity, ...) that are NOT part of this project
# and must be excluded.
MODEL_ATTR_COLUMNS = [
    "attr_Abutment Material",
    "attr_Bridge Type",
    "attr_Decking Material",
    "attr_Fall Height",
    "attr_Has Edge Guard",
    "attr_Has Pedestrian Railing",
    "attr_Length",
    "attr_Material (Frame, Tank, Body)",
    "attr_Number of Steps",
    "attr_Structure Material",
    "attr_Structure Position",
    "attr_Width",
]

# Labels remapped for readability in the categorical-distribution figure.
CATEGORICAL_REMAP = {
    "attr_Has Pedestrian Railing": {"No": "No railings"},
}

MISSING_SENTINELS = {0.0, -1.0}


def is_valid(val: object) -> bool:
    """Return True if a cell holds a usable attribute value.

    Treats NaN, any 'TBD' marker, and 0/-1 sentinels (numeric or string) as
    missing.
    """
    if pd.isna(val):
        return False
    if isinstance(val, str):
        val_stripped = val.strip()
        if "TBD" in val_stripped.upper():
            return False
        try:
            if float(val_stripped) in MISSING_SENTINELS:
                return False
        except ValueError:
            pass
    if val in MISSING_SENTINELS:
        return False
    return True


def attribute_columns(df: pd.DataFrame) -> list[str]:
    """The curated modelling attribute columns present in the dataframe.

    Deliberately excludes other attr_* columns in the raw manifest (inspections,
    accessibility, capacity, etc.) that are not part of this project.
    """
    return [c for c in MODEL_ATTR_COLUMNS if c in df.columns]


def plot_fill_rate(attributes_df: pd.DataFrame, output_dir: Path) -> Path:
    records = []
    for category, info in PROFILE_ATTRS.items():
        subset = attributes_df[attributes_df["profile_name"] == category]
        for col in info["attrs"]:
            if col not in attributes_df.columns:
                continue
            records.append({
                "category": category,
                "attr": col.replace("attr_", ""),
                "fill_rate": subset[col].apply(is_valid).mean() if len(subset) else 0.0,
            })
    fill_df = pd.DataFrame(records)

    categories = list(PROFILE_ATTRS.keys())
    n_cols = math.ceil(len(categories) / 2)
    n_rows = math.ceil(len(categories) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for ax, category in zip(axes, categories):
        data = fill_df[fill_df["category"] == category].sort_values("fill_rate")
        ax.barh(data["attr"], data["fill_rate"], color=BAR_COLOR, edgecolor="white")
        for val, patch in zip(data["fill_rate"], ax.patches):
            ax.text(patch.get_width() + 0.02, patch.get_y() + patch.get_height() / 2,
                    f"{val:.0%}", va="center", fontsize=9)
        ax.set_xlim(0, 1.15)
        ax.set_title(category, fontsize=10, fontweight="bold")
        ax.set_xlabel("Fill rate")
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(categories):]:
        fig.delaxes(ax)

    plt.tight_layout()
    out = output_dir / "attribute_fill_rate.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_categorical_distributions(attributes_df: pd.DataFrame, cat_attrs: list[str],
                                   output_dir: Path) -> Path:
    ncols = 2
    nrows = math.ceil(len(cat_attrs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 4))
    axes = np.array(axes).flatten().tolist()

    for ax, col in zip(axes, cat_attrs):
        valid_vals = attributes_df[col][attributes_df[col].apply(is_valid)]
        if col in CATEGORICAL_REMAP:
            valid_vals = valid_vals.replace(CATEGORICAL_REMAP[col])
        counts = valid_vals.value_counts()
        if counts.empty:
            ax.set_visible(False)
            continue
        bars = ax.barh(counts.index.astype(str), counts.values,
                       color=BAR_COLOR, edgecolor="white")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=9)
        ax.set_title(col.replace("attr_", ""), fontsize=10, fontweight="bold")
        ax.set_xlabel("Count")
        ax.set_xlim(0, counts.values.max() * 1.15)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(cat_attrs):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = output_dir / "categorical_distributions.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_numerical_distributions(attributes_df: pd.DataFrame, num_attrs: list[str],
                                 output_dir: Path) -> Path:
    ncols = 2
    nrows = math.ceil(len(num_attrs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 4))
    axes = np.array(axes).flatten().tolist()

    for ax, col in zip(axes, num_attrs):
        valid_vals = attributes_df[col][attributes_df[col].apply(is_valid)].dropna()
        valid_vals = pd.to_numeric(valid_vals, errors="coerce").dropna()
        if valid_vals.empty:
            ax.set_visible(False)
            continue
        ax.hist(valid_vals, bins=30, color=BAR_COLOR, edgecolor="white")
        ax.axvline(valid_vals.median(), color="#e74c3c", linestyle="--",
                   linewidth=1.5, label=f"Median: {valid_vals.median():.1f}")
        ax.axvline(valid_vals.mean(), color="#f39c12", linestyle="--",
                   linewidth=1.5, label=f"Mean: {valid_vals.mean():.1f}")
        ax.set_title(col.replace("attr_", ""), fontsize=10, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(num_attrs):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = output_dir / "numerical_distributions.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/citywide/image_attributes_manifest.csv"),
        help="Raw CityWide attribute manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/report_images"),
        help="Directory where the figure PNGs are written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(
            f"Raw attribute manifest not found at {args.input}.\n"
            "This file is large and is not committed to the repository. "
            "Obtain it (see README) or pass --input pointing to a local copy. "
            "The report still renders against the committed PNGs in "
            f"{args.output_dir}.",
            file=sys.stderr,
        )
        return 1

    attributes_df = pd.read_csv(args.input)
    if "profile_name" not in attributes_df.columns:
        print(
            f"Input {args.input} has no 'profile_name' column; cannot build the "
            "fill-rate figure. Columns found: "
            f"{attributes_df.columns.tolist()[:20]}...",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    attr_cols = attribute_columns(attributes_df)
    cat_attrs = [c for c in attr_cols if c not in NUM_ATTRS]
    num_attrs = [c for c in NUM_ATTRS if c in attributes_df.columns]

    written = [
        plot_fill_rate(attributes_df, args.output_dir),
        plot_categorical_distributions(attributes_df, cat_attrs, args.output_dir),
        plot_numerical_distributions(attributes_df, num_attrs, args.output_dir),
    ]
    print("Wrote EDA figures:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())