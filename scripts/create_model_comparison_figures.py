"""Create reproducible figures for baseline, DINOv3, and SigLIP results.

The figures are intended for reports/presentations and can be regenerated after
any model run:

    python scripts/create_model_comparison_figures.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


METRICS = ["accuracy_mean", "weighted_f1_mean", "macro_f1_mean"]
MODEL_ORDER = ["Baseline", "DINOv3", "SigLIP"]
MODEL_COLORS = {
    "Baseline": "#6B7280",
    "DINOv3": "#2563EB",
    "SigLIP": "#059669",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script."""
    parser = argparse.ArgumentParser(
        description="Create clean comparison figures for baseline, DINOv3, and SigLIP."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/baseline_results/baseline_classification_results.csv"),
    )
    parser.add_argument(
        "--dinov3-comparison",
        type=Path,
        default=Path("results/dinov3_results/dinov3_logistic/dinov3_vs_baseline_comparison.csv"),
    )
    parser.add_argument(
        "--siglip-comparison",
        type=Path,
        default=Path("results/siglip_results/siglip_logistic_reg/siglip_vs_baseline_comparison.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures"),
    )
    return parser.parse_args()


def _read_inputs(
    baseline_path: Path,
    dinov3_path: Path,
    siglip_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load comparison result tables used to build report figures."""
    for path in [baseline_path, dinov3_path, siglip_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required result file is missing: {path}")
    return (
        pd.read_csv(baseline_path),
        pd.read_csv(dinov3_path),
        pd.read_csv(siglip_path),
    )


def build_model_summary(
    baseline: pd.DataFrame,
    dinov3_comparison: pd.DataFrame,
    siglip_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per attribute/model with common metric columns."""
    if "strategy" in baseline.columns:
        baseline = baseline[baseline["strategy"].eq("majority_class_group_cv")].copy()
    baseline = baseline.drop_duplicates("attribute")

    baseline_rows = baseline[
        ["attribute", "n_assets", "n_labels", *METRICS]
    ].copy()
    baseline_rows["model"] = "Baseline"

    dinov3_rows = dinov3_comparison[
        [
            "attribute",
            "n_assets_dinov3",
            "n_labels_dinov3",
            "accuracy_mean_dinov3",
            "weighted_f1_mean_dinov3",
            "macro_f1_mean_dinov3",
        ]
    ].copy()
    dinov3_rows.columns = ["attribute", "n_assets", "n_labels", *METRICS]
    dinov3_rows["model"] = "DINOv3"

    siglip_rows = siglip_comparison[
        [
            "attribute",
            "n_assets_siglip",
            "n_labels_siglip",
            "accuracy_mean_siglip",
            "weighted_f1_mean_siglip",
            "macro_f1_mean_siglip",
        ]
    ].copy()
    siglip_rows.columns = ["attribute", "n_assets", "n_labels", *METRICS]
    siglip_rows["model"] = "SigLIP"

    summary = pd.concat([baseline_rows, dinov3_rows, siglip_rows], ignore_index=True)
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    return summary.sort_values(["attribute", "model"]).reset_index(drop=True)


def add_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """Add baseline and DINOv3 deltas for each metric."""
    wide = summary.pivot(index="attribute", columns="model", values=METRICS)
    rows: list[dict[str, object]] = []
    for attribute in wide.index:
        row: dict[str, object] = {"attribute": attribute}
        for metric in METRICS:
            baseline_value = wide.loc[attribute, (metric, "Baseline")]
            dinov3_value = wide.loc[attribute, (metric, "DINOv3")]
            siglip_value = wide.loc[attribute, (metric, "SigLIP")]
            row[f"dinov3_minus_baseline_{metric}"] = dinov3_value - baseline_value
            row[f"siglip_minus_baseline_{metric}"] = siglip_value - baseline_value
            row[f"siglip_minus_dinov3_{metric}"] = siglip_value - dinov3_value
        rows.append(row)
    return pd.DataFrame(rows)


def _configure_style() -> None:
    """Apply the shared Matplotlib/Seaborn figure style."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 14,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "font.family": "DejaVu Sans",
        }
    )


def _ordered_attributes(summary: pd.DataFrame) -> list[str]:
    """Return attributes in a stable display order for figures."""
    siglip = summary[summary["model"].eq("SigLIP")]
    return (
        siglip.sort_values("macro_f1_mean", ascending=False)["attribute"]
        .drop_duplicates()
        .tolist()
    )


def plot_macro_f1_by_attribute(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot macro-F1 values by model and attribute."""
    ordered_attributes = _ordered_attributes(summary)
    figure, axis = plt.subplots(figsize=(12, 7))
    sns.barplot(
        data=summary,
        x="macro_f1_mean",
        y="attribute",
        hue="model",
        hue_order=MODEL_ORDER,
        order=ordered_attributes,
        palette=MODEL_COLORS,
        ax=axis,
    )
    axis.set_title("Macro F1 by Attribute")
    axis.set_xlabel("Macro F1")
    axis.set_ylabel("")
    axis.set_xlim(0, 1)
    axis.legend(title="")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_macro_f1_delta_vs_baseline(deltas: pd.DataFrame, output_path: Path) -> None:
    """Plot each embedding model's macro-F1 gain over baseline."""
    rows = []
    for _, row in deltas.iterrows():
        rows.append(
            {
                "attribute": row["attribute"],
                "model": "DINOv3",
                "macro_f1_delta": row["dinov3_minus_baseline_macro_f1_mean"],
            }
        )
        rows.append(
            {
                "attribute": row["attribute"],
                "model": "SigLIP",
                "macro_f1_delta": row["siglip_minus_baseline_macro_f1_mean"],
            }
        )
    long = pd.DataFrame(rows)
    order = (
        long[long["model"].eq("SigLIP")]
        .sort_values("macro_f1_delta", ascending=False)["attribute"]
        .tolist()
    )

    figure, axis = plt.subplots(figsize=(12, 7))
    sns.barplot(
        data=long,
        x="macro_f1_delta",
        y="attribute",
        hue="model",
        hue_order=["DINOv3", "SigLIP"],
        order=order,
        palette={key: MODEL_COLORS[key] for key in ["DINOv3", "SigLIP"]},
        ax=axis,
    )
    axis.axvline(0, color="#111827", linewidth=1)
    axis.set_title("Macro F1 Gain Over Majority Baseline")
    axis.set_xlabel("Macro F1 delta")
    axis.set_ylabel("")
    axis.legend(title="")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_average_metric_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot average model metrics across attributes."""
    average = summary.groupby("model", observed=False)[METRICS].mean().reset_index()
    long = average.melt(id_vars="model", var_name="metric", value_name="score")
    long["metric"] = long["metric"].map(
        {
            "accuracy_mean": "Accuracy",
            "weighted_f1_mean": "Weighted F1",
            "macro_f1_mean": "Macro F1",
        }
    )

    figure, axis = plt.subplots(figsize=(9, 5.5))
    sns.barplot(
        data=long,
        x="metric",
        y="score",
        hue="model",
        hue_order=MODEL_ORDER,
        palette=MODEL_COLORS,
        ax=axis,
    )
    axis.set_title("Average Score Across 12 Attributes")
    axis.set_xlabel("")
    axis.set_ylabel("Mean cross-validation score")
    axis.set_ylim(0, max(1.0, long["score"].max() + 0.05))
    axis.legend(title="")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_siglip_minus_dinov3_macro_f1(deltas: pd.DataFrame, output_path: Path) -> None:
    """Plot the per-attribute macro-F1 difference between SigLIP and DINOv3."""
    data = deltas[["attribute", "siglip_minus_dinov3_macro_f1_mean"]].copy()
    data = data.sort_values("siglip_minus_dinov3_macro_f1_mean", ascending=True)
    data["color"] = data["siglip_minus_dinov3_macro_f1_mean"].map(
        lambda value: MODEL_COLORS["SigLIP"] if value > 0 else MODEL_COLORS["DINOv3"]
    )

    figure, axis = plt.subplots(figsize=(10, 6.5))
    axis.barh(
        data["attribute"],
        data["siglip_minus_dinov3_macro_f1_mean"],
        color=data["color"],
    )
    axis.axvline(0, color="#111827", linewidth=1)
    axis.set_title("SigLIP vs DINOv3: Macro F1 Difference")
    axis.set_xlabel("SigLIP macro F1 minus DINOv3 macro F1")
    axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    """Run the script from parsed command-line arguments."""
    args = parse_args()
    _configure_style()

    baseline, dinov3, siglip = _read_inputs(
        args.baseline,
        args.dinov3_comparison,
        args.siglip_comparison,
    )
    summary = build_model_summary(baseline, dinov3, siglip)
    deltas = add_deltas(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "model_comparison_summary.csv"
    deltas_path = args.output_dir / "model_comparison_deltas.csv"
    summary.to_csv(summary_path, index=False)
    deltas.to_csv(deltas_path, index=False)

    plot_macro_f1_by_attribute(summary, args.output_dir / "model_macro_f1_by_attribute.png")
    plot_macro_f1_delta_vs_baseline(
        deltas,
        args.output_dir / "model_macro_f1_delta_vs_baseline.png",
    )
    plot_average_metric_summary(summary, args.output_dir / "model_average_metric_summary.png")
    plot_siglip_minus_dinov3_macro_f1(
        deltas,
        args.output_dir / "siglip_minus_dinov3_macro_f1.png",
    )

    print(f"Wrote summary data to {summary_path}")
    print(f"Wrote delta data to {deltas_path}")
    print(f"Wrote figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
