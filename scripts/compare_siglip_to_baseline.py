"""Compare SigLIP classifier results against majority-class baselines.

Usage:
    python scripts/compare_siglip_to_baseline.py
    python scripts/compare_siglip_to_baseline.py \
        --siglip-glob 'results/siglip_*_linear_svm_classification_results.csv' \
        --output results/siglip_linear_svm_vs_baseline_comparison.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_siglip_classifier import CLASSIFIER_OUTPUT_DIRS, default_output_dir  # noqa: E402


METRICS = ["accuracy_mean", "weighted_f1_mean", "macro_f1_mean"]
CLASSIFIER_CHOICES = tuple(CLASSIFIER_OUTPUT_DIRS)
DEFAULT_BASELINE_STRATEGY = "majority_class_group_cv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SigLIP classifier metrics with baseline metrics."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline summary CSV. Defaults to the organized baseline folder.",
    )
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIER_CHOICES,
        default="logistic_regression",
        help="SigLIP classifier results to compare.",
    )
    parser.add_argument(
        "--siglip-glob",
        default=None,
        help=(
            "Glob for SigLIP summary CSVs. Defaults to the standard folder "
            "for --classifier."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output comparison CSV. Defaults to the standard folder for --classifier.",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include attributes missing from either baseline or SigLIP results.",
    )
    parser.add_argument(
        "--baseline-strategy",
        default=DEFAULT_BASELINE_STRATEGY,
        help=(
            "Baseline strategy to compare against when the baseline file "
            "contains multiple strategies per attribute."
        ),
    )
    return parser.parse_args()


def default_baseline_path() -> Path:
    """Return the standard baseline result path, with legacy fallback."""
    organized = Path("results/baseline_results/baseline_classification_results.csv")
    if organized.exists():
        return organized
    return Path("results/baseline_classification_results.csv")


def default_result_glob(classifier: str) -> str:
    """Return the standard SigLIP result glob for a classifier."""
    suffix = "" if classifier == "logistic_regression" else f"_{classifier}"
    return (
        default_output_dir(classifier)
        / f"siglip_*{suffix}_classification_results.csv"
    ).as_posix()


def default_comparison_output(classifier: str) -> Path:
    """Return the standard SigLIP-vs-baseline comparison path."""
    suffix = "" if classifier == "logistic_regression" else f"_{classifier}"
    return default_output_dir(classifier) / f"siglip{suffix}_vs_baseline_comparison.csv"


def read_siglip_results(pattern: str) -> pd.DataFrame:
    paths = sorted(Path().glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No SigLIP result files matched: {pattern}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["siglip_result_file"] = str(path)
        frames.append(frame)

    if not frames:
        raise ValueError("All matched SigLIP result files were empty.")

    return pd.concat(frames, ignore_index=True)


def make_comparison(
    baseline: pd.DataFrame,
    siglip: pd.DataFrame,
    *,
    include_missing: bool = False,
    baseline_strategy: str | None = DEFAULT_BASELINE_STRATEGY,
) -> pd.DataFrame:
    if baseline_strategy is not None and "strategy" in baseline.columns:
        filtered = baseline[baseline["strategy"].eq(baseline_strategy)].copy()
        if filtered.empty:
            raise ValueError(f"No baseline rows matched strategy {baseline_strategy!r}.")
        baseline = filtered
    baseline = baseline.drop_duplicates("attribute")

    baseline_cols = [
        "attribute",
        "prediction",
        "n_labels",
        "n_assets",
        *METRICS,
    ]
    siglip_cols = [
        "attribute",
        "n_labels",
        "n_assets",
        "n_features",
        "feature_file",
        "siglip_result_file",
        *METRICS,
    ]

    comparison = baseline[baseline_cols].merge(
        siglip[siglip_cols],
        on="attribute",
        how="outer" if include_missing else "inner",
        suffixes=("_baseline", "_siglip"),
        indicator=True,
    )

    for metric in METRICS:
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_siglip"] - comparison[f"{metric}_baseline"]
        )

    ordered_columns = [
        "attribute",
        "_merge",
        "prediction",
        "n_labels_baseline",
        "n_labels_siglip",
        "n_assets_baseline",
        "n_assets_siglip",
        "n_features",
        "accuracy_mean_baseline",
        "accuracy_mean_siglip",
        "accuracy_mean_delta",
        "weighted_f1_mean_baseline",
        "weighted_f1_mean_siglip",
        "weighted_f1_mean_delta",
        "macro_f1_mean_baseline",
        "macro_f1_mean_siglip",
        "macro_f1_mean_delta",
        "feature_file",
        "siglip_result_file",
    ]
    comparison = comparison[ordered_columns]
    comparison = comparison.sort_values("macro_f1_mean_delta", ascending=False, na_position="last")
    return comparison.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    baseline = pd.read_csv(args.baseline or default_baseline_path())
    siglip_glob = args.siglip_glob or default_result_glob(args.classifier)
    output = args.output or default_comparison_output(args.classifier)
    siglip = read_siglip_results(siglip_glob)
    comparison = make_comparison(
        baseline,
        siglip,
        include_missing=args.include_missing,
        baseline_strategy=args.baseline_strategy,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False)

    print(f"Wrote {len(comparison)} comparison rows to {output}")
    print()
    print(
        comparison[
            [
                "attribute",
                "accuracy_mean_delta",
                "weighted_f1_mean_delta",
                "macro_f1_mean_delta",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
