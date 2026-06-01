"""Compare DINOv3 classifier results against majority-class baselines.

Usage:
    python scripts/compare_dinov3_to_baseline.py
    python scripts/compare_dinov3_to_baseline.py \
        --dinov3-glob 'results/dinov3_*_linear_svm_classification_results.csv' \
        --output results/dinov3_linear_svm_vs_baseline_comparison.csv

This reads:
    results/baseline_classification_results.csv
    results/dinov3_*_classification_results.csv

and writes:
    results/dinov3_vs_baseline_comparison.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["accuracy_mean", "weighted_f1_mean", "macro_f1_mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare DINOv3 classifier metrics with baseline metrics."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/baseline_classification_results.csv"),
        help="Baseline summary CSV.",
    )
    parser.add_argument(
        "--dinov3-glob",
        default="results/dinov3_*_classification_results.csv",
        help="Glob for DINOv3 summary CSVs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dinov3_vs_baseline_comparison.csv"),
        help="Output comparison CSV.",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include attributes missing from either baseline or DINOv3 results.",
    )
    return parser.parse_args()


def read_dinov3_results(pattern: str) -> pd.DataFrame:
    paths = sorted(Path().glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No DINOv3 result files matched: {pattern}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["dinov3_result_file"] = str(path)
        frames.append(frame)

    if not frames:
        raise ValueError("All matched DINOv3 result files were empty.")

    return pd.concat(frames, ignore_index=True)


def make_comparison(
    baseline: pd.DataFrame,
    dinov3: pd.DataFrame,
    *,
    include_missing: bool = False,
) -> pd.DataFrame:
    baseline_cols = [
        "attribute",
        "prediction",
        "n_labels",
        "n_assets",
        *METRICS,
    ]
    dinov3_cols = [
        "attribute",
        "n_labels",
        "n_assets",
        "n_features",
        "feature_file",
        "dinov3_result_file",
        *METRICS,
    ]

    comparison = baseline[baseline_cols].merge(
        dinov3[dinov3_cols],
        on="attribute",
        how="outer" if include_missing else "inner",
        suffixes=("_baseline", "_dinov3"),
        indicator=True,
    )

    for metric in METRICS:
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_dinov3"] - comparison[f"{metric}_baseline"]
        )

    ordered_columns = [
        "attribute",
        "_merge",
        "prediction",
        "n_labels_baseline",
        "n_labels_dinov3",
        "n_assets_baseline",
        "n_assets_dinov3",
        "n_features",
        "accuracy_mean_baseline",
        "accuracy_mean_dinov3",
        "accuracy_mean_delta",
        "weighted_f1_mean_baseline",
        "weighted_f1_mean_dinov3",
        "weighted_f1_mean_delta",
        "macro_f1_mean_baseline",
        "macro_f1_mean_dinov3",
        "macro_f1_mean_delta",
        "feature_file",
        "dinov3_result_file",
    ]
    comparison = comparison[ordered_columns]
    comparison = comparison.sort_values("macro_f1_mean_delta", ascending=False, na_position="last")
    return comparison.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    baseline = pd.read_csv(args.baseline)
    dinov3 = read_dinov3_results(args.dinov3_glob)
    comparison = make_comparison(
        baseline,
        dinov3,
        include_missing=args.include_missing,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False)

    print(f"Wrote {len(comparison)} comparison rows to {args.output}")
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
