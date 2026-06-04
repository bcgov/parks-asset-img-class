"""Compare SigLIP classifier results against majority-class baselines.

Usage:
    python scripts/compare_siglip_to_baseline.py
    python scripts/compare_siglip_to_baseline.py \
        --siglip-glob 'results/siglip_*_linear_svm_classification_results.csv' \
        --output results/siglip_linear_svm_vs_baseline_comparison.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["accuracy_mean", "weighted_f1_mean", "macro_f1_mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SigLIP classifier metrics with baseline metrics."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/baseline_classification_results.csv"),
        help="Baseline summary CSV.",
    )
    parser.add_argument(
        "--siglip-glob",
        default="results/siglip_*_classification_results.csv",
        help="Glob for SigLIP summary CSVs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/siglip_vs_baseline_comparison.csv"),
        help="Output comparison CSV.",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include attributes missing from either baseline or SigLIP results.",
    )
    return parser.parse_args()


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
) -> pd.DataFrame:
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
    baseline = pd.read_csv(args.baseline)
    siglip = read_siglip_results(args.siglip_glob)
    comparison = make_comparison(
        baseline,
        siglip,
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
