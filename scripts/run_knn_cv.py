"""Run grouped k-NN cross-validation on frozen DINOv3 asset embeddings.

Example (single attribute type):
    python scripts/run_knn_cv.py \
        --labels data/processed/train/attr_decking_material_train.csv \
        --features data/features/dinov3_attr_decking_material_train_assets.csv \
        --target attr_decking_material

Example (all targets):
    python scripts/run_knn_cv.py --all

To only write local CSVs:
    python scripts/run_knn_cv.py ... --no-mlflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import mlflow
import dagshub

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.knn import cross_validate_knn, DEFAULT_K
from src.mlflow_utils import setup_mlflow

METRIC_COLUMNS = [
    "accuracy_mean",
    "accuracy_std",
    "weighted_f1_mean",
    "weighted_f1_std",
    "macro_f1_mean",
    "macro_f1_std",
]

PARAM_COLUMNS = [
    "target_column",
    "target_file",
    "feature_file",
    "splitter",
    "n_folds",
    "n_valid_assets",
    "knn_k",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script."""
    parser = argparse.ArgumentParser(
        description="Evaluate frozen DINOv3 embeddings with grouped k-NN CV."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all attributes automatically by scanning feature directory.",
    )
    parser.add_argument("--labels", type=Path, help="Task train CSV (required if not using --all).")
    parser.add_argument("--features", type=Path, help="Asset-level DINOv3 feature CSV (required if not using --all).")
    parser.add_argument("--target", help="Target column (required if not using --all).")
    parser.add_argument("--output-dir", type=Path, default=Path("results/new_dinov3_knn_cv"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"), help="Directory to scan for feature CSVs when using --all.")
    parser.add_argument("--train-dir", type=Path, default=Path("data/processed/train"), help="Directory to scan for train CSVs when using --all.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="k-NN neighbors to use.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name/tag to use in MLflow. Defaults to dinov3_vitb16_knn.",
    )
    parser.add_argument(
        "--data-version",
        default="processed-train",
        help="Data version tag to attach to MLflow runs.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow/DagsHub logging and only write result CSVs.",
    )
    return parser.parse_args()


def run_single_target(
    labels_path: Path,
    features_path: Path,
    target: str,
    output_dir: Path,
    folds: int,
    k: int,
    seed: int,
    model_name: str | None,
    data_version: str,
    experiment_name: str | None,
    no_mlflow: bool,
) -> tuple[bool, str]:
    """Run k-NN CV for a single target. Returns (success, message)."""

    try:
        labels = pd.read_csv(labels_path)
        features = pd.read_csv(features_path)
    except Exception as e:
        return False, f"Failed to load files: {e}"

    summary, fold_results = cross_validate_knn(
        labels,
        features,
        target,
        target_file=str(labels_path),
        feature_file=str(features_path),
        n_splits=folds,
        random_state=seed,
        knn_k=k,
    )

    if summary.empty:
        return False, f"No results for {target}"

    output_dir.mkdir(parents=True, exist_ok=True)
    target_stem = target.replace("attr_", "")
    summary_path = output_dir / f"{target_stem}_knn_summary.csv"
    folds_path = output_dir / f"{target_stem}_knn_folds.csv"

    summary.to_csv(summary_path, index=False)
    fold_results.to_csv(folds_path, index=False)

    msg = f"✓ {target}: {summary_path}"

    if not no_mlflow:
        try:
            dagshub.init(repo_owner='sgauth01', repo_name='parks-asset-img-class', mlflow=True)

            exp_name = experiment_name or "dinov3_knn_evaluation"
            model = model_name or f"dinov3_vitb16_knn_k{k}"

            setup_mlflow(experiment_name=exp_name)
            with mlflow.start_run(run_name=f"{target_stem}_fold_cv"):
                mlflow.log_params(
                    {
                        "target": target,
                        "model": model,
                        "knn_k": k,
                        "n_splits": folds,
                        "random_state": seed,
                        "data_version": data_version,
                    }
                )
                for col in METRIC_COLUMNS:
                    if col in summary.columns:
                        value = summary[col].iloc[0]
                        mlflow.log_metric(col, float(value))

                mlflow.log_artifact(str(summary_path))
                mlflow.log_artifact(str(folds_path))
        except ImportError:
            pass
        except Exception:
            pass

    return True, msg


def find_attribute_pairs(
    feature_dir: Path, train_dir: Path
) -> list[tuple[str, Path, Path]]:
    """Find matching label and feature files for k-NN experiments."""
    pairs = []
    feature_dir = Path(feature_dir)
    train_dir = Path(train_dir)

    # Match actual naming: dinov3_attr_*_train_assets.csv
    for feature_csv in sorted(feature_dir.glob("dinov3_attr_*_train_assets.csv")):
        stem = feature_csv.stem  # e.g., "dinov3_attr_decking_material_train_assets"
        attr_name = stem.replace("dinov3_", "").replace("_train_assets", "")  # e.g., "attr_decking_material"

        train_csv = train_dir / f"{attr_name}_train.csv"
        if train_csv.exists():
            pairs.append((attr_name, feature_csv, train_csv))
        else:
            print(f"  [skip] no train CSV found for {attr_name} (expected {train_csv})")
    
    # Bin targets reuse parent feature files
    BIN_TARGETS = {
        "fall_height_bin": "dinov3_attr_fall_height_train_assets.csv",
        "steps_bin": "dinov3_attr_number_of_steps_train_assets.csv",
        "length_bin": "dinov3_attr_length_train_assets.csv",
        "width_bin": "dinov3_attr_width_train_assets.csv",
    }
    for attr_name, feature_filename in BIN_TARGETS.items():
        feature_csv = feature_dir / feature_filename
        train_csv   = train_dir / f"{attr_name}_train.csv"
        if feature_csv.exists() and train_csv.exists():
            pairs.append((attr_name, feature_csv, train_csv))
        else:
            if not feature_csv.exists():
                print(f"  [skip] feature file missing: {feature_csv}")
            if not train_csv.exists():
                print(f"  [skip] train file missing: {train_csv}")

    return pairs


def main() -> int:
    """Run the script from parsed command-line arguments."""
    args = parse_args()

    if args.all:
        pairs = find_attribute_pairs(args.feature_dir, args.train_dir)
        if not pairs:
            print(f"No attribute pairs found in {args.feature_dir} and {args.train_dir}")
            return 1

        print(f"Found {len(pairs)} attributes to process\n")

        success_count = 0
        summary_dfs = []

        for target, feature_path, train_path in pairs:
            success, msg = run_single_target(
                train_path,
                feature_path,
                target,
                args.output_dir,
                args.folds,
                args.k,
                args.seed,
                args.model_name,
                args.data_version,
                args.experiment_name,
                args.no_mlflow,
            )
            print(msg)
            if success:
                success_count += 1
                try:
                    target_stem = target.replace("attr_", "")
                    summary_path = args.output_dir / f"{target_stem}_knn_summary.csv"
                    summary_dfs.append(pd.read_csv(summary_path))
                except Exception:
                    pass

        # Write combined summary
        if summary_dfs:
            combined = pd.concat(summary_dfs, ignore_index=True)
            combined_path = args.output_dir / "knn_summary_all.csv"
            combined.to_csv(combined_path, index=False)
            print(f"\nWrote combined summary to {combined_path}")

        print(f"\nCompleted {success_count}/{len(pairs)} attributes successfully")
        return 0 if success_count > 0 else 1

    # Single target mode
    if not args.labels or not args.features or not args.target:
        print("Error: --labels, --features, and --target are required when not using --all")
        return 1

    success, msg = run_single_target(
        args.labels,
        args.features,
        args.target,
        args.output_dir,
        args.folds,
        args.k,
        args.seed,
        args.model_name,
        args.data_version,
        args.experiment_name,
        args.no_mlflow,
    )
    print(msg)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

