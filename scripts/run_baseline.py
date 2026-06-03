"""Run all three baseline strategies from processed train CSVs.
 
Strategies:
    - majority_class_group_cv   : always predict the most common training label
    - uniform_random_group_cv   : predict a label uniformly at random
    - stratified_random_group_cv: predict a label weighted by training frequency
 
Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --no-mlflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline import DEFAULT_TRAIN_DIR, cross_validate_train_folder  # noqa: E402

import dagshub
dagshub.init(repo_owner='sgauth01', repo_name='parks-asset-img-class', mlflow=True)


METRIC_COLUMNS = [
    "train_majority_share_mean",
    "train_majority_share_std",
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
    "splitter",
    "n_folds",
    "n_labels",
    "n_assets",
    "prediction",
]

STRATEGIES = [
    "majority_class_group_cv",
    "uniform_random_group_cv",
    "stratified_random_group_cv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all three grouped baseline strategies."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Directory containing *_train.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/baseline_results"),
        help="Directory where baseline CSV outputs are written.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Maximum CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--data-version",
        default="processed-train",
        help="Data version tag to attach to MLflow runs.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name. Defaults to the project standard.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging and only write result CSVs.",
    )
    return parser.parse_args()
 
 
def log_results_to_mlflow(
    *,
    summary_path: Path,
    folds_path: Path,
    train_dir: Path,
    output_dir: Path,
    n_splits: int,
    random_state: int,
    data_version: str,
    experiment_name: str | None,
) -> None:
    """Log one parent run plus nested runs per attribute × strategy."""
    try:
        import mlflow
 
        from src.mlflow_utils import make_run_name, make_standard_tags, setup_mlflow
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MLflow is not installed in this Python environment. Install the "
            "project environment or rerun with --no-mlflow."
        ) from exc
 
    import pandas as pd
 
    summary = pd.read_csv(summary_path)
    setup_kwargs = {}
    if experiment_name is not None:
        setup_kwargs["experiment_name"] = experiment_name
    setup_mlflow(**setup_kwargs)
 
    mlflow.autolog()
 
    parent_tags = make_standard_tags(
        task="all_classification_attributes",
        model_family="baseline",
        model_name="all_baselines_group_cv",
        data_version=data_version,
        split_seed=random_state,
        extra={"cv_group": "asset_id"},
    )
    with mlflow.start_run(
        run_name=make_run_name("all_classification_attributes", "all_baselines_group_cv"),
        tags=parent_tags,
    ):
        mlflow.log_params(
            {
                "train_dir": str(train_dir),
                "output_dir": str(output_dir),
                "n_splits_requested": n_splits,
                "random_state": random_state,
                "n_attributes": summary["attribute"].nunique(),
                "strategies": ", ".join(STRATEGIES),
            }
        )
        mlflow.log_artifact(str(summary_path), artifact_path="results")
        mlflow.log_artifact(str(folds_path), artifact_path="results")
 
        # One nested run per attribute × strategy combination
        for _, row in summary.iterrows():
            attribute = str(row["attribute"])
            strategy = str(row["strategy"])
            tags = make_standard_tags(
                task=attribute,
                model_family="baseline",
                model_name=strategy,
                data_version=data_version,
                split_seed=random_state,
                extra={"cv_group": "asset_id"},
            )
            with mlflow.start_run(
                run_name=make_run_name(attribute, strategy),
                tags=tags,
                nested=True,
            ):
                mlflow.log_params(
                    {
                        column: row[column]
                        for column in PARAM_COLUMNS
                        if column in row.index
                    }
                )
                mlflow.log_metrics(
                    {
                        column: float(row[column])
                        for column in METRIC_COLUMNS
                        if column in row.index
                    }
                )
 
 
def main() -> int:
    args = parse_args()
    summary, fold_results = cross_validate_train_folder(
        train_dir=args.train_dir,
        n_splits=args.folds,
        random_state=args.seed,
    )
 
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "baseline_classification_results.csv"
    folds_path = args.output_dir / "baseline_classification_cv_folds.csv"
    summary.to_csv(summary_path, index=False)
    fold_results.to_csv(folds_path, index=False)
 
    n_attributes = summary["attribute"].nunique() if not summary.empty else 0
    n_strategies = summary["strategy"].nunique() if not summary.empty else 0
    print(f"Wrote {len(summary)} summary rows ({n_attributes} attributes × {n_strategies} strategies) to {summary_path}")
    print(f"Wrote {len(fold_results)} fold rows to {folds_path}")
 
    if not args.no_mlflow:
        log_results_to_mlflow(
            summary_path=summary_path,
            folds_path=folds_path,
            train_dir=args.train_dir,
            output_dir=args.output_dir,
            n_splits=args.folds,
            random_state=args.seed,
            data_version=args.data_version,
            experiment_name=args.experiment_name,
        )
        print("Logged all baseline results to MLflow")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())