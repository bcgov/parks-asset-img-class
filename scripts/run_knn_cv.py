"""Run grouped k-NN cross-validation from processed train CSVs and feature embeddings.

Usage:
    python scripts/run_knn_cv.py
    python scripts/run_knn_cv.py --knn-k 5 --folds 5
    python scripts/run_knn_cv.py --no-mlflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.vectordb_knn import cross_validate_knn_folder

import dagshub

dagshub.init(repo_owner="sgauth01", repo_name="parks-asset-img-class", mlflow=True)


CLASSIFICATION_METRIC_COLUMNS = [
    "accuracy_mean",
    "accuracy_std",
    "weighted_f1_mean",
    "weighted_f1_std",
    "macro_f1_mean",
    "macro_f1_std",
]

REGRESSION_METRIC_COLUMNS = [
    "mae_mean",
    "mae_std",
    "rmse_mean",
    "rmse_std",
    "r2_mean",
    "r2_std",
]

PARAM_COLUMNS = [
    "target_column",
    "target_file",
    "task_type",
    "splitter",
    "n_folds",
    "knn_k",
    "n_labels",
    "n_assets",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grouped k-NN cross-validation."
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("data/features"),
        help="Directory containing dinov3 feature CSV files.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/processed/train"),
        help="Directory containing *_train.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dinov3_knn_cv"),
        help="Directory where k-NN CV CSV outputs are written.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=10,
        help="Number of nearest neighbors for k-NN (default: 10).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
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
    feature_dir: Path,
    train_dir: Path,
    output_dir: Path,
    knn_k: int,
    n_splits: int,
    random_state: int,
    data_version: str,
    experiment_name: str | None,
) -> None:
    """Log one parent run plus one nested run per attribute."""
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
        model_family="knn",
        model_name=f"knn_group_cv_k{knn_k}",
        data_version=data_version,
        split_seed=random_state,
        extra={"cv_group": "asset_id", "knn_k": knn_k},
    )
    with mlflow.start_run(
        run_name=make_run_name("all_classification_attributes", f"knn_group_cv_k{knn_k}"),
        tags=parent_tags,
    ):
        mlflow.log_params(
            {
                "feature_dir": str(feature_dir),
                "train_dir": str(train_dir),
                "output_dir": str(output_dir),
                "knn_k": knn_k,
                "n_splits_requested": n_splits,
                "random_state": random_state,
                "n_attributes": len(summary),
            }
        )
        mlflow.log_artifact(str(summary_path), artifact_path="results")
        mlflow.log_artifact(str(folds_path), artifact_path="results")

        for _, row in summary.iterrows():
            attribute = str(row["attribute"])
            task_type = str(row.get("task_type", "classification"))
            tags = make_standard_tags(
                task=attribute,
                model_family="knn",
                model_name=f"knn_group_cv_k{knn_k}",
                data_version=data_version,
                split_seed=random_state,
                extra={"cv_group": "asset_id", "task_type": task_type, "knn_k": knn_k},
            )
            with mlflow.start_run(
                run_name=make_run_name(attribute, f"knn_group_cv_k{knn_k}"),
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
                # Log appropriate metrics based on task type
                metric_columns = (
                    CLASSIFICATION_METRIC_COLUMNS
                    if task_type == "classification"
                    else REGRESSION_METRIC_COLUMNS
                )
                mlflow.log_metrics(
                    {
                        column: float(row[column])
                        for column in metric_columns
                        if column in row.index
                    }
                )


def main() -> int:
    args = parse_args()

    summary, fold_results = cross_validate_knn_folder(
        feature_dir=args.feature_dir,
        train_dir=args.train_dir,
        knn_k=args.knn_k,
        n_splits=args.folds,
        random_state=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"knn_cv_results_k{args.knn_k}.csv"
    folds_path = args.output_dir / f"knn_cv_folds_k{args.knn_k}.csv"
    summary.to_csv(summary_path, index=False)
    fold_results.to_csv(folds_path, index=False)

    print(f"Wrote {len(summary)} summary rows to {summary_path}")
    print(f"Wrote {len(fold_results)} fold rows to {folds_path}")
    if not args.no_mlflow:
        log_results_to_mlflow(
            summary_path=summary_path,
            folds_path=folds_path,
            feature_dir=args.feature_dir,
            train_dir=args.train_dir,
            output_dir=args.output_dir,
            knn_k=args.knn_k,
            n_splits=args.folds,
            random_state=args.seed,
            data_version=args.data_version,
            experiment_name=args.experiment_name,
        )
        print("Logged k-NN CV results to MLflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
