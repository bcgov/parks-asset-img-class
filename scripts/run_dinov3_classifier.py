"""Run a grouped classifier on frozen DINOv3 asset embeddings.

Example:
    python scripts/run_dinov3_classifier.py \
        --labels data/processed/train/attr_decking_material_train.csv \
        --features data/features/dinov3_vitb16_attr_decking_material_assets.csv \
        --target attr_decking_material \
        --classifier logistic_regression

To only write local CSVs:
    python scripts/run_dinov3_classifier.py ... --no-mlflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dinov3_classifier import CLASSIFIER_CHOICES, run_task_from_files  # noqa: E402

DINO_RESULTS_ROOT = Path("results/dinov3_results")
DINO_PREDICTIONS_ROOT = Path("data/predictions/dinov3_predictions")
CLASSIFIER_OUTPUT_DIRS = {
    "logistic_regression": "dinov3_logistic",
    "logistic_regression_tuned": "dinov3_logistic_tuned",
    "linear_svm": "dinov3_linear_svm",
    "random_forest": "dinov3_random_forest",
    "hist_gradient_boosting": "dinov3_gradient_boost",
}

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
    "n_labels",
    "n_assets",
    "n_features",
    "classifier",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen DINOv3 embeddings with grouped CV."
    )
    parser.add_argument("--labels", type=Path, required=True, help="Task train CSV.")
    parser.add_argument("--features", type=Path, required=True, help="Asset-level DINOv3 feature CSV.")
    parser.add_argument("--target", required=True, help="Target column, for example attr_decking_material.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for result CSVs. Defaults to "
            "results/dinov3_results/<classifier-specific-folder>."
        ),
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help=(
            "Directory for out-of-fold prediction CSVs. Defaults to "
            "data/predictions/dinov3_predictions/<classifier-specific-folder>."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIER_CHOICES,
        default="logistic_regression",
        help="Classifier to train on frozen DINOv3 embeddings.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name/tag to use in MLflow. Defaults to dinov3_vitb16_<classifier>.",
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
        help="Skip MLflow/DagsHub logging and only write result CSVs.",
    )
    parser.add_argument(
        "--model-family",
        choices=["dinov3", "openclip"],
        default="dinov3",
        help="Which embedding model produced the features.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Directory where prediction CSVs are written. Defaults to <output-dir>/predictions.",
    )
    return parser.parse_args()


def default_output_dir(classifier: str) -> Path:
    """Return the standard DINOv3 result folder for a classifier."""
    return DINO_RESULTS_ROOT / CLASSIFIER_OUTPUT_DIRS[classifier]


def default_prediction_dir(classifier: str) -> Path:
    """Return the standard DINOv3 prediction folder for a classifier."""
    return DINO_PREDICTIONS_ROOT / CLASSIFIER_OUTPUT_DIRS[classifier]


def log_results_to_mlflow(
    *,
    summary_path: Path,
    folds_path: Path,
    predictions_path: Path,
    labels_path: Path,
    features_path: Path,
    target: str,
    model_name: str,
    n_splits: int,
    random_state: int,
    data_version: str,
    experiment_name: str | None,
) -> None:
    """Log one DINOv3 classifier run to DagsHub/MLflow."""
    try:
        import dagshub
        import mlflow
        import pandas as pd

        from src.mlflow_utils import make_run_name, make_standard_tags, setup_mlflow
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MLflow/DagsHub logging dependencies are missing. Install the project "
            "environment or rerun with --no-mlflow."
        ) from exc

    dagshub.init(repo_owner="sgauth01", repo_name="parks-asset-img-class", mlflow=True)

    setup_kwargs = {}
    if experiment_name is not None:
        setup_kwargs["experiment_name"] = experiment_name
    setup_mlflow(**setup_kwargs)

    summary = pd.read_csv(summary_path)
    if summary.empty:
        print("Skipping MLflow logging because the DINOv3 summary is empty.")
        return

    row = summary.iloc[0]
    tags = make_standard_tags(
        task=target,
        model_family="dinov3",
        model_name=model_name,
        data_version=data_version,
        split_seed=random_state,
        extra={"cv_group": "asset_id"},
    )

    with mlflow.start_run(
        run_name=make_run_name(target, model_name),
        tags=tags,
    ):
        mlflow.log_params(
            {
                "labels_path": str(labels_path),
                "features_path": str(features_path),
                "output_summary_path": str(summary_path),
                "output_folds_path": str(folds_path),
                "output_predictions_path": str(predictions_path),
                "n_splits_requested": n_splits,
                "random_state": random_state,
            }
        )
        mlflow.log_params(
            {column: row[column] for column in PARAM_COLUMNS if column in row.index}
        )
        mlflow.log_metrics(
            {
                column: float(row[column])
                for column in METRIC_COLUMNS
                if column in row.index
            }
        )
        mlflow.log_artifact(str(summary_path), artifact_path="results")
        mlflow.log_artifact(str(folds_path), artifact_path="results")
        mlflow.log_artifact(str(predictions_path), artifact_path="predictions")


def main() -> int:
    args = parse_args()
    summary, folds, predictions = run_task_from_files(
        labels_path=args.labels,
        features_path=args.features,
        target=args.target,
        n_splits=args.folds,
        random_state=args.seed,
        classifier=args.classifier,
    )

    output_dir = args.output_dir or default_output_dir(args.classifier)
    prediction_dir = args.prediction_dir or default_prediction_dir(args.classifier)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.classifier == "logistic_regression" else f"_{args.classifier}"
    summary_path = output_dir / f"dinov3_{args.target}{suffix}_classification_results.csv"
    folds_path = output_dir / f"dinov3_{args.target}{suffix}_classification_cv_folds.csv"
    predictions_path = prediction_dir / f"dinov3_{args.target}{suffix}_classification_predictions.csv"
    summary.to_csv(summary_path, index=False)
    folds.to_csv(folds_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print(f"Wrote {len(summary)} summary rows to {summary_path}")
    print(f"Wrote {len(folds)} fold rows to {folds_path}")
    print(f"Wrote {len(predictions)} prediction rows to {predictions_path}")
    if not args.no_mlflow:
        model_name = args.model_name or f"dinov3_vitb16_{args.classifier}"
        log_results_to_mlflow(
            summary_path=summary_path,
            folds_path=folds_path,
            predictions_path=predictions_path,
            labels_path=args.labels,
            features_path=args.features,
            target=args.target,
            model_name=model_name,
            n_splits=args.folds,
            random_state=args.seed,
            data_version=args.data_version,
            experiment_name=args.experiment_name,
        )
        print("Logged DINOv3 classifier results to MLflow/DagsHub")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
