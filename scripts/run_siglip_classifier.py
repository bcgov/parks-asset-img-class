"""Run a grouped classifier on frozen SigLIP asset embeddings.

Example:
    python scripts/run_siglip_classifier.py \
        --labels data/processed/train/attr_decking_material_train.csv \
        --features data/features/siglip2_base_patch16_224_attr_decking_material_assets.csv \
        --target attr_decking_material

To only write local CSVs:
    python scripts/run_siglip_classifier.py ... --no-mlflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.siglip_classifier import CLASSIFIER_CHOICES, run_task_from_files  # noqa: E402
from src.siglip_features import DEFAULT_SIGLIP_MODEL, model_slug  # noqa: E402


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

SIGLIP_RESULTS_ROOT = Path("results/siglip_results")
CLASSIFIER_OUTPUT_DIRS = {
    "logistic_regression": "siglip_logistic_reg",
    "linear_svm": "siglip_linear_svm",
    "random_forest": "siglip_random_forest",
    "hist_gradient_boosting": "siglip_gradient_boost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen SigLIP embeddings with grouped CV."
    )
    parser.add_argument("--labels", type=Path, required=True, help="Task train CSV.")
    parser.add_argument("--features", type=Path, required=True, help="Asset-level SigLIP feature CSV.")
    parser.add_argument("--target", required=True, help="Target column, for example attr_decking_material.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for result CSVs. Defaults to "
            "results/siglip_results/<classifier-specific-folder>."
        ),
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help=(
            "Directory for prediction CSVs. Defaults to "
            "results/siglip_results/<classifier-specific-folder>/predictions."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIER_CHOICES,
        default="logistic_regression",
        help="Classifier to train on frozen SigLIP embeddings.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name/tag to use in MLflow. Defaults to <siglip_model_slug>_<classifier>.",
    )
    parser.add_argument(
        "--siglip-model",
        default=DEFAULT_SIGLIP_MODEL,
        help="Hugging Face SigLIP/SigLIP2 model id used to create the features.",
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
        "--per-asset-type",
        action="store_true",
        help="Train a separate model per asset type (for binned numeric attributes).",
    )
    return parser.parse_args()


def default_output_dir(classifier: str) -> Path:
    """Return the standard SigLIP result folder for a classifier."""
    return SIGLIP_RESULTS_ROOT / CLASSIFIER_OUTPUT_DIRS[classifier]


def default_prediction_dir(classifier: str) -> Path:
    """Return the prediction folder nested inside the classifier's result folder."""
    return default_output_dir(classifier) / "predictions"


def log_results_to_mlflow(
    *,
    summary_path: Path,
    folds_path: Path,
    predictions_path: Path,
    labels_path: Path,
    features_path: Path,
    target: str,
    model_name: str,
    siglip_model: str,
    n_splits: int,
    random_state: int,
    data_version: str,
    experiment_name: str | None,
) -> None:
    """Log one SigLIP classifier run to DagsHub/MLflow."""
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
        print("Skipping MLflow logging because the SigLIP summary is empty.")
        return

    row = summary.iloc[0]
    tags = make_standard_tags(
        task=target,
        model_family="siglip",
        model_name=model_name,
        data_version=data_version,
        split_seed=random_state,
        extra={"cv_group": "asset_id", "siglip_model": siglip_model},
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
                "siglip_model": siglip_model,
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
    summary_path = output_dir / f"siglip_{args.target}{suffix}_classification_results.csv"
    folds_path = output_dir / f"siglip_{args.target}{suffix}_classification_cv_folds.csv"
    predictions_path = prediction_dir / f"siglip_{args.target}{suffix}_classification_predictions.csv"
    summary.to_csv(summary_path, index=False)
    folds.to_csv(folds_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print(f"Wrote {len(summary)} summary rows to {summary_path}")
    print(f"Wrote {len(folds)} fold rows to {folds_path}")
    print(f"Wrote {len(predictions)} prediction rows to {predictions_path}")
    if not args.no_mlflow:
        model_name = args.model_name or f"{model_slug(args.siglip_model)}_{args.classifier}"
        log_results_to_mlflow(
            summary_path=summary_path,
            folds_path=folds_path,
            predictions_path=predictions_path,
            labels_path=args.labels,
            features_path=args.features,
            target=args.target,
            model_name=model_name,
            siglip_model=args.siglip_model,
            n_splits=args.folds,
            random_state=args.seed,
            data_version=args.data_version,
            experiment_name=args.experiment_name,
            per_asset_type=args.per_asset_type,
        )
        print("Logged SigLIP classifier results to MLflow/DagsHub")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())