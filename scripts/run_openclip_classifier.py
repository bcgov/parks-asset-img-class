"""Run a grouped classifier on frozen OpenCLIP asset embeddings.

Pipeline role:
- reads one target label CSV from ``data/processed/train``;
- reads asset embeddings produced by ``scripts/extract_openclip_features.py``;
- evaluates a lightweight classifier with grouped cross-validation by
  ``asset_id``;
- writes OpenCLIP metrics, fold metrics, and out-of-fold predictions to the
  organized OpenCLIP results/prediction folders.

Self-contained per-model runner (mirrors run_siglip_classifier.py). Saves
metrics, per-fold metrics, and per-asset predictions.

Example:
    python scripts/run_openclip_classifier.py \
        --labels data/processed/train/attr_decking_material_train.csv \
        --features data/features/openclip_vitb16_all_attributes_assets.csv \
        --target attr_decking_material

To skip MLflow:
    python scripts/run_openclip_classifier.py ... --no-mlflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.openclip_classifier import CLASSIFIER_CHOICES, run_task_from_files  # noqa: E402


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

OPENCLIP_RESULTS_ROOT = Path("results/openclip_results")
CLASSIFIER_OUTPUT_DIRS = {
    "logistic_regression": "openclip_logistic_reg",
    "linear_svm": "openclip_linear_svm",
    "random_forest": "openclip_random_forest",
    "hist_gradient_boosting": "openclip_gradient_boost",
}

def default_output_dir(classifier: str) -> Path:
    """Return the organized results directory for a classifier type."""
    return OPENCLIP_RESULTS_ROOT / CLASSIFIER_OUTPUT_DIRS[classifier]

def default_prediction_dir(classifier: str) -> Path:
    """Return the organized prediction directory for a classifier type."""
    return default_output_dir(classifier) / "predictions"

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script."""
    parser = argparse.ArgumentParser(
        description="Evaluate frozen OpenCLIP embeddings with grouped CV."
    )
    parser.add_argument("--labels", type=Path, required=True, help="Task train CSV.")
    parser.add_argument("--features", type=Path, required=True, help="Asset-level OpenCLIP feature CSV.")
    parser.add_argument("--target", required=True, help="Target column, e.g. attr_decking_material.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIER_CHOICES,
        default="logistic_regression",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name/tag for MLflow. Defaults to openclip_vitb16_<classifier>.",
    )
    parser.add_argument("--data-version", default="processed-train")
    parser.add_argument("--experiment-name", default=None)
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
    """Log one OpenCLIP classifier run to DagsHub/MLflow."""
    try:
        import dagshub
        import mlflow
        import pandas as pd

        from src.mlflow_utils import make_run_name, make_standard_tags, setup_mlflow
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MLflow/DagsHub logging dependencies are missing. "
            "Rerun with --no-mlflow to skip."
        ) from exc

    dagshub.init(repo_owner="sgauth01", repo_name="parks-asset-img-class", mlflow=True)

    setup_kwargs = {}
    if experiment_name is not None:
        setup_kwargs["experiment_name"] = experiment_name
    setup_mlflow(**setup_kwargs)

    summary = pd.read_csv(summary_path)
    if summary.empty:
        print("Skipping MLflow logging because the OpenCLIP summary is empty.")
        return

    row = summary.iloc[0]
    tags = make_standard_tags(
        task=target,
        model_family="openclip",
        model_name=model_name,
        data_version=data_version,
        split_seed=random_state,
        extra={"cv_group": "asset_id"},
    )

    with mlflow.start_run(run_name=make_run_name(target, model_name), tags=tags):
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
        mlflow.log_artifact(str(predictions_path), artifact_path="results")


def main() -> int:
    """Run the script from parsed command-line arguments."""
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
    predictions_dir = args.predictions_dir or default_prediction_dir(args.classifier)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    suffix = "" if args.classifier == "logistic_regression" else f"_{args.classifier}"
    summary_path = output_dir / f"openclip_{args.target}{suffix}_classification_results.csv"
    folds_path = output_dir / f"openclip_{args.target}{suffix}_classification_cv_folds.csv"
    predictions_path = predictions_dir / f"openclip_{args.target}{suffix}_classification_predictions.csv"

    summary.to_csv(summary_path, index=False)
    folds.to_csv(folds_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print(f"Wrote {len(summary)} summary rows to {summary_path}")
    print(f"Wrote {len(folds)} fold rows to {folds_path}")
    print(f"Wrote {len(predictions)} prediction rows to {predictions_path}")

    if not args.no_mlflow:
        model_name = args.model_name or f"openclip_vitb16_{args.classifier}"
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
            per_asset_type=args.per_asset_type,
        )
        print("Logged OpenCLIP classifier results to MLflow/DagsHub")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
