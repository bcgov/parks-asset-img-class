"""Run DINOv3 experiments for the remaining classification attributes.

The expensive step is DINOv3 feature extraction. Because DINOv3 features do
not depend on the target label, this script extracts one shared feature table
from the union of the requested train CSVs, then reuses that feature table for
each attribute classifier.

Default target order excludes ``attr_decking_material`` because that was the
first completed DINOv3 experiment.

Usage:
    python scripts/run_dinov3_remaining_attributes.py

To include decking material too:
    python scripts/run_dinov3_remaining_attributes.py --include-decking

To only write local CSVs and skip DagsHub/MLflow:
    python scripts/run_dinov3_remaining_attributes.py --no-mlflow

To run Linear SVM instead of logistic regression:
    python scripts/run_dinov3_remaining_attributes.py --classifier linear_svm

To tune logistic regression:
    python scripts/run_dinov3_remaining_attributes.py --classifier logistic_regression_tuned

To run Random Forest:
    python scripts/run_dinov3_remaining_attributes.py --classifier random_forest

To run histogram-based gradient boosting:
    python scripts/run_dinov3_remaining_attributes.py --classifier hist_gradient_boosting
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_dinov3_classifier import default_output_dir, default_prediction_dir  # noqa: E402
from src.dinov3_classifier import CLASSIFIER_CHOICES  # noqa: E402
from src.dinov3_features import DEFAULT_IMAGE_ROOT  # noqa: E402


DEFAULT_TARGETS = [
    "attr_abutment_material",
    "attr_bridge_type",
    "attr_has_edge_guard",
    "attr_has_pedestrian_railing",
    "attr_material_frame_tank_body",
    "attr_structure_material",
    "attr_structure_position",
    "fall_height_bin",
    "length_bin",
    "steps_bin",
    "width_bin",
]

ALL_TARGETS_IN_BASELINE_ORDER = [
    "attr_abutment_material",
    "attr_bridge_type",
    "attr_decking_material",
    "attr_has_edge_guard",
    "attr_has_pedestrian_railing",
    "attr_material_frame_tank_body",
    "attr_structure_material",
    "attr_structure_position",
    "fall_height_bin",
    "length_bin",
    "steps_bin",
    "width_bin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract shared DINOv3 features and run classifiers for multiple attributes."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/processed/train"),
        help="Directory containing <target>_train.csv files.",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("data/features"),
        help="Directory where shared DINOv3 feature CSVs are written.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where classifier result CSVs are written. Defaults to "
            "results/dinov3_results/<classifier-specific-folder>."
        ),
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=None,
        help=(
            "Directory where out-of-fold prediction CSVs are written. Defaults "
            "to data/predictions/dinov3_predictions/<classifier-specific-folder>."
        ),
    )
    parser.add_argument("--model", default="dinov3_vitb16")
    parser.add_argument(
        "--weights",
        default="models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    )
    parser.add_argument("--model-source", default="facebookresearch/dinov3")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIER_CHOICES,
        default="logistic_regression",
        help="Classifier to train on frozen DINOv3 embeddings.",
    )
    parser.add_argument("--data-version", default="processed-train")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Optional explicit target list. Defaults to the remaining 11 attributes.",
    )
    parser.add_argument(
        "--include-decking",
        action="store_true",
        help="Run all 12 baseline targets, including attr_decking_material.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Recreate the shared DINOv3 feature files even if they already exist.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip DagsHub/MLflow logging for classifier runs.",
    )
    return parser.parse_args()


def selected_targets(args: argparse.Namespace) -> list[str]:
    if args.targets is not None:
        return args.targets
    if args.include_decking:
        return ALL_TARGETS_IN_BASELINE_ORDER
    return DEFAULT_TARGETS


def target_train_path(train_dir: Path, target: str) -> Path:
    return train_dir / f"{target}_train.csv"


def build_union_input(targets: list[str], train_dir: Path, union_path: Path) -> None:
    frames = []
    for target in targets:
        csv_path = target_train_path(train_dir, target)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing train CSV for {target}: {csv_path}")
        frame = pd.read_csv(csv_path, usecols=["asset_id", "image_path"])
        frame["source_target"] = target
        frames.append(frame)

    union = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["asset_id", "image_path"])
        .drop_duplicates(["asset_id", "image_path"])
        .sort_values(["asset_id", "image_path"])
    )
    union_path.parent.mkdir(parents=True, exist_ok=True)
    union.to_csv(union_path, index=False)


def run_command(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main() -> int:
    args = parse_args()
    targets = selected_targets(args)
    output_dir = args.output_dir or default_output_dir(args.classifier)
    prediction_dir = args.prediction_dir or default_prediction_dir(args.classifier)

    args.feature_dir.mkdir(parents=True, exist_ok=True)
    union_path = args.feature_dir / f"{args.model}_remaining_attributes_union_input.csv"
    image_features_path = args.feature_dir / f"{args.model}_remaining_attributes_images.csv"
    asset_features_path = args.feature_dir / f"{args.model}_remaining_attributes_assets.csv"

    print("Targets, in order:")
    for index, target in enumerate(targets, start=1):
        print(f"{index}. {target}")

    if args.force_extract or not asset_features_path.exists():
        build_union_input(targets, args.train_dir, union_path)
        extraction_command = [
            sys.executable,
            "scripts/extract_dinov3_features.py",
            "--input",
            str(union_path),
            "--output",
            str(image_features_path),
            "--asset-output",
            str(asset_features_path),
            "--model",
            args.model,
            "--weights",
            args.weights,
            "--model-source",
            args.model_source,
            "--image-root",
            str(args.image_root),
        ]
        if args.device is not None:
            extraction_command.extend(["--device", args.device])
        run_command(extraction_command)
    else:
        print(f"Reusing existing shared asset features: {asset_features_path}")

    for target in targets:
        classifier_command = [
            sys.executable,
            "scripts/run_dinov3_classifier.py",
            "--labels",
            str(target_train_path(args.train_dir, target)),
            "--features",
            str(asset_features_path),
            "--target",
            target,
            "--output-dir",
            str(output_dir),
            "--prediction-dir",
            str(prediction_dir),
            "--folds",
            str(args.folds),
            "--seed",
            str(args.seed),
            "--classifier",
            args.classifier,
            "--model-name",
            f"{args.model}_{args.classifier}",
            "--data-version",
            args.data_version,
        ]
        if args.experiment_name is not None:
            classifier_command.extend(["--experiment-name", args.experiment_name])
        if args.no_mlflow:
            classifier_command.append("--no-mlflow")
        run_command(classifier_command)

    print("\nFinished DINOv3 runs for requested attributes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
