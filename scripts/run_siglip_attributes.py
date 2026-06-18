"""Run SigLIP experiments for multiple classification attributes.

The expensive step is SigLIP feature extraction. Since SigLIP features do not
depend on the target label, this script extracts one shared feature table from
the union of the requested train CSVs, then reuses it for each classifier.

Usage:
    python scripts/run_siglip_attributes.py

To only write local CSVs and skip DagsHub/MLflow:
    python scripts/run_siglip_attributes.py --no-mlflow
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

from scripts.run_dinov3_remaining_attributes import (
    DEFAULT_TARGETS,
    PER_ASSET_TYPE_TARGETS,
    target_train_path,
)
from scripts.run_siglip_classifier import default_output_dir  # noqa: E402
from src.dinov3_classifier import CLASSIFIER_CHOICES  # noqa: E402
from src.siglip_features import DEFAULT_IMAGE_ROOT, DEFAULT_SIGLIP_MODEL, model_slug  # noqa: E402

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract shared SigLIP features and run classifiers for multiple attributes."
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
        help="Directory where shared SigLIP feature CSVs are written.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where classifier result CSVs are written. Defaults to "
            "results/siglip_results/<classifier-specific-folder>."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_SIGLIP_MODEL,
        help="Hugging Face SigLIP/SigLIP2 model id.",
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIER_CHOICES,
        default="logistic_regression",
        help="Classifier to train on frozen SigLIP embeddings.",
    )
    parser.add_argument("--data-version", default="processed-train")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="Optional explicit target list. Defaults to all 12 attributes.",
    )
    parser.add_argument(
        "--include-decking",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Recreate the shared SigLIP feature files even if they already exist.",
    )
    parser.add_argument(
        "--limit-assets",
        type=int,
        default=None,
        help="Optional smoke-test limit applied during feature extraction.",
    )
    parser.add_argument(
        "--smoke-assets-per-class",
        type=int,
        default=None,
        help=(
            "Optional class-balanced smoke-test sample. Keeps up to this many "
            "assets per class for each target before feature extraction."
        ),
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
    return DEFAULT_TARGETS


def target_set_slug(args: argparse.Namespace, targets: list[str]) -> str:
    """Return a stable filename slug for the requested target set."""
    if args.targets is not None:
        return "_".join(targets)
    return "all_attributes"


def _target_column(frame: pd.DataFrame, target: str) -> str:
    if target in frame.columns:
        return target
    attr_columns = [column for column in frame.columns if column.startswith("attr_")]
    if len(attr_columns) == 1:
        return attr_columns[0]
    raise ValueError(f"Could not infer target column for {target}.")


def _class_balanced_asset_sample(frame: pd.DataFrame, target: str, per_class: int) -> pd.DataFrame:
    target_column = _target_column(frame, target)
    labelled = frame.dropna(subset=["asset_id", target_column])
    asset_labels = labelled[["asset_id", target_column]].drop_duplicates("asset_id")
    keep_assets = (
        asset_labels.groupby(target_column, group_keys=False)
        .head(per_class)["asset_id"]
    )
    return frame[frame["asset_id"].isin(keep_assets)]


def build_union_input(
    targets: list[str],
    train_dir: Path,
    union_path: Path,
    *,
    smoke_assets_per_class: int | None = None,
) -> None:
    frames = []
    for target in targets:
        csv_path = target_train_path(train_dir, target)
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing train CSV for {target}: {csv_path}")
        frame = pd.read_csv(csv_path)
        if smoke_assets_per_class is not None:
            frame = _class_balanced_asset_sample(frame, target, smoke_assets_per_class)
        frame = frame[["asset_id", "image_path"]]
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
    slug = model_slug(args.model_name)
    targets_slug = target_set_slug(args, targets)
    output_dir = args.output_dir or default_output_dir(args.classifier)

    args.feature_dir.mkdir(parents=True, exist_ok=True)
    union_path = args.feature_dir / f"{slug}_{targets_slug}_union_input.csv"
    image_features_path = args.feature_dir / f"{slug}_{targets_slug}_images.csv"
    asset_features_path = args.feature_dir / f"{slug}_{targets_slug}_assets.csv"

    print("SigLIP targets, in order:")
    for index, target in enumerate(targets, start=1):
        print(f"{index}. {target}")

    if args.force_extract or not asset_features_path.exists():
        build_union_input(
            targets,
            args.train_dir,
            union_path,
            smoke_assets_per_class=args.smoke_assets_per_class,
        )
        extraction_command = [
            sys.executable,
            "scripts/extract_siglip_features.py",
            "--input",
            str(union_path),
            "--output",
            str(image_features_path),
            "--asset-output",
            str(asset_features_path),
            "--model-name",
            args.model_name,
            "--image-root",
            str(args.image_root),
        ]
        if args.device is not None:
            extraction_command.extend(["--device", args.device])
        if args.limit_assets is not None:
            extraction_command.extend(["--limit-assets", str(args.limit_assets)])
        run_command(extraction_command)
    else:
        print(f"Reusing existing shared asset features: {asset_features_path}")

    for target in targets:
        classifier_command = [
            sys.executable,
            "scripts/run_siglip_classifier.py",
            "--labels",
            str(target_train_path(args.train_dir, target)),
            "--features",
            str(asset_features_path),
            "--target",
            target,
            "--output-dir",
            str(output_dir),
            "--folds",
            str(args.folds),
            "--seed",
            str(args.seed),
            "--classifier",
            args.classifier,
            "--model-name",
            f"{slug}_{args.classifier}",
            "--siglip-model",
            args.model_name,
            "--data-version",
            args.data_version,
        ]
        if args.experiment_name is not None:
            classifier_command.extend(["--experiment-name", args.experiment_name])
        if args.no_mlflow:
            classifier_command.append("--no-mlflow")
        if target in PER_ASSET_TYPE_TARGETS:
            classifier_command.append("--per-asset-type")
        run_command(classifier_command)

    print("\nFinished SigLIP runs for requested attributes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
