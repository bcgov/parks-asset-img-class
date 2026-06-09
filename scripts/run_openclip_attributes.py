"""Run OpenCLIP experiments for all classification attributes.

The expensive step is OpenCLIP feature extraction. Because OpenCLIP features
do not depend on the target label, this script extracts one shared feature
table from the union of all train CSVs, then reuses it for each attribute.

Default model is ViT-B-16 (~86M params) to match DINOv3 ViT-B/16 and SigLIP2
base for a fair, size-matched comparison.

Usage:
    python scripts/run_openclip_attributes.py

To skip MLflow:
    python scripts/run_openclip_attributes.py --no-mlflow

To use a different classifier:
    python scripts/run_openclip_attributes.py --classifier random_forest

To force re-extraction even if features already exist:
    python scripts/run_openclip_attributes.py --force-extract
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


DEFAULT_TARGETS = [
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
        description="Extract shared OpenCLIP features and run classifiers for all attributes."
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
        help="Directory where shared OpenCLIP feature CSVs are written.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/openclip_results"),
        help="Directory where classifier metric CSVs are written.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("results/openclip_results/predictions"),
        help="Directory where prediction CSVs are written.",
    )
    parser.add_argument(
        "--model",
        default="ViT-B-16",
        help="OpenCLIP model architecture.",
    )
    parser.add_argument(
        "--pretrained",
        default="laion2b_s34b_b88k",
        help="OpenCLIP pretrained weights tag.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/processed/images_clean"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classifier",
        choices=[
            "logistic_regression",
            "linear_svm",
            "random_forest",
            "hist_gradient_boosting",
        ],
        default="logistic_regression",
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
        "--force-extract",
        action="store_true",
        help="Recreate the shared OpenCLIP feature files even if they exist.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip DagsHub/MLflow logging.",
    )
    return parser.parse_args()


def selected_targets(args: argparse.Namespace) -> list[str]:
    if args.targets is not None:
        return args.targets
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

    # Sanitize model name for filenames (e.g. ViT-B-16 -> vitb16)
    model_slug = args.model.lower().replace("-", "").replace("/", "")

    args.feature_dir.mkdir(parents=True, exist_ok=True)
    union_path = args.feature_dir / f"openclip_{model_slug}_all_attributes_union_input.csv"
    image_features_path = args.feature_dir / f"openclip_{model_slug}_all_attributes_images.csv"
    asset_features_path = args.feature_dir / f"openclip_{model_slug}_all_attributes_assets.csv"

    print("Targets, in order:")
    for index, target in enumerate(targets, start=1):
        print(f"{index}. {target}")

    # Step 1 — extract shared features once for all attributes
    if args.force_extract or not asset_features_path.exists():
        build_union_input(targets, args.train_dir, union_path)
        extraction_command = [
            sys.executable,
            "scripts/extract_openclip_features.py",
            "--input", str(union_path),
            "--output", str(image_features_path),
            "--asset-output", str(asset_features_path),
            "--model", args.model,
            "--pretrained", args.pretrained,
            "--image-root", str(args.image_root),
        ]
        if args.device is not None:
            extraction_command.extend(["--device", args.device])
        run_command(extraction_command)
    else:
        print(f"Reusing existing shared asset features: {asset_features_path}")

    # Step 2 — run the OpenCLIP classifier for each attribute
    for target in targets:
        classifier_command = [
            sys.executable,
            "scripts/run_openclip_classifier.py",
            "--labels", str(target_train_path(args.train_dir, target)),
            "--features", str(asset_features_path),
            "--target", target,
            "--output-dir", str(args.output_dir),
            "--predictions-dir", str(args.predictions_dir),
            "--folds", str(args.folds),
            "--seed", str(args.seed),
            "--classifier", args.classifier,
            "--model-name", f"openclip_{model_slug}_{args.classifier}",
            "--data-version", args.data_version,
        ]
        if args.experiment_name is not None:
            classifier_command.extend(["--experiment-name", args.experiment_name])
        if args.no_mlflow:
            classifier_command.append("--no-mlflow")
        run_command(classifier_command)

    print("\nFinished OpenCLIP runs for all requested attributes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())