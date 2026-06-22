"""Train and save final classifiers for production-style inference.

Pipeline role:
- reads frozen asset embeddings produced by ``scripts/extract_dinov3_features.py``
  or ``scripts/build_asset_features_from_image_features.py``;
- reads label CSVs from ``data/processed/train`` and the applicability matrix
  from ``data/processed/attribute_applicability.csv``;
- writes a reusable sklearn bundle under ``models/final/...``.

The saved bundle is consumed by ``scripts/export_bcparks_predictions.py`` for
partner deliverables and by ``scripts/predict_new_images.py`` for unseen image
folders. This separates final training from inference/export so ``make all``
does not need to retrain classifiers when the saved bundle is current.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attribute_applicability import load_applicability  # noqa: E402
from src.baseline import DEFAULT_CLASSIFICATION_TARGETS  # noqa: E402
from src.dinov3_classifier import CLASSIFIER_CHOICES  # noqa: E402
from src.final_model_artifacts import (  # noqa: E402
    save_final_model_bundle,
    train_final_model_bundle,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/dinov3_vitb16_master_assets.csv"),
        help="Asset-level feature CSV containing asset_id and f_* columns.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/processed/train"),
        help="Directory containing <target>_train.csv files.",
    )
    parser.add_argument(
        "--applicability",
        type=Path,
        default=Path("data/processed/attribute_applicability.csv"),
        help="CSV matrix mapping attributes to asset types.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_CLASSIFICATION_TARGETS,
        help="Targets to train. Defaults to all final classification targets.",
    )
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIER_CHOICES,
        default="logistic_regression",
    )
    parser.add_argument("--model-family", default="dinov3")
    parser.add_argument("--model-name", default="dinov3_vitb16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/final/dinov3_vitb16_logistic_regression"),
        help="Directory where final_classifiers.joblib and manifest.json are written.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the script from parsed command-line arguments."""
    args = parse_args()
    asset_features = pd.read_csv(args.features)
    applicability = load_applicability(args.applicability)
    bundle = train_final_model_bundle(
        train_dir=args.train_dir,
        asset_features=asset_features,
        targets=args.targets,
        classifier=args.classifier,
        model_family=args.model_family,
        model_name=args.model_name,
        random_state=args.seed,
        applicability=applicability,
    )
    bundle_path, manifest_path = save_final_model_bundle(bundle, args.model_dir)
    print(f"Wrote final model bundle to {bundle_path}")
    print(f"Wrote final model manifest to {manifest_path}")
    print(f"Trained {len(bundle['targets'])} target classifier group(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
