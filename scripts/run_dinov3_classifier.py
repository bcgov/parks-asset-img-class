"""Run a grouped classifier on frozen DINOv3 asset embeddings.

Example:
    python scripts/run_dinov3_classifier.py \
        --labels data/processed/train/attr_decking_material_train.csv \
        --features data/features/dinov3_vitb16_attr_decking_material_assets.csv \
        --target attr_decking_material
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dinov3_classifier import run_task_from_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen DINOv3 embeddings with grouped CV."
    )
    parser.add_argument("--labels", type=Path, required=True, help="Task train CSV.")
    parser.add_argument("--features", type=Path, required=True, help="Asset-level DINOv3 feature CSV.")
    parser.add_argument("--target", required=True, help="Target column, for example attr_decking_material.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, folds = run_task_from_files(
        labels_path=args.labels,
        features_path=args.features,
        target=args.target,
        n_splits=args.folds,
        random_state=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"dinov3_{args.target}_classification_results.csv"
    folds_path = args.output_dir / f"dinov3_{args.target}_classification_cv_folds.csv"
    summary.to_csv(summary_path, index=False)
    folds.to_csv(folds_path, index=False)

    print(f"Wrote {len(summary)} summary rows to {summary_path}")
    print(f"Wrote {len(folds)} fold rows to {folds_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

