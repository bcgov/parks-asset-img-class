"""Run the DINO embedding + k-NN attribute predictor and log to MLflow.

Usage (after ``scripts/build_features.py``):
    python scripts/run_vectordb_knn.py
    python scripts/run_vectordb_knn.py --k 5
    python scripts/run_vectordb_knn.py --model facebook/dinov2-large
    python scripts/run_vectordb_knn.py --no-mlflow
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.splits import DEFAULT_SPLIT_SEED, load_split  # noqa: E402
from src.embed import DEFAULT_DINOV3_MODEL, load_features  # noqa: E402
from src.models import run_pipeline  # noqa: E402
from src.models.vectordb_knn import DEFAULT_K, predict  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_DINOV3_MODEL)
    p.add_argument("--features-dir", type=Path, default=Path("data/features"))
    p.add_argument("--feature-suffix", default="")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--data-version", default="processed-main")
    p.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--predictions-dir", type=Path, default=Path("data/predictions"))
    p.add_argument(
        "--predict-train",
        action="store_true",
        help="Also write __train.csv via OOF (k-NN training-set neighbours are excluded by the K-fold).",
    )
    p.add_argument("--train-n-folds", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cache = load_features(args.model, out_dir=args.features_dir, suffix=args.feature_suffix)
    train_df, test_df = load_split(processed_dir=args.data_dir, split_seed=args.split_seed)
    print(
        f"Loaded {len(cache.df)}-image feature cache (dim={cache.dim}); "
        f"k={args.k}; train={len(train_df)} test={len(test_df)}"
    )

    def predict_fn(_train_df, _test_df, schema):
        return predict(_train_df, _test_df, schema, feature_cache=cache, k=args.k)

    backbone_slug = args.model.split("/")[-1].replace("-", "_").lower()
    model_name = f"knn_k{args.k}__{backbone_slug}"
    result = run_pipeline(
        pipeline="vectordb_knn",
        model_family="dinov3",
        model_name=model_name,
        train_df=train_df,
        test_df=test_df,
        predict_fn=predict_fn,
        data_version=args.data_version,
        split_seed=args.split_seed,
        params={"backbone": args.model, "k": args.k},
        extra_tags={"backbone": args.model, "k": str(args.k)},
        log_to_mlflow=not args.no_mlflow,
        predictions_dir=args.predictions_dir,
        produce_train_predictions=args.predict_train,
        train_prediction_mode="oof",
        train_n_folds=args.train_n_folds,
    )

    print("\nPer-attribute report:")
    print(result.report.per_attribute_table().to_string(index=False))
    print("\nAggregate:")
    for k, v in result.report.aggregate().items():
        print(f"  {k}: {v:.4f}")
    if result.mlflow_run_id is not None:
        print(f"\nMLflow run: {result.mlflow_run_id}")
    print(f"Predictions CSV: {result.extras['predictions_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
