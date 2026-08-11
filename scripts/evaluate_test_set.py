"""Evaluate the final DINOv3 classifier on the held-out test set.

For each attribute this script:
  1. trains the logistic classifier on ALL of that attribute's training data
     (per-asset-type for the binned targets, pooled otherwise),
  2. predicts the held-out test assets (fresh DINOv3 embeddings, asset-level),
  3. scores predictions against the known test labels.

It reuses the same building blocks as the partner export so the test numbers
are directly comparable to the cross-validation numbers reported elsewhere:
  - _fit_and_predict / PER_ASSET_TYPE_TARGETS from export_bcparks_predictions
  - extract_image_features / aggregate_asset_features / feature_columns from dinov3_features
  - weighted/macro F1 via sklearn f1_score(..., zero_division=0), matching CV.

Scoring unit is the ASSET (images aggregated to one embedding per asset, one
prediction per asset). Results are reported one row per attribute. For the
per-asset-type binned targets, predictions across all asset types are pooled
into a single per-attribute score.

Usage:
    python scripts/evaluate_test_set.py \
        --train-dir data/processed/train \
        --test-dir data/processed/test \
        --training-features data/features/dinov3_vitb16_master_assets.csv \
        --weights models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
        --image-root data/processed/images_clean \
        --output results/final/test_set_results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_bcparks_predictions import (  # noqa: E402
    PER_ASSET_TYPE_TARGETS,
    _fit_and_predict,
    _labelled_assets,
)
from src.baseline import first_mode  # noqa: E402
from src.dinov3_features import (  # noqa: E402
    aggregate_asset_features,
    extract_image_features,
    feature_columns,
)

# The 12 model attributes. Categoricals use the attr_<name>_train.csv / _test.csv
# convention; the binned targets mirror train with <name>_bin_train.csv / _test.csv.
ALL_TARGETS = [
    "attr_abutment_material",
    "attr_bridge_type",
    "attr_decking_material",
    "attr_has_edge_guard",
    "attr_has_pedestrian_railing",
    "attr_material_frame_tank_body",
    "attr_structure_material",
    "attr_structure_position",
    "length_bin",
    "width_bin",
    "fall_height_bin",
    "steps_bin",
]


def _score(y_true: list, y_pred: list) -> tuple[float, float, int]:
    """Return (weighted_f1, macro_f1, n) over aligned true/pred labels."""
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return float(weighted), float(macro), len(y_true)


def _baseline_predictions(
    y_train: pd.Series, n_test: int, strategy: str, seed: int = 42
) -> list:
    """Baseline predictions fit on TRAIN labels, emitted for the test set.

    Mirrors src/baseline.py: the baseline learns only from the training labels
    (majority class / observed classes / class frequencies) and applies that to
    the held-out test rows -- exactly how the CV baseline fits on the train fold
    and scores on the validation fold.
    """
    y_train = y_train.dropna()
    if strategy == "majority_class":
        return [first_mode(y_train)] * n_test
    counts = y_train.value_counts()
    classes = counts.index.tolist()
    rng = np.random.default_rng(seed)
    if strategy == "uniform_random":
        return rng.choice(classes, size=n_test).tolist()
    if strategy == "stratified_random":
        probs = (counts / counts.sum()).tolist()
        return rng.choice(classes, size=n_test, p=probs).tolist()
    raise ValueError(f"Unknown baseline strategy: {strategy}")


def _baseline_scores(
    train: pd.DataFrame,
    test_assets: pd.DataFrame,
    target_column: str,
    per_type: bool,
    seed: int = 42,
) -> dict:
    """Compute majority-class + uniform-random baseline F1 on the test set.

    Follows the same per-asset-type vs pooled structure as the model, so the
    baseline is a fair reference: per-type majority fit on each type's training
    labels and scored on that type's test assets (pooled), else pooled overall.
    """
    out = {}
    for strategy in ("majority_class", "uniform_random"):
        y_true: list = []
        y_pred: list = []
        if per_type:
            for atype in sorted(train["profile_name"].dropna().unique()):
                tr = train[train["profile_name"] == atype]
                mask = test_assets["profile_name"].astype(str) == str(atype)
                if not mask.any() or tr[target_column].dropna().empty:
                    continue
                truths = test_assets.loc[mask, target_column].tolist()
                preds = _baseline_predictions(
                    tr[target_column], len(truths), strategy, seed
                )
                y_true.extend(truths)
                y_pred.extend(preds)
        else:
            y_true = test_assets[target_column].tolist()
            y_pred = _baseline_predictions(
                train[target_column], len(y_true), strategy, seed
            )
        if y_true:
            w, m, _ = _score(y_true, y_pred)
        else:
            w, m = float("nan"), float("nan")
        out[f"baseline_{strategy}_weighted_f1"] = round(w, 4)
        out[f"baseline_{strategy}_macro_f1"] = round(m, 4)
    return out


def _build_asset_features(
    rows: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    """Fresh-extract DINOv3 embeddings for the given rows and aggregate to assets."""
    image_features, skipped = extract_image_features(
        rows,
        image_root=args.image_root,
        model_name=args.model,
        weights=args.weights,
        repo_root=ROOT,
    )
    if len(skipped):
        print(f"  [warn] {len(skipped)} image(s) skipped during extraction.")
    if image_features.empty:
        return pd.DataFrame()
    return aggregate_asset_features(image_features)


def evaluate_target(
    target: str, args: argparse.Namespace, train_features: pd.DataFrame
) -> dict | None:
    """Train on all train data, predict the test assets, score one attribute."""
    train_path = Path(args.train_dir) / f"{target}_train.csv"
    test_path = Path(args.test_dir) / f"{target}_test.csv"
    if not train_path.exists():
        print(f"[skip] {target}: missing {train_path}")
        return None
    if not test_path.exists():
        print(f"[skip] {target}: missing {test_path}")
        return None

    features = feature_columns(train_features.columns)
    if not features:
        raise ValueError("training features must contain f_* feature columns.")

    # --- training side: all labelled training assets with cached embeddings ---
    train_labels = pd.read_csv(train_path)
    train_labelled, target_column = _labelled_assets(train_labels, target)
    train = train_labelled.merge(
        train_features[["asset_id", "profile_name", *features]]
        if "profile_name" in train_features.columns
        else train_features[["asset_id", *features]],
        on="asset_id",
        how="inner",
    )
    if train[target_column].nunique(dropna=True) < 2:
        print(f"[skip] {target}: fewer than two training classes.")
        return None

    # --- test side: fresh embeddings for the held-out test images ---
    test_labels = pd.read_csv(test_path)
    test_labelled, _ = _labelled_assets(test_labels, target)
    # keep only test rows that actually have a label for this attribute
    test_labelled = test_labelled[test_labelled[target_column].notna()].copy()
    if test_labelled.empty:
        print(f"[skip] {target}: no labelled test assets.")
        return None

    test_asset_features = _build_asset_features(test_labels, args)
    if test_asset_features.empty:
        print(f"[skip] {target}: no test embeddings extracted.")
        return None

    # attach profile_name to test assets for per-asset-type routing
    profile_lookup = (
        test_labels[["asset_id", "profile_name"]].drop_duplicates("asset_id")
        if "profile_name" in test_labels.columns
        else None
    )
    test_assets = test_labelled.merge(
        test_asset_features[["asset_id", *features]], on="asset_id", how="inner"
    )
    if profile_lookup is not None and "profile_name" not in test_assets.columns:
        test_assets = test_assets.merge(profile_lookup, on="asset_id", how="left")
    test_assets = test_assets.reset_index(drop=True)
    if test_assets.empty:
        print(f"[skip] {target}: no test assets after feature join.")
        return None

    # --- predict ---
    per_type = (
        target in PER_ASSET_TYPE_TARGETS
        and "profile_name" in test_assets.columns
        and "profile_name" in train.columns
    )

    y_true: list = []
    y_pred: list = []

    if per_type:
        # one model per asset type; pool predictions across types for scoring
        for atype in sorted(train["profile_name"].dropna().unique()):
            train_subset = train[train["profile_name"] == atype]
            mask = test_assets["profile_name"].astype(str) == str(atype)
            if not mask.any():
                continue
            preds, _ = _fit_and_predict(
                train_subset, test_assets[mask], features, target_column,
                classifier=args.classifier, random_state=args.seed,
            )
            if not preds:
                print(f"  [skip] {target} / '{atype}': <2 classes; "
                      f"{int(mask.sum())} test asset(s) unscored.")
                continue
            y_true.extend(test_assets.loc[mask, target_column].tolist())
            y_pred.extend(preds)
    else:
        preds, _ = _fit_and_predict(
            train, test_assets, features, target_column,
            classifier=args.classifier, random_state=args.seed,
        )
        if not preds:
            print(f"[skip] {target}: <2 classes at predict time.")
            return None
        y_true = test_assets[target_column].tolist()
        y_pred = preds

    if not y_true:
        print(f"[skip] {target}: no scored test assets.")
        return None

    weighted, macro, n = _score(y_true, y_pred)
    print(f"  {target}: weighted_f1={weighted:.3f} macro_f1={macro:.3f} n={n}")
    result = {
        "attribute": target,
        "weighted_f1": round(weighted, 4),
        "macro_f1": round(macro, 4),
        "n_test_assets": n,
        "per_asset_type": per_type,
    }
    # baselines computed on the SAME test assets (fit on train, scored on test)
    result.update(_baseline_scores(train, test_assets, target_column, per_type, args.seed))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir", type=Path, default=Path("data/processed/train"))
    p.add_argument("--test-dir", type=Path, default=Path("data/processed/test"))
    p.add_argument("--training-features", type=Path, required=True,
                   help="Cached asset-level training embeddings (DINO_MASTER_FEATURES).")
    p.add_argument("--weights", type=Path, required=True,
                   help="DINOv3 weights file.")
    p.add_argument("--model", default="dinov3_vitb16")
    p.add_argument("--image-root", type=str, default="data/processed/images_clean")
    p.add_argument("--classifier", default="logistic_regression")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path,
                   default=Path("results/final/test_set_results.csv"))
    p.add_argument("--targets", nargs="+", default=None,
                   help="Subset of attributes to evaluate (default: all 12).")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.training_features.exists():
        print(f"Training features not found: {args.training_features}", file=sys.stderr)
        return 1
    train_features = pd.read_csv(args.training_features)

    targets = args.targets if args.targets else ALL_TARGETS
    rows: list[dict] = []
    for target in targets:
        print(f"Evaluating {target} ...")
        result = evaluate_target(target, args, train_features)
        if result is not None:
            rows.append(result)

    if not rows:
        print("No attributes scored.", file=sys.stderr)
        return 1

    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    # quick summary: structural (categorical) vs measurement (binned) means
    binned = results["attribute"].isin(PER_ASSET_TYPE_TARGETS | {"steps_bin"})
    cat_mean = results.loc[~binned, "weighted_f1"].mean()
    meas_mean = results.loc[binned, "weighted_f1"].mean()
    print("\nTest-set evaluation complete.")
    print(f"  attributes scored:            {len(results)}")
    print(f"  categorical mean weighted F1: {cat_mean:.3f}")
    print(f"  measurement mean weighted F1: {meas_mean:.3f}")
    print(f"  results written to:           {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())