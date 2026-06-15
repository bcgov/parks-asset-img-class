"""Export final asset-attribute predictions for BC Parks.

This script trains one final classifier per target attribute using all available
labels, then predicts labels for asset-level embeddings from the master dataset.
It writes:

- a long CSV with one row per asset x predicted attribute
- a wide CSV with one row per asset and prediction/confidence columns per target

The default model path is intentionally non-VLM: frozen image embeddings plus a
small supervised classifier. Confidence is included only when the classifier
supports ``predict_proba``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline import DEFAULT_CLASSIFICATION_TARGETS, infer_target_column  # noqa: E402
from src.dinov3_classifier import CLASSIFIER_CHOICES, make_classifier  # noqa: E402
from src.dinov3_features import feature_columns  # noqa: E402


DEFAULT_METADATA_COLUMNS = [
    "asset_id",
    "profile_id",
    "profile_name",
    "description",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master",
        type=Path,
        default=Path("data/processed/master_dataset.csv"),
        help="Master asset/image CSV.",
    )
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
        "--targets",
        nargs="+",
        default=DEFAULT_CLASSIFICATION_TARGETS,
        help="Targets to train and export.",
    )
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIER_CHOICES,
        default="logistic_regression",
        help="Final supervised classifier trained on frozen embeddings.",
    )
    parser.add_argument("--model-family", default="dinov3")
    parser.add_argument("--model-name", default="dinov3_vitb16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--predict-all-assets",
        action="store_true",
        help=(
            "Predict every asset for every target. By default, each target is "
            "limited to profile_name values observed in that target's train CSV."
        ),
    )
    parser.add_argument("--high-threshold", type=float, default=0.80)
    parser.add_argument("--medium-threshold", type=float, default=0.60)
    parser.add_argument(
        "--output-long",
        type=Path,
        default=Path("results/final/bcparks_asset_attribute_predictions_long.csv"),
    )
    parser.add_argument(
        "--output-wide",
        type=Path,
        default=Path("results/final/bcparks_asset_attribute_predictions_wide.csv"),
    )
    return parser.parse_args()


def confidence_level(
    score: object,
    *,
    high_threshold: float = 0.80,
    medium_threshold: float = 0.60,
) -> str:
    """Bucket a probability-like confidence score."""
    if pd.isna(score):
        return "unavailable"
    value = float(score)
    if value >= high_threshold:
        return "high"
    if value >= medium_threshold:
        return "medium"
    return "low"


def build_asset_metadata(master: pd.DataFrame) -> pd.DataFrame:
    """Return one metadata row per asset_id plus an image count."""
    if "asset_id" not in master.columns:
        raise ValueError("master dataset must contain an 'asset_id' column.")

    columns = [column for column in DEFAULT_METADATA_COLUMNS if column in master.columns]
    metadata = (
        master[columns]
        .drop_duplicates("asset_id")
        .sort_values("asset_id")
        .reset_index(drop=True)
    )
    image_counts = (
        master.groupby("asset_id", as_index=False)
        .size()
        .rename(columns={"size": "image_count"})
    )
    return metadata.merge(image_counts, on="asset_id", how="left")


def _labelled_assets(labels: pd.DataFrame, target: str) -> tuple[pd.DataFrame, str]:
    target_column = infer_target_column(labels, target)
    labelled = labels[["asset_id", target_column]].dropna(subset=[target_column])
    labelled = labelled.drop_duplicates(["asset_id", target_column])
    labelled = labelled.drop_duplicates("asset_id").reset_index(drop=True)
    return labelled, target_column


def _applicable_profiles(labels: pd.DataFrame) -> list[str]:
    if "profile_name" not in labels.columns:
        return []
    return sorted(labels["profile_name"].dropna().astype(str).unique().tolist())


def _filter_applicable_assets(
    assets: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    predict_all_assets: bool,
) -> tuple[pd.DataFrame, str]:
    profiles = _applicable_profiles(labels)
    if predict_all_assets or "profile_name" not in assets.columns or not profiles:
        return assets, "all_profiles"
    filtered = assets[assets["profile_name"].astype(str).isin(profiles)].copy()
    return filtered, "|".join(profiles)


def _predict_confidence(model: object, X: pd.DataFrame) -> list[object]:
    if not hasattr(model, "predict_proba"):
        return [pd.NA] * len(X)
    probabilities = model.predict_proba(X)
    return probabilities.max(axis=1).tolist()


def predict_target(
    *,
    target: str,
    train_dir: Path,
    asset_features: pd.DataFrame,
    asset_metadata: pd.DataFrame,
    classifier: str,
    model_family: str,
    model_name: str,
    random_state: int,
    generated_at: str,
    predict_all_assets: bool,
    high_threshold: float,
    medium_threshold: float,
) -> pd.DataFrame:
    """Train a final classifier for one target and return prediction rows."""
    train_path = train_dir / f"{target}_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train CSV for {target}: {train_path}")

    labels = pd.read_csv(train_path)
    labelled, target_column = _labelled_assets(labels, target)
    features = feature_columns(asset_features.columns)
    if not features:
        raise ValueError("asset_features must contain f_* feature columns.")

    train = labelled.merge(
        asset_features[["asset_id", *features]],
        on="asset_id",
        how="inner",
    )
    if train[target_column].nunique(dropna=True) < 2:
        print(f"Skipping {target}: fewer than two classes after feature join.")
        return pd.DataFrame()

    model = make_classifier(classifier=classifier, random_state=random_state)
    model.fit(train[features], train[target_column])

    assets = asset_metadata.merge(
        asset_features[["asset_id", *features]],
        on="asset_id",
        how="inner",
    )
    assets, applicable_profiles = _filter_applicable_assets(
        assets,
        labels,
        predict_all_assets=predict_all_assets,
    )
    if assets.empty:
        print(f"Skipping {target}: no applicable assets with features.")
        return pd.DataFrame()

    X = assets[features]
    predictions = model.predict(X)
    confidence_scores = _predict_confidence(model, X)

    rows = assets[
        [column for column in DEFAULT_METADATA_COLUMNS if column in assets.columns]
        + ["image_count"]
    ].copy()
    rows["attribute"] = target
    rows["target_column"] = target_column
    rows["predicted_value"] = predictions
    rows["confidence_score"] = confidence_scores
    rows["confidence_level"] = [
        confidence_level(
            score,
            high_threshold=high_threshold,
            medium_threshold=medium_threshold,
        )
        for score in confidence_scores
    ]
    rows["model_family"] = model_family
    rows["model_name"] = model_name
    rows["classifier"] = classifier
    rows["feature_source"] = str(asset_features.attrs.get("source_path", ""))
    rows["target_train_file"] = str(train_path)
    rows["training_n_assets"] = int(train["asset_id"].nunique())
    rows["training_n_labels"] = int(len(train))
    rows["n_features"] = len(features)
    rows["applicable_profile_names"] = applicable_profiles
    rows["generated_at"] = generated_at
    return rows


def build_wide_output(long_predictions: pd.DataFrame, asset_metadata: pd.DataFrame) -> pd.DataFrame:
    """Return one row per asset with prediction/confidence columns per target."""
    wide = asset_metadata.copy()
    if long_predictions.empty:
        return wide

    for attribute, group in long_predictions.groupby("attribute", sort=True):
        by_asset = group.drop_duplicates("asset_id").set_index("asset_id")
        wide[f"{attribute}_prediction"] = wide["asset_id"].map(by_asset["predicted_value"])
        wide[f"{attribute}_confidence_score"] = wide["asset_id"].map(by_asset["confidence_score"])
        wide[f"{attribute}_confidence_level"] = wide["asset_id"].map(by_asset["confidence_level"])

    return wide


def export_predictions(
    *,
    master_path: Path,
    features_path: Path,
    train_dir: Path,
    targets: Iterable[str],
    classifier: str,
    model_family: str,
    model_name: str,
    random_state: int,
    predict_all_assets: bool,
    high_threshold: float,
    medium_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = pd.read_csv(master_path)
    asset_metadata = build_asset_metadata(master)
    asset_features = pd.read_csv(features_path)
    asset_features.attrs["source_path"] = str(features_path)

    generated_at = datetime.now(UTC).isoformat()
    frames = []
    for target in targets:
        frame = predict_target(
            target=target,
            train_dir=train_dir,
            asset_features=asset_features,
            asset_metadata=asset_metadata,
            classifier=classifier,
            model_family=model_family,
            model_name=model_name,
            random_state=random_state,
            generated_at=generated_at,
            predict_all_assets=predict_all_assets,
            high_threshold=high_threshold,
            medium_threshold=medium_threshold,
        )
        if not frame.empty:
            frames.append(frame)

    long_predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    wide_predictions = build_wide_output(long_predictions, asset_metadata)
    return long_predictions, wide_predictions


def main() -> int:
    args = parse_args()
    long_predictions, wide_predictions = export_predictions(
        master_path=args.master,
        features_path=args.features,
        train_dir=args.train_dir,
        targets=args.targets,
        classifier=args.classifier,
        model_family=args.model_family,
        model_name=args.model_name,
        random_state=args.seed,
        predict_all_assets=args.predict_all_assets,
        high_threshold=args.high_threshold,
        medium_threshold=args.medium_threshold,
    )

    args.output_long.parent.mkdir(parents=True, exist_ok=True)
    args.output_wide.parent.mkdir(parents=True, exist_ok=True)
    long_predictions.to_csv(args.output_long, index=False)
    wide_predictions.to_csv(args.output_wide, index=False)

    print(f"Wrote {len(long_predictions)} long prediction rows to {args.output_long}")
    print(f"Wrote {len(wide_predictions)} wide asset rows to {args.output_wide}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
