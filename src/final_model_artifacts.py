"""Train, save, load, and apply final asset-attribute classifiers.

The final production-style path is:

1. Train one set of lightweight classifiers on frozen asset embeddings.
2. Save the trained classifiers as a versioned local artifact.
3. Load that artifact for final exports and new-image prediction.

The DINOv3/OpenCLIP/SigLIP backbone is not fine-tuned here. The saved models
are the small sklearn classifiers trained on precomputed ``f_*`` embeddings.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from src.attribute_applicability import applicable_profiles_for_target
from src.baseline import infer_target_column
from src.dinov3_classifier import make_classifier
from src.dinov3_features import feature_columns


BUNDLE_FILENAME = "final_classifiers.joblib"
MANIFEST_FILENAME = "manifest.json"
POOLED_MODEL_KEY = "__pooled__"
PER_ASSET_TYPE_TARGETS = {"length_bin", "width_bin", "fall_height_bin"}


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


def _predict_confidence(model: object, X: pd.DataFrame) -> list[object]:
    """Return the model confidence for each predicted class when available."""
    if not hasattr(model, "predict_proba"):
        return [pd.NA] * len(X)
    probabilities = model.predict_proba(X)
    return probabilities.max(axis=1).tolist()


def _labelled_assets(labels: pd.DataFrame, target: str) -> tuple[pd.DataFrame, str]:
    """Return one labelled row per asset for a target column."""
    target_column = infer_target_column(labels, target)
    keep = ["asset_id", target_column]
    if "profile_name" in labels.columns:
        keep.append("profile_name")
    labelled = labels[keep].dropna(subset=[target_column])
    labelled = labelled.drop_duplicates(["asset_id", target_column])
    labelled = labelled.drop_duplicates("asset_id").reset_index(drop=True)
    return labelled, target_column


def _fit_model(
    train: pd.DataFrame,
    features: list[str],
    target_column: str,
    *,
    classifier: str,
    random_state: int,
) -> object | None:
    """Fit one classifier for a target using the configured classifier type."""
    if train[target_column].nunique(dropna=True) < 2:
        return None
    model = make_classifier(classifier=classifier, random_state=random_state)
    model.fit(train[features], train[target_column])
    return model


def train_final_model_bundle(
    *,
    train_dir: Path,
    asset_features: pd.DataFrame,
    targets: Iterable[str],
    classifier: str,
    model_family: str,
    model_name: str,
    random_state: int,
    applicability: dict[str, set[str]],
) -> dict:
    """Train final classifiers on all available labelled asset embeddings."""
    features = feature_columns(asset_features.columns)
    if not features:
        raise ValueError("asset_features must contain f_* feature columns.")

    generated_at = datetime.now(UTC).isoformat()
    bundle: dict[str, object] = {
        "schema_version": 1,
        "generated_at": generated_at,
        "model_family": model_family,
        "model_name": model_name,
        "classifier": classifier,
        "random_state": random_state,
        "feature_columns": features,
        "targets": {},
    }

    for target in targets:
        train_path = train_dir / f"{target}_train.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing train CSV for {target}: {train_path}")

        labels = pd.read_csv(train_path)
        labelled, target_column = _labelled_assets(labels, target)
        train = labelled.merge(
            asset_features[["asset_id", *features]],
            on="asset_id",
            how="inner",
        )
        if train[target_column].nunique(dropna=True) < 2:
            print(f"Skipping {target}: fewer than two classes after feature join.")
            continue

        target_models: dict[str, object] = {}
        mode = "pooled"
        if (
            target in PER_ASSET_TYPE_TARGETS
            and "profile_name" in train.columns
        ):
            mode = "per_asset_type"
            for asset_type in sorted(train["profile_name"].dropna().astype(str).unique()):
                train_subset = train[train["profile_name"].astype(str) == asset_type]
                model = _fit_model(
                    train_subset,
                    features,
                    target_column,
                    classifier=classifier,
                    random_state=random_state,
                )
                if model is None:
                    print(
                        f"  [skip] {target} / '{asset_type}': fewer than two classes."
                    )
                    continue
                target_models[asset_type] = model
        else:
            model = _fit_model(
                train,
                features,
                target_column,
                classifier=classifier,
                random_state=random_state,
            )
            if model is not None:
                target_models[POOLED_MODEL_KEY] = model

        if not target_models:
            print(f"Skipping {target}: no trainable final model.")
            continue

        bundle["targets"][target] = {
            "target": target,
            "target_column": target_column,
            "target_train_file": str(train_path),
            "mode": mode,
            "models": target_models,
            "applicable_profile_names": applicable_profiles_for_target(
                applicability, target
            ),
            "training_n_assets": int(train["asset_id"].nunique()),
            "training_n_labels": int(len(train)),
            "n_features": len(features),
        }

    if not bundle["targets"]:
        raise ValueError("No final classifiers were trained.")
    return bundle


def _manifest_from_bundle(bundle: dict, model_dir: Path) -> dict:
    """Return a JSON-serializable model manifest."""
    targets = {}
    for target, info in bundle["targets"].items():
        targets[target] = {
            key: value
            for key, value in info.items()
            if key != "models"
        }
        targets[target]["model_keys"] = sorted(info["models"])

    return {
        "schema_version": bundle["schema_version"],
        "generated_at": bundle["generated_at"],
        "model_family": bundle["model_family"],
        "model_name": bundle["model_name"],
        "classifier": bundle["classifier"],
        "random_state": bundle["random_state"],
        "n_features": len(bundle["feature_columns"]),
        "feature_columns": bundle["feature_columns"],
        "bundle_path": str(model_dir / BUNDLE_FILENAME),
        "targets": targets,
    }


def save_final_model_bundle(bundle: dict, model_dir: Path) -> tuple[Path, Path]:
    """Save a trained final model bundle and a readable manifest."""
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = model_dir / BUNDLE_FILENAME
    manifest_path = model_dir / MANIFEST_FILENAME
    joblib.dump(bundle, bundle_path)
    manifest_path.write_text(
        json.dumps(_manifest_from_bundle(bundle, model_dir), indent=2),
        encoding="utf-8",
    )
    return bundle_path, manifest_path


def load_final_model_bundle(model_dir: Path) -> dict:
    """Load a final model bundle from a model directory or joblib path."""
    bundle_path = model_dir
    if model_dir.is_dir():
        bundle_path = model_dir / BUNDLE_FILENAME
    if not bundle_path.exists():
        raise FileNotFoundError(f"Final model bundle not found: {bundle_path}")
    return joblib.load(bundle_path)


def _validate_features(bundle: dict, asset_features: pd.DataFrame) -> list[str]:
    """Validate that inference features match the saved model feature schema."""
    features = list(bundle["feature_columns"])
    missing = [feature for feature in features if feature not in asset_features.columns]
    if missing:
        raise ValueError(
            "Asset features are missing columns expected by final model bundle: "
            f"{missing[:5]}"
        )
    return features


def predict_with_final_model_bundle(
    *,
    bundle: dict,
    asset_features: pd.DataFrame,
    asset_metadata: pd.DataFrame,
    predict_all_assets: bool = False,
    high_threshold: float = 0.80,
    medium_threshold: float = 0.60,
) -> pd.DataFrame:
    """Predict all bundle targets for asset-level embeddings."""
    features = _validate_features(bundle, asset_features)
    assets = asset_metadata.merge(
        asset_features[["asset_id", *features]],
        on="asset_id",
        how="inner",
    ).reset_index(drop=True)
    if assets.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    generated_at = datetime.now(UTC).isoformat()
    for target, info in bundle["targets"].items():
        target_assets = assets
        applicable_profiles = info.get("applicable_profile_names")
        if (
            not predict_all_assets
            and applicable_profiles is not None
            and "profile_name" in target_assets.columns
        ):
            if not applicable_profiles:
                continue
            target_assets = target_assets[
                target_assets["profile_name"].astype(str).isin(applicable_profiles)
            ]
        if target_assets.empty:
            continue

        target_assets = target_assets.reset_index(drop=True)
        predictions: list[object] = [pd.NA] * len(target_assets)
        confidence_scores: list[object] = [pd.NA] * len(target_assets)
        models = info["models"]

        if info["mode"] == "per_asset_type" and "profile_name" in target_assets.columns:
            for asset_type, model in models.items():
                mask = target_assets["profile_name"].astype(str) == str(asset_type)
                if not mask.any():
                    continue
                X = target_assets.loc[mask, features]
                preds = model.predict(X).tolist()
                confs = _predict_confidence(model, X)
                for pos, pred, conf in zip(target_assets.index[mask], preds, confs):
                    predictions[pos] = pred
                    confidence_scores[pos] = conf
        else:
            model = models.get(POOLED_MODEL_KEY)
            if model is None:
                continue
            X = target_assets[features]
            predictions = model.predict(X).tolist()
            confidence_scores = _predict_confidence(model, X)

        metadata_columns = [
            column
            for column in [
                "asset_id",
                "profile_id",
                "profile_name",
                "description",
                "image_count",
            ]
            if column in target_assets.columns
        ]
        rows = target_assets[metadata_columns].copy()
        rows["attribute"] = target
        rows["target_column"] = info["target_column"]
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
        rows["model_family"] = bundle["model_family"]
        rows["model_name"] = bundle["model_name"]
        rows["classifier"] = bundle["classifier"]
        rows["target_train_file"] = info["target_train_file"]
        rows["training_n_assets"] = info["training_n_assets"]
        rows["training_n_labels"] = info["training_n_labels"]
        rows["n_features"] = info["n_features"]
        rows["applicable_profile_names"] = (
            "all_profiles"
            if predict_all_assets
            else "|".join(info.get("applicable_profile_names") or [])
        )
        rows["generated_at"] = generated_at
        frames.append(rows)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
