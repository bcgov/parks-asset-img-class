"""DINOv3 feature extraction helpers.

This module keeps the DINOv3 workflow in small reusable pieces:

1. Resolve project image paths from the CSV ``image_path`` column.
2. Load a pretrained DINOv3 backbone lazily.
3. Convert images into numeric embeddings.
4. Aggregate image-level embeddings into one vector per ``asset_id``.

The first project experiment should use frozen embeddings plus a small
scikit-learn classifier. Fine-tuning can come later if the frozen baseline is
promising.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DINOV3_MODEL = "dinov3_vitb16"
DEFAULT_IMAGE_ROOT = Path("data/processed/images_clean")
FEATURE_PREFIX = "f_"


def resolve_image_path(
    image_path: str | Path,
    *,
    image_root: str | Path = DEFAULT_IMAGE_ROOT,
    repo_root: str | Path | None = None,
) -> Path:
    """Return the most likely local path for a CSV image path.

    The project CSVs store paths like ``data/citywide/images/...``. Cleaned
    images are expected under ``data/processed/images_clean/citywide/images/...``.
    This helper also supports direct paths for local experiments.
    """
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    raw_path = Path(str(image_path))
    image_root_path = Path(image_root)

    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        path_text = raw_path.as_posix()
        if path_text.startswith("data/"):
            candidates.append(root / Path(path_text.replace("data/", f"{image_root_path.as_posix()}/", 1)))
            candidates.append(root / image_root_path / Path(path_text.removeprefix("data/")))

        candidates.append(root / raw_path)
        candidates.append(root / image_root_path / raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def feature_columns(columns: Iterable[str]) -> list[str]:
    """Return embedding feature columns in stable numeric order."""
    return sorted(
        [column for column in columns if column.startswith(FEATURE_PREFIX)],
        key=lambda value: int(value.removeprefix(FEATURE_PREFIX)),
    )


def aggregate_asset_features(image_features: pd.DataFrame) -> pd.DataFrame:
    """Average image-level embeddings into one row per asset."""
    if "asset_id" not in image_features.columns:
        raise ValueError("image_features must contain an 'asset_id' column.")

    features = feature_columns(image_features.columns)
    if not features:
        raise ValueError("image_features does not contain any f_* feature columns.")

    return (
        image_features.groupby("asset_id", as_index=False)[features]
        .mean()
        .sort_values("asset_id")
        .reset_index(drop=True)
    )


def _load_dinov3_model(
    model_name: str = DEFAULT_DINOV3_MODEL,
    *,
    model_source: str | None = None,
    weights: str | Path | None = None,
    device: str | None = None,
) -> tuple[Any, str]:
    import torch

    # Auto-detect cached hub repo to avoid GitHub SSL call
    if model_source is None or model_source == "facebookresearch/dinov3":
        hub_cache = Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main"
        if hub_cache.exists():
            model_source = str(hub_cache)
        else:
            model_source = "facebookresearch/dinov3"

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    source_kind = "local" if Path(model_source).exists() else "github"

    try:
        model = torch.hub.load(
            model_source,
            model_name,
            source=source_kind,
            pretrained=(weights is None),
            trust_repo=True,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise ModuleNotFoundError(
            "DINOv3 could not be loaded because an optional dependency is "
            f"missing: {missing!r}. Update the environment with "
            "`conda env update -f environment.yml --prune`, or install the "
            "missing package directly in the active environment."
        ) from exc
    except Exception as exc:
        if "HTTP Error 403" in str(exc):
            raise RuntimeError(
                "DINOv3 downloaded the repo but could not download the model "
                "checkpoint because the default weight URL returned HTTP 403. "
                "Request DINOv3 model-weight access from Meta, then rerun with "
                "`--weights <CHECKPOINT_URL_OR_LOCAL_PTH>`. If you download the "
                "checkpoint first, pass the local .pth path."
            ) from exc
        raise

    if weights is not None:
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location=resolved_device)
        if isinstance(state_dict, dict):
            state_dict = state_dict.get("model", state_dict.get("state_dict", state_dict))
        model.load_state_dict(state_dict, strict=False)

    model.eval().to(resolved_device)
    return model, resolved_device


def _make_transform(image_size: int = 224) -> Any:
    """Build the image transform used before DINOv3 inference."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _pool_model_output(output: Any) -> Any:
    """Convert common vision-backbone outputs into a 2D feature tensor."""
    if isinstance(output, dict):
        for key in ("x_norm_clstoken", "features", "pooler_output", "last_hidden_state"):
            if key in output:
                output = output[key]
                break
        else:
            output = next(iter(output.values()))

    if isinstance(output, (tuple, list)):
        output = output[0]

    if output.ndim == 3:
        return output[:, 0, :]
    if output.ndim == 4:
        return output.mean(dim=(2, 3))
    if output.ndim == 2:
        return output

    raise ValueError(f"Unsupported DINOv3 output shape: {tuple(output.shape)}")


def extract_image_features(
    rows: pd.DataFrame,
    *,
    image_root: str | Path = DEFAULT_IMAGE_ROOT,
    model_name: str = DEFAULT_DINOV3_MODEL,
    model_source: str | None = None,
    weights: str | Path | None = None,
    device: str | None = None,
    image_size: int = 224,
    repo_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract one DINOv3 embedding per unique image path.

    Returns ``(features, skipped)``. ``skipped`` records missing or unreadable
    images so a batch run can be audited without digging through terminal logs.
    """
    if "asset_id" not in rows.columns or "image_path" not in rows.columns:
        raise ValueError("rows must contain 'asset_id' and 'image_path' columns.")

    import torch
    from PIL import Image, UnidentifiedImageError

    model, resolved_device = _load_dinov3_model(
        model_name=model_name,
        model_source=model_source,
        weights=weights,
        device=device,
    )
    transform = _make_transform(image_size=image_size)

    unique_images = (
        rows[["asset_id", "image_path"]]
        .dropna(subset=["asset_id", "image_path"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    feature_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for row in unique_images.itertuples(index=False):
            resolved_path = resolve_image_path(
                row.image_path,
                image_root=image_root,
                repo_root=repo_root,
            )
            if not resolved_path.exists():
                skipped_rows.append(
                    {
                        "asset_id": row.asset_id,
                        "image_path": row.image_path,
                        "resolved_path": str(resolved_path),
                        "reason": "missing_file",
                    }
                )
                continue

            try:
                image = Image.open(resolved_path).convert("RGB")
            except (OSError, UnidentifiedImageError) as exc:
                skipped_rows.append(
                    {
                        "asset_id": row.asset_id,
                        "image_path": row.image_path,
                        "resolved_path": str(resolved_path),
                        "reason": type(exc).__name__,
                    }
                )
                continue

            batch = transform(image).unsqueeze(0).to(resolved_device)
            embedding = _pool_model_output(model(batch)).squeeze(0).detach().cpu()
            values = embedding.tolist()

            feature_rows.append(
                {
                    "asset_id": row.asset_id,
                    "image_path": row.image_path,
                    "resolved_path": str(resolved_path),
                    **{f"{FEATURE_PREFIX}{idx:04d}": value for idx, value in enumerate(values)},
                }
            )

    return pd.DataFrame(feature_rows), pd.DataFrame(skipped_rows)
