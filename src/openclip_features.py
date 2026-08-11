"""OpenCLIP feature extraction helpers.

Mirrors the structure of src/dinov3_features.py but uses OpenCLIP as the
backbone. Produces the same output format (asset_id + f_* columns) so the
downstream classifier scripts work without any changes.

Install:
    pip install open-clip-torch
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.dinov3_features import (
    FEATURE_PREFIX,
    aggregate_asset_features,
    feature_columns,
    resolve_image_path,
)

DEFAULT_OPENCLIP_MODEL = "ViT-B-16"
DEFAULT_OPENCLIP_PRETRAINED = "laion2b_s34b_b88k"
DEFAULT_IMAGE_ROOT = Path("data/processed/images_clean")


def _load_openclip_model(
    model_name: str = DEFAULT_OPENCLIP_MODEL,
    pretrained: str = DEFAULT_OPENCLIP_PRETRAINED,
    *,
    device: str | None = None,
) -> tuple[Any, Any, str]:
    """Load an OpenCLIP model and its image preprocessor.

    Returns (model, preprocess, device).
    Model weights are downloaded automatically on first run and cached in
    ~/.cache/huggingface/ — no manual download or access request needed.
    """
    try:
        import open_clip
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCLIP is not installed. Run: pip install open-clip-torch"
        ) from exc

    import torch

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    model.eval().to(resolved_device)
    return model, preprocess, resolved_device


def extract_openclip_features(
    rows: pd.DataFrame,
    *,
    image_root: str | Path = DEFAULT_IMAGE_ROOT,
    model_name: str = DEFAULT_OPENCLIP_MODEL,
    pretrained: str = DEFAULT_OPENCLIP_PRETRAINED,
    device: str | None = None,
    repo_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract one OpenCLIP embedding per unique image path.

    Returns ``(features, skipped)``. ``skipped`` records missing or unreadable
    images so a batch run can be audited without digging through terminal logs.

    Output format is identical to extract_image_features() in dinov3_features.py:
    asset_id, image_path, resolved_path, f_0000, f_0001, ...
    """
    if "asset_id" not in rows.columns or "image_path" not in rows.columns:
        raise ValueError("rows must contain 'asset_id' and 'image_path' columns.")

    import torch
    from PIL import Image, UnidentifiedImageError

    model, preprocess, resolved_device = _load_openclip_model(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
    )

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

            batch = preprocess(image).unsqueeze(0).to(resolved_device)
            embedding = model.encode_image(batch)
            # Normalize — recommended for linear classifiers
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            values = embedding.squeeze(0).detach().cpu().tolist()

            feature_rows.append(
                {
                    "asset_id": row.asset_id,
                    "image_path": row.image_path,
                    "resolved_path": str(resolved_path),
                    **{f"{FEATURE_PREFIX}{idx:04d}": value for idx, value in enumerate(values)},
                }
            )

    return pd.DataFrame(feature_rows), pd.DataFrame(skipped_rows)


__all__ = [
    "DEFAULT_OPENCLIP_MODEL",
    "DEFAULT_OPENCLIP_PRETRAINED",
    "DEFAULT_IMAGE_ROOT",
    "extract_openclip_features",
    "aggregate_asset_features",
    "feature_columns",
]