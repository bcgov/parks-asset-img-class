"""SigLIP feature extraction helpers.

SigLIP is used here the same way as DINOv3: as a frozen image encoder that
turns each asset photo into a numeric embedding. The embeddings can then be
averaged per asset and passed to the existing small classifiers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.dinov3_features import (
    DEFAULT_IMAGE_ROOT,
    aggregate_asset_features,
    feature_columns,
    resolve_image_path,
)


DEFAULT_SIGLIP_MODEL = "google/siglip2-base-patch16-224"
FEATURE_PREFIX = "f_"


def model_slug(model_name: str) -> str:
    """Return a filesystem-friendly model slug."""
    return (
        model_name.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def _resolve_device(device: str | None = None) -> str:
    """Choose a local inference device."""
    import torch

    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_siglip_model(
    model_name: str = DEFAULT_SIGLIP_MODEL,
    *,
    device: str | None = None,
) -> tuple[Any, Any, str]:
    """Load a standalone SigLIP vision encoder and processor."""
    import torch
    from transformers import AutoConfig, AutoImageProcessor
    from transformers import Siglip2VisionModel, SiglipVisionModel

    resolved_device = _resolve_device(device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    vision_model_cls = Siglip2VisionModel if config.model_type == "siglip2" else SiglipVisionModel
    model = vision_model_cls.from_pretrained(model_name)
    model.eval().to(resolved_device)

    if resolved_device == "cpu":
        model.to(torch.float32)

    return model, processor, resolved_device


def _pool_siglip_output(output: Any) -> Any:
    """Convert common SigLIP/vision-transformer outputs to a 2D tensor."""
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]
    if isinstance(output, dict):
        for key in ("image_embeds", "pooler_output", "last_hidden_state"):
            if key in output and output[key] is not None:
                value = output[key]
                return value[:, 0, :] if value.ndim == 3 else value
    if isinstance(output, (tuple, list)) and output:
        value = output[0]
        return value[:, 0, :] if value.ndim == 3 else value

    raise ValueError("Unsupported SigLIP model output.")


def _encode_image(model: Any, inputs: dict[str, Any]) -> Any:
    """Run the SigLIP vision path and return one embedding tensor."""
    return _pool_siglip_output(model(**inputs))


def extract_image_features(
    rows: pd.DataFrame,
    *,
    image_root: str | Path = DEFAULT_IMAGE_ROOT,
    model_name: str = DEFAULT_SIGLIP_MODEL,
    device: str | None = None,
    repo_root: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract one SigLIP embedding per unique image path.

    Returns ``(features, skipped)``. ``skipped`` records missing or unreadable
    images so batch runs can be audited.
    """
    if "asset_id" not in rows.columns or "image_path" not in rows.columns:
        raise ValueError("rows must contain 'asset_id' and 'image_path' columns.")

    import torch
    from PIL import Image, ImageOps, UnidentifiedImageError

    model, processor, resolved_device = _load_siglip_model(
        model_name=model_name,
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
                with Image.open(resolved_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
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

            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
            embedding = _encode_image(model, inputs).squeeze(0).detach().cpu()
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


__all__ = [
    "DEFAULT_SIGLIP_MODEL",
    "aggregate_asset_features",
    "extract_image_features",
    "feature_columns",
    "model_slug",
    "resolve_image_path",
]
