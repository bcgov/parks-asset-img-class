"""DINOv3 / DINOv2 feature extraction with a per-image parquet cache.

Heavy lifting is run **once** on the DGX Spark via
``scripts/build_features.py`` and the resulting parquet (one row per
image, one column per embedding dim plus ``image_path``) is then read
by every downstream pipeline (DINOv3 heads, k-NN, YOLO crop
classifier, stacking).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.data.splits import absolute_image_path

logger = logging.getLogger(__name__)

DEFAULT_DINOV3_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_FEATURES_DIR = Path("data/features")


def slug_for_model(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_").lower()


def _select_device(requested: str | None = None) -> str:
    import torch

    if requested is not None:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class FeatureCache:
    """One DataFrame keyed by ``image_path`` plus 1..D embedding columns."""

    df: pd.DataFrame
    model_id: str

    @property
    def dim(self) -> int:
        return len([c for c in self.df.columns if c.startswith("f_")])

    @property
    def image_paths(self) -> pd.Series:
        return self.df["image_path"]

    def features(self) -> np.ndarray:
        cols = sorted(
            [c for c in self.df.columns if c.startswith("f_")],
            key=lambda c: int(c.split("_", 1)[1]),
        )
        return self.df[cols].to_numpy(dtype=np.float32)

    def aligned_to(self, image_paths: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(features, missing_mask)`` aligned to ``image_paths``.

        Rows whose image_path is missing in the cache get all-NaN
        features and ``True`` in ``missing_mask``.  Duplicate image_paths
        in the cache are deduplicated (first occurrence kept).
        """
        deduped = self.df.drop_duplicates(subset="image_path", keep="first")
        lookup = deduped.set_index("image_path")
        cols = sorted(
            [c for c in lookup.columns if c.startswith("f_")],
            key=lambda c: int(c.split("_", 1)[1]),
        )
        feature_matrix = lookup[cols].to_numpy(dtype=np.float32)
        index_map = {p: i for i, p in enumerate(lookup.index)}

        paths_list = list(image_paths)
        out = np.full((len(paths_list), len(cols)), np.nan, dtype=np.float32)
        missing = np.zeros(len(out), dtype=bool)
        for i, p in enumerate(paths_list):
            idx = index_map.get(p)
            if idx is None:
                missing[i] = True
            else:
                out[i] = feature_matrix[idx]
        return out, missing


def load_dinov3(
    model_id: str = DEFAULT_DINOV3_MODEL,
    *,
    device: str | None = None,
    dtype: str | None = None,
) -> tuple[Any, Any, str]:
    """Load a DINOv3 (or DINOv2) backbone + image processor.

    Returns ``(model, processor, resolved_device)``.
    """
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = _select_device(device)
    torch_dtype = None
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype).to(device).eval()
    return model, processor, device


def _load_image(image_path: str, repo_root: Path | None = None) -> Image.Image | None:
    p = absolute_image_path(image_path, repo_root=repo_root)
    if not p.exists():
        return None
    try:
        img = Image.open(p).convert("RGB")
        return ImageOps.exif_transpose(img)
    except (OSError, ValueError) as exc:
        logger.warning("Failed to load %s: %s", p, exc)
        return None


def _pool_features(outputs: Any) -> Any:
    """Pull the CLS / pooler representation out of a transformers output."""
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    last_hidden = outputs.last_hidden_state
    # First token is the CLS / class token in DINOv2 / DINOv3 ViT.
    return last_hidden[:, 0, :]


def extract_features_for_split(
    df: pd.DataFrame,
    *,
    model_id: str = DEFAULT_DINOV3_MODEL,
    device: str | None = None,
    dtype: str | None = None,
    batch_size: int = 16,
    max_images: int | None = None,
    repo_root: Path | None = None,
    extractor: Any | None = None,
) -> FeatureCache:
    """Compute one embedding per ``image_path`` row in ``df``.

    The caller decides whether ``df`` is the full corpus, train-only,
    test-only, etc.  Embeddings are L2-normalised so cosine similarity
    is a simple dot product downstream.

    Pass an ``extractor`` callable for testing — it should accept a
    list of PIL images and return ``(N, D)`` numpy features.
    """
    import torch

    paths = df["image_path"].tolist()
    if max_images is not None:
        paths = paths[:max_images]

    if extractor is None:
        model, processor, device = load_dinov3(model_id, device=device, dtype=dtype)

        @torch.no_grad()
        def _extract(batch_images: list[Image.Image]) -> np.ndarray:
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            feats = _pool_features(outputs)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return feats.float().cpu().numpy()

        extractor_fn = _extract
    else:
        extractor_fn = extractor

    rows: list[dict] = []
    batch_imgs: list[Image.Image] = []
    batch_paths: list[str] = []

    def _flush() -> None:
        nonlocal batch_imgs, batch_paths
        if not batch_imgs:
            return
        feats = extractor_fn(batch_imgs)
        for path, vec in zip(batch_paths, feats, strict=False):
            row = {"image_path": path}
            row.update({f"f_{i}": float(v) for i, v in enumerate(vec)})
            rows.append(row)
        batch_imgs = []
        batch_paths = []

    for path in paths:
        img = _load_image(path, repo_root=repo_root)
        if img is None:
            continue
        batch_imgs.append(img)
        batch_paths.append(path)
        if len(batch_imgs) >= batch_size:
            _flush()
    _flush()

    return FeatureCache(df=pd.DataFrame(rows), model_id=model_id)


def save_features(
    cache: FeatureCache,
    *,
    out_dir: str | Path = DEFAULT_FEATURES_DIR,
    suffix: str = "",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slug_for_model(cache.model_id)
    if suffix:
        slug = f"{slug}__{suffix}"
    p = out_dir / f"{slug}.parquet"
    cache.df.to_parquet(p, index=False)
    return p


def load_features(
    model_id: str = DEFAULT_DINOV3_MODEL,
    *,
    out_dir: str | Path = DEFAULT_FEATURES_DIR,
    suffix: str = "",
) -> FeatureCache:
    out_dir = Path(out_dir)
    slug = slug_for_model(model_id)
    if suffix:
        slug = f"{slug}__{suffix}"
    p = out_dir / f"{slug}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No feature cache at {p}. Run scripts/build_features.py first.")
    return FeatureCache(df=pd.read_parquet(p), model_id=model_id)
