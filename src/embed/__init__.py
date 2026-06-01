"""Image embedding extraction (DINOv3, DINOv2, ...).

The extracted features are written to a parquet cache under
``data/features/<model_slug>.parquet`` keyed by ``image_path`` so they
can be reused across pipelines (DINOv3 head, k-NN, YOLO crop
classifier, stacking).
"""

from src.embed.dinov3 import (
    DEFAULT_DINOV3_MODEL,
    FeatureCache,
    extract_features_for_split,
    load_dinov3,
    load_features,
    save_features,
    slug_for_model,
)

__all__ = [
    "DEFAULT_DINOV3_MODEL",
    "FeatureCache",
    "extract_features_for_split",
    "load_dinov3",
    "load_features",
    "save_features",
    "slug_for_model",
]
