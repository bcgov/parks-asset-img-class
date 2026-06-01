"""Vector-DB k-NN attribute prediction over the DINOv3 feature cache.

For each test image we retrieve the top-k most similar **train** images
(by cosine similarity of the cached DINOv3 embeddings) and, per
attribute, take the majority-vote / mean of the neighbours' labels.
The mean cosine similarity of the retrieved neighbours is logged as a
confidence proxy.

Key design points:
- Features are already L2-normalised at extraction time, so cosine
  similarity is a single ``A @ B.T`` matmul — no Qdrant install needed.
- Retrieval is restricted to neighbours of the **same asset type** as
  the query image (no point voting with stair images for a bridge's
  bridge_type).
- The same retrieval index can be reused as a "find similar BC Parks
  asset" tool — we expose ``retrieve_similar_assets`` for that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.schema import AttributeKind, Schema, load_schema
from src.embed.dinov3 import FeatureCache
from src.models.base import predict_per_asset_type

logger = logging.getLogger(__name__)

DEFAULT_K = 10
from src.utils.labels import MISSING_TOKENS as _MISSING_TOKENS  # re-export


@dataclass
class KNNIndex:
    """Cosine k-NN index over a feature matrix, scoped to a single asset type."""

    feature_matrix: np.ndarray
    image_paths: np.ndarray

    def query(self, query_features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(top_k_indices, top_k_similarities)`` shapes (N_q, k)."""
        if self.feature_matrix.shape[0] == 0 or query_features.shape[0] == 0:
            return (
                np.zeros((query_features.shape[0], 0), dtype=int),
                np.zeros((query_features.shape[0], 0), dtype=float),
            )
        sims = query_features @ self.feature_matrix.T
        k_eff = min(k, self.feature_matrix.shape[0])
        top_idx = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
        rows = np.arange(sims.shape[0])[:, None]
        top_sims = sims[rows, top_idx]
        order = np.argsort(-top_sims, axis=1)
        top_idx = top_idx[rows, order]
        top_sims = top_sims[rows, order]
        return top_idx, top_sims


def _build_index(train_subset: pd.DataFrame, cache: FeatureCache) -> KNNIndex:
    feats, missing = cache.aligned_to(train_subset["image_path"].tolist())
    keep = ~missing
    return KNNIndex(
        feature_matrix=feats[keep],
        image_paths=train_subset.loc[train_subset.index[keep], "image_path"].to_numpy(),
    )


def _query_features(
    test_subset: pd.DataFrame, cache: FeatureCache
) -> tuple[np.ndarray, pd.Index]:
    feats, missing = cache.aligned_to(test_subset["image_path"].tolist())
    keep = ~missing
    return feats[keep], test_subset.index[keep]


from src.utils.labels import clean_labels as _clean_labels  # re-export


def _knn_one_attribute(
    asset_type_id: str,
    attribute_column: str,
    train_subset: pd.DataFrame,
    test_subset: pd.DataFrame,
    *,
    schema: Schema,
    cache: FeatureCache,
    k: int,
) -> pd.Series | None:
    if attribute_column not in train_subset.columns:
        return None

    kind = schema.kind_of(attribute_column)
    index = _build_index(train_subset, cache)
    if index.feature_matrix.shape[0] == 0:
        return None

    query_feats, test_index = _query_features(test_subset, cache)
    if query_feats.shape[0] == 0:
        return None

    top_idx, _ = index.query(query_feats, k=k)

    if kind in {AttributeKind.CATEGORICAL, AttributeKind.BOOLEAN, AttributeKind.ORDINAL_BIN}:
        cleaned = _clean_labels(train_subset[attribute_column]).reindex(train_subset.index)
        path_to_label = dict(
            zip(train_subset["image_path"].astype(str).values, cleaned.values, strict=False)
        )
        neighbour_paths = index.image_paths[top_idx]
        preds: list = []
        for row in neighbour_paths:
            votes = pd.Series(
                [path_to_label.get(str(p)) for p in row]
            ).dropna()
            preds.append(votes.value_counts().index[0] if not votes.empty else np.nan)
        return pd.Series(preds, index=test_index, dtype=object)

    # numeric / count
    path_to_value = dict(
        zip(
            train_subset["image_path"].astype(str).values,
            pd.to_numeric(train_subset[attribute_column], errors="coerce").values,
        )
    )
    neighbour_paths = index.image_paths[top_idx]
    preds = []
    for row in neighbour_paths:
        values = pd.Series([path_to_value.get(str(p)) for p in row]).dropna()
        if values.empty:
            preds.append(np.nan)
        else:
            preds.append(float(values.mean()))
    return pd.Series(preds, index=test_index)


def predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: Schema,
    *,
    feature_cache: FeatureCache,
    k: int = DEFAULT_K,
) -> pd.DataFrame:
    """Pipeline predict function compatible with :func:`src.models.run_pipeline`."""

    def per_attr(at, ac, train_subset, test_subset):
        return _knn_one_attribute(
            at,
            ac,
            train_subset,
            test_subset,
            schema=schema,
            cache=feature_cache,
            k=k,
        )

    return predict_per_asset_type(train_df, test_df, schema, per_attr)


def retrieve_similar_assets(
    query_image_paths: list[str],
    *,
    train_df: pd.DataFrame,
    feature_cache: FeatureCache,
    k: int = 5,
    same_asset_type: bool = True,
) -> pd.DataFrame:
    """Return a top-k similar-image table for partner-facing review.

    Output columns: ``query_image_path``, ``rank``, ``neighbor_asset_id``,
    ``neighbor_image_path``, ``cosine_similarity``, ``neighbor_profile_name``.
    """
    out_rows: list[dict] = []
    train_lookup = train_df.set_index("image_path", drop=False)
    query_feats, missing = feature_cache.aligned_to(query_image_paths)

    if same_asset_type:
        if not hasattr(retrieve_similar_assets, "_schema_cache"):
            retrieve_similar_assets._schema_cache = load_schema()  # type: ignore[attr-defined]

    for i, qpath in enumerate(query_image_paths):
        if missing[i]:
            continue
        if same_asset_type and qpath in train_lookup.index:
            profile_name = train_lookup.loc[qpath, "profile_name"]
            subset = train_df[train_df["profile_name"] == profile_name]
        else:
            subset = train_df
        index = _build_index(subset, feature_cache)
        if index.feature_matrix.shape[0] == 0:
            continue
        top_idx, top_sims = index.query(query_feats[i : i + 1], k=k)
        for rank, (idx, sim) in enumerate(
            zip(top_idx[0], top_sims[0], strict=False), start=1
        ):
            neighbour_path = index.image_paths[idx]
            row = train_lookup.loc[neighbour_path]
            out_rows.append(
                {
                    "query_image_path": qpath,
                    "rank": rank,
                    "neighbor_asset_id": int(row["asset_id"]),
                    "neighbor_image_path": neighbour_path,
                    "neighbor_profile_name": row.get("profile_name"),
                    "cosine_similarity": float(sim),
                }
            )
    return pd.DataFrame(out_rows)
