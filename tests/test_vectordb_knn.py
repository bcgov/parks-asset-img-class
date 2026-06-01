"""Unit tests for the vectordb k-NN pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.embed.dinov3 import FeatureCache  # noqa: E402
from src.models import run_pipeline  # noqa: E402
from src.models.vectordb_knn import (  # noqa: E402
    KNNIndex,
    predict,
    retrieve_similar_assets,
)


def _make_cache(paths_to_vec: dict[str, list[float]]) -> FeatureCache:
    rows = []
    for path, vec in paths_to_vec.items():
        row = {"image_path": path}
        row.update({f"f_{i}": float(v) for i, v in enumerate(vec)})
        rows.append(row)
    return FeatureCache(df=pd.DataFrame(rows), model_id="fake")


def test_knn_index_returns_closest_first() -> None:
    feats = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32
    )
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    index = KNNIndex(feature_matrix=feats, image_paths=np.array(["a", "b", "c"]))
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    top_idx, top_sims = index.query(q, k=2)
    assert list(top_idx[0]) == [0, 2]
    assert top_sims[0][0] >= top_sims[0][1]


def test_vectordb_knn_predict_round_trip(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "image_path": [f"train_{i}.jpg" for i in range(12)],
            "asset_id": list(range(12)),
            "profile_name": ["Trail Bridge"] * 12,
            "attr_decking_material": ["Timber", "Steel"] * 6,
            "attr_length": [1.0, 10.0] * 6,
        }
    )
    test = pd.DataFrame(
        {
            "image_path": [f"test_{i}.jpg" for i in range(4)],
            "asset_id": [100 + i for i in range(4)],
            "profile_name": ["Trail Bridge"] * 4,
            "attr_decking_material": ["Timber", "Steel"] * 2,
            "attr_length": [1.0, 10.0] * 2,
        }
    )

    # Each Timber row gets [1,0]; each Steel row gets [0,1].
    paths_to_vec: dict[str, list[float]] = {}
    for _, row in pd.concat([train, test], ignore_index=True).iterrows():
        paths_to_vec[row["image_path"]] = (
            [1.0, 0.0] if row["attr_decking_material"] == "Timber" else [0.0, 1.0]
        )
    cache = _make_cache(paths_to_vec)

    def predict_fn(_train_df, _test_df, schema):
        return predict(_train_df, _test_df, schema, feature_cache=cache, k=3)

    result = run_pipeline(
        pipeline="vectordb_knn",
        model_family="dinov3",
        model_name="knn_k3_fake",
        train_df=train,
        test_df=test,
        predict_fn=predict_fn,
        log_to_mlflow=False,
        predictions_dir=tmp_path / "preds",
    )

    assert (result.predictions["attr_decking_material"] == test["attr_decking_material"]).all()
    assert (
        result.predictions["attr_length"].astype(float) == test["attr_length"].astype(float)
    ).all()


def test_retrieve_similar_assets_returns_top_k_table() -> None:
    train = pd.DataFrame(
        {
            "image_path": [f"train_{i}.jpg" for i in range(4)],
            "asset_id": list(range(4)),
            "profile_name": ["Trail Bridge"] * 2 + ["Stairs"] * 2,
        }
    )
    paths_to_vec = {
        "train_0.jpg": [1.0, 0.0],
        "train_1.jpg": [0.9, 0.1],
        "train_2.jpg": [0.0, 1.0],
        "train_3.jpg": [0.1, 0.9],
        "query.jpg": [1.0, 0.0],
    }
    cache = _make_cache(paths_to_vec)
    # 'query.jpg' has no entry in train -> ``same_asset_type=False``
    out = retrieve_similar_assets(
        ["query.jpg"], train_df=train, feature_cache=cache, k=2, same_asset_type=False
    )
    assert len(out) == 2
    assert list(out["rank"]) == [1, 2]
    assert out.iloc[0]["neighbor_image_path"] == "train_0.jpg"
