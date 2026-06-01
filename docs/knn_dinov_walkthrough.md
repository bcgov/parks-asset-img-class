# DINO Embeddings + k-NN Walkthrough

This pipeline predicts all **12 BC Parks attributes** from image similarity.
There is **no training step** — we extract frozen DINO/DINOv2 embeddings once,
then classify each test image by looking at its nearest neighbours in the
training set.

Peter's DINOv3 work on `main` (`docs/dinov3_walkthrough.md`) uses a different
path: per-attribute CSVs, asset-averaged embeddings, and a trained logistic /
linear-SVM classifier. Both are valid; this doc covers **retrieval k-NN only**.

---

## How it works

```text
Images  →  build_features.py  →  parquet cache (one row per image)
                                        ↓
train/test split  →  run_vectordb_knn.py  →  top-k cosine neighbours vote
                                        ↓
                              predictions CSV + MLflow metrics
```

For each test image:

1. Load its L2-normalized embedding from the parquet cache.
2. Find the **k** most similar training images (cosine similarity = dot product).
3. Restrict neighbours to the **same asset type** (stairs vs bridge vs …).
4. **Classification / boolean:** majority vote among neighbour labels.
5. **Numeric / count:** mean of neighbour values.

---

## Quick start (3 commands)

**Step 1 — build the feature cache** (run once per backbone; needs GPU):

```bash
python scripts/build_features.py --model facebook/dinov2-large
```

Output: `data/features/dinov2_large*.parquet`

**Step 2 — run k-NN prediction:**

```bash
python scripts/run_vectordb_knn.py \
    --model facebook/dinov2-large \
    --k 10 \
    --no-mlflow
```

Drop `--no-mlflow` to log to the local `./mlruns` store (or DagsHub if configured).

**Step 3 — inspect output:**

- Predictions: `data/predictions/vectordb_knn__knn_k10__dinov2_large.csv`
- Per-attribute metrics printed to stdout

---

## Key files

| File | Role |
|---|---|
| [`scripts/build_features.py`](../scripts/build_features.py) | Extract frozen DINO/DINOv2 embeddings → parquet |
| [`src/embed/dinov3.py`](../src/embed/dinov3.py) | Feature cache loader / extractor |
| [`src/models/vectordb_knn.py`](../src/models/vectordb_knn.py) | Cosine k-NN neighbour voting |
| [`scripts/run_vectordb_knn.py`](../scripts/run_vectordb_knn.py) | CLI runner + MLflow logging |
| [`configs/schema.yaml`](../configs/schema.yaml) | 12 attributes, kinds, allowed values |
| [`src/data/splits.py`](../src/data/splits.py) | Shared asset-level train/test split |

---

## Default run in this PR

| Setting | Value |
|---|---|
| Backbone | `facebook/dinov2-large` |
| k | 10 |
| Backend | numpy (in-process matmul; no Vector DB install) |
| Split | Asset-level 80/20 via `src/data/splits.py` (seed 42) |

---

## Optional ablations

Same pipeline, different hyperparameters:

```bash
# Fewer / more neighbours
python scripts/run_vectordb_knn.py --model facebook/dinov2-large --k 5
python scripts/run_vectordb_knn.py --model facebook/dinov2-large --k 20

# Smaller / larger backbone
python scripts/build_features.py --model facebook/dinov2-base
python scripts/run_vectordb_knn.py --model facebook/dinov2-base --k 10
```

These are **variants of one pipeline**, not separate implementations.

---

## vs Peter's DINOv3 classifier on `main`

| | Peter (`run_dinov3_classifier.py`) | This pipeline (`run_vectordb_knn.py`) |
|---|---|---|
| Split | Per-attribute train CSV + grouped CV | Global asset-level train/test split |
| Embedding level | Asset-averaged (mean over images) | Per-image |
| Classifier | Logistic regression / linear SVM (trained) | k-NN retrieval (no training) |
| Numeric attrs | Binned into classification bins | Direct RMSE / MAE on raw values |
| Model weights | Local `.pth` checkpoint | HuggingFace model id |
| Scope | One attribute at a time | All 12 attributes in one run |

When comparing numbers across the two approaches, note the **split difference**.

---

## Tests

```bash
pytest tests/test_vectordb_knn.py tests/test_dinov3_classifier.py tests/test_dinov3_features.py
```
