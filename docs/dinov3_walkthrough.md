# DINOv3 Walkthrough

This project should use DINOv3 first as a frozen image feature extractor.
DINOv3 turns each asset image into a numeric embedding. A small classifier then
learns the BC Parks attribute labels from those embeddings.

## First Attribute

Start with:

```text
attr_decking_material
```

This is a good first task because the label is often visible in the image:
timber, concrete, steel, or similar surface/material patterns.

## Workflow

1. Read a task CSV from `data/processed/train/`.
2. Resolve each row's `image_path` to the local image directory.
3. Run each image through `dinov3_vitb16`.
4. Save one embedding row per image.
5. Average image embeddings into one feature row per `asset_id`.
6. Join those asset features back to the task labels.
7. Train logistic regression with grouped cross-validation by `asset_id`.
8. Compare accuracy, macro F1, and weighted F1 against the majority baseline.

## Why Frozen DINOv3 First

Frozen embeddings are the simplest defensible first experiment:

- no GPU-heavy fine-tuning loop
- less risk of overfitting
- easier comparison against `scripts/run_baseline.py`
- embeddings can be reused across multiple classifiers and attributes

## Model Choice

Main local model:

```text
dinov3_vitb16
```

The local downloaded checkpoint is:

```text
models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

ViT-B/16 is the main experiment model. The smaller ViT-S/16 checkpoint is also
available locally and is useful for quick smoke tests:

```text
models/downloaded_model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

Always keep `--model` and `--weights` matched. Use `dinov3_vitb16` with the
ViT-B/16 checkpoint and `dinov3_vits16` with the ViT-S/16 checkpoint.

## Step 1: Extract Features

```bash
python scripts/extract_dinov3_features.py \
  --input data/processed/train/attr_decking_material_train.csv \
  --output data/features/dinov3_vitb16_attr_decking_material_images.csv \
  --asset-output data/features/dinov3_vitb16_attr_decking_material_assets.csv \
  --model dinov3_vitb16 \
  --weights models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --image-root data/raw
```

The `--weights` value can be either a local `.pth` file or the approved
checkpoint URL provided after requesting DINOv3 model-weight access from Meta.
Passing a local path is more reliable because the generated URLs may be
restricted or time-limited.

If the DINOv3 Torch Hub load fails with a missing package such as
`torchmetrics`, update the environment first:

```bash
conda env update -f environment.yml --prune
conda activate bcparks_capstone
```

The DINOv3 Torch Hub entrypoint imports some evaluation and segmentation
utilities at load time, so support packages such as `torchmetrics`, `omegaconf`,
`submitit`, and `termcolor` need to be available even for frozen feature
extraction.

Expected outputs:

```text
data/features/dinov3_vitb16_attr_decking_material_images.csv
data/features/dinov3_vitb16_attr_decking_material_assets.csv
data/features/dinov3_vitb16_attr_decking_material_images_skipped.csv
```

The image-level file has one row per image. The asset-level file has one row
per `asset_id`, created by averaging all image embeddings for that asset.

If torch.hub cannot download the official DINOv3 repo, clone it manually and
pass the local path:

```bash
python scripts/extract_dinov3_features.py \
  --input data/processed/train/attr_decking_material_train.csv \
  --output data/features/dinov3_vitb16_attr_decking_material_images.csv \
  --asset-output data/features/dinov3_vitb16_attr_decking_material_assets.csv \
  --model dinov3_vitb16 \
  --model-source /path/to/dinov3 \
  --weights models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --image-root data/raw
```

For a quick smoke test:

```bash
python scripts/extract_dinov3_features.py \
  --input data/processed/train/attr_decking_material_train.csv \
  --output data/features/dinov3_vits16_attr_decking_material_images_smoke.csv \
  --asset-output data/features/dinov3_vits16_attr_decking_material_assets_smoke.csv \
  --model dinov3_vits16 \
  --weights models/downloaded_model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --image-root data/raw \
  --limit-assets 10
```

## Step 2: Run the Classifier

```bash
python scripts/run_dinov3_classifier.py \
  --labels data/processed/train/attr_decking_material_train.csv \
  --features data/features/dinov3_vitb16_attr_decking_material_assets.csv \
  --target attr_decking_material
```

Expected outputs:

```text
results/dinov3_attr_decking_material_classification_results.csv
results/dinov3_attr_decking_material_classification_cv_folds.csv
```

The classifier uses grouped cross-validation by `asset_id`, matching the
baseline leakage-prevention strategy.

## Reusing for Other Attributes

Once `attr_decking_material` works, repeat the same pattern for another target:

```bash
python scripts/extract_dinov3_features.py \
  --input data/processed/train/attr_bridge_type_train.csv \
  --output data/features/dinov3_vitb16_attr_bridge_type_images.csv \
  --asset-output data/features/dinov3_vitb16_attr_bridge_type_assets.csv \
  --model dinov3_vitb16 \
  --weights models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --image-root data/raw

python scripts/run_dinov3_classifier.py \
  --labels data/processed/train/attr_bridge_type_train.csv \
  --features data/features/dinov3_vitb16_attr_bridge_type_assets.csv \
  --target attr_bridge_type
```

Recommended next attributes:

```text
attr_bridge_type
attr_has_pedestrian_railing
attr_structure_material
steps_bin
```

## What Each File Does

```text
src/dinov3_features.py
```

Reusable functions for resolving image paths, loading DINOv3, extracting
image-level embeddings, and averaging them to asset-level features.

```text
src/dinov3_classifier.py
```

Reusable functions for joining labels to embeddings and evaluating a logistic
regression classifier with grouped cross-validation.

```text
scripts/extract_dinov3_features.py
```

Command-line script for the slow step: image embedding extraction.

```text
scripts/run_dinov3_classifier.py
```

Command-line script for the fast step: train/evaluate a classifier from saved
asset-level embeddings.
