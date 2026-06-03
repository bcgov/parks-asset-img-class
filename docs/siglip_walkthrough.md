# SigLIP Walkthrough

SigLIP is used as a frozen full-image feature extractor. It replaces the
DINOv3 embedding step, but keeps the same labels, asset-level aggregation,
grouped cross-validation, classifiers, metrics, and DagsHub/MLflow logging.

## Default Model

The default model is:

```text
google/siglip2-base-patch16-224
```

This is the current SigLIP 2 base-size checkpoint used by the project. Keep it
as the default unless a newer Google SigLIP 2 checkpoint is selected for a
specific experiment. Every SigLIP script accepts `--model-name`, so the model
can be updated without changing code.

The output slug for this default model is:

```text
google_siglip2_base_patch16_224
```

## Vision Encoder Only

The project only needs image embeddings, so `src/siglip_features.py` loads the
standalone Hugging Face vision model (`SiglipVisionModel` or
`Siglip2VisionModel`) instead of the full image-text `SiglipModel`.

This avoids keeping the text encoder in memory. Depending on how the Hugging
Face repository stores its checkpoint files, the first download may still need
the large model weight file. If the download stalls, stop the run and download
the model separately with `huggingface-cli download`, then pass the local model
folder with `--model-name`.

## Workflow

1. Read one or more train CSVs from `data/processed/train/`.
2. Resolve each `image_path` under `data/raw`.
3. Run each image through the SigLIP image encoder.
4. Save one feature row per image.
5. Average image features into one row per `asset_id`.
6. Join asset features to each attribute's labels.
7. Train a classifier with grouped cross-validation by `asset_id`.
8. Compare SigLIP results against the majority baseline and DINOv3.
9. Log classifier results to DagsHub/MLflow unless `--no-mlflow` is used.

## Smoke Test

Start with one attribute and a small asset limit:

```bash
python scripts/run_siglip_attributes.py \
  --targets attr_decking_material \
  --smoke-assets-per-class 2 \
  --force-extract \
  --no-mlflow
```

Prefer `--smoke-assets-per-class` over `--limit-assets` when you also want to
run the classifier. It avoids accidentally selecting only one class, such as
the first 10 `Timber` assets for `attr_decking_material`.

You can also run extraction and classification manually:

```bash
python scripts/extract_siglip_features.py \
  --input data/processed/train/attr_decking_material_train.csv \
  --output data/features/google_siglip2_base_patch16_224_attr_decking_material_images_smoke.csv \
  --asset-output data/features/google_siglip2_base_patch16_224_attr_decking_material_assets_smoke.csv \
  --image-root data/raw \
  --limit-assets 10

python scripts/run_siglip_classifier.py \
  --labels data/processed/train/attr_decking_material_train.csv \
  --features data/features/google_siglip2_base_patch16_224_attr_decking_material_assets_smoke.csv \
  --target attr_decking_material \
  --no-mlflow
```

## One Full Attribute

```bash
python scripts/extract_siglip_features.py \
  --input data/processed/train/attr_decking_material_train.csv \
  --output data/features/google_siglip2_base_patch16_224_attr_decking_material_images.csv \
  --asset-output data/features/google_siglip2_base_patch16_224_attr_decking_material_assets.csv \
  --image-root data/raw

python scripts/run_siglip_classifier.py \
  --labels data/processed/train/attr_decking_material_train.csv \
  --features data/features/google_siglip2_base_patch16_224_attr_decking_material_assets.csv \
  --target attr_decking_material
```

Expected outputs:

```text
data/features/google_siglip2_base_patch16_224_attr_decking_material_images.csv
data/features/google_siglip2_base_patch16_224_attr_decking_material_assets.csv
results/siglip_attr_decking_material_classification_results.csv
results/siglip_attr_decking_material_classification_cv_folds.csv
```

## All Attributes

Run all 12 attributes in baseline order:

```bash
python scripts/run_siglip_attributes.py --include-decking --force-extract
```

This creates one shared SigLIP feature table from the union of all selected
train CSVs, then trains one classifier per attribute.

Expected shared feature outputs for all 12 attributes:

```text
data/features/google_siglip2_base_patch16_224_all_attributes_union_input.csv
data/features/google_siglip2_base_patch16_224_all_attributes_images.csv
data/features/google_siglip2_base_patch16_224_all_attributes_assets.csv
```

To skip DagsHub/MLflow logging:

```bash
python scripts/run_siglip_attributes.py --include-decking --no-mlflow
```

To force feature extraction again:

```bash
python scripts/run_siglip_attributes.py --include-decking --force-extract
```

To run a different classifier:

```bash
python scripts/run_siglip_attributes.py --include-decking --classifier linear_svm
python scripts/run_siglip_attributes.py --include-decking --classifier random_forest
python scripts/run_siglip_attributes.py --include-decking --classifier hist_gradient_boosting
```

## Compare With Baseline

After SigLIP classifiers finish:

```bash
python scripts/compare_siglip_to_baseline.py
```

Expected output:

```text
results/siglip_vs_baseline_comparison.csv
```

For other classifiers:

```bash
python scripts/compare_siglip_to_baseline.py \
  --siglip-glob 'results/siglip_*_linear_svm_classification_results.csv' \
  --output results/siglip_linear_svm_vs_baseline_comparison.csv
```

## How To Explain The Experiment

The comparison is:

```text
majority baseline
vs
DINOv3 frozen full-image embeddings
vs
SigLIP 2 frozen full-image embeddings
```

SigLIP keeps the full scene context, unlike the removed SAM2 crop experiment.
That makes it a cleaner comparison for attributes that may depend on the whole
image, such as railings, bridge type, size bins, and structure position.

## Files Added

```text
src/siglip_features.py
```

Reusable functions for loading SigLIP, resolving images, extracting image-level
embeddings, and averaging them into asset-level features.

```text
src/siglip_classifier.py
```

Reusable functions for evaluating frozen SigLIP embeddings with grouped
cross-validation.

```text
scripts/extract_siglip_features.py
```

Command-line script for SigLIP embedding extraction.

```text
scripts/run_siglip_classifier.py
```

Command-line script for one target classifier.

```text
scripts/run_siglip_attributes.py
```

Batch runner for many or all attributes.

```text
scripts/compare_siglip_to_baseline.py
```

Comparison script for baseline vs SigLIP summary metrics.
