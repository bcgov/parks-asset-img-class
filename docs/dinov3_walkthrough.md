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
  --image-root data/processed/images_clean
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
  --image-root data/processed/images_clean
```

For a quick smoke test:

```bash
python scripts/extract_dinov3_features.py \
  --input data/processed/train/attr_decking_material_train.csv \
  --output data/features/dinov3_vits16_attr_decking_material_images_smoke.csv \
  --asset-output data/features/dinov3_vits16_attr_decking_material_assets_smoke.csv \
  --model dinov3_vits16 \
  --weights models/downloaded_model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --image-root data/processed/images_clean \
  --limit-assets 10
```

## Step 2: Run the Classifier

```bash
python scripts/run_dinov3_classifier.py \
  --labels data/processed/train/attr_decking_material_train.csv \
  --features data/features/dinov3_vitb16_attr_decking_material_assets.csv \
  --target attr_decking_material
```

This step logs to DagsHub/MLflow by default. The run stores the summary metrics,
fold-level CSV artifact, out-of-fold prediction CSV artifact, and standard tags:

```text
task = attr_decking_material
model_family = dinov3
model_name = dinov3_vitb16_logistic_regression
data_version = processed-train
split_seed = 42
```

If you only want local CSV files, add:

```bash
--no-mlflow
```

To try Linear SVM on the same DINOv3 features:

```bash
python scripts/run_dinov3_classifier.py \
  --labels data/processed/train/attr_decking_material_train.csv \
  --features data/features/dinov3_vitb16_attr_decking_material_assets.csv \
  --target attr_decking_material \
  --classifier linear_svm
```

To tune logistic regression on the same DINOv3 features:

```bash
python scripts/run_dinov3_classifier.py \
  --labels data/processed/train/attr_decking_material_train.csv \
  --features data/features/dinov3_vitb16_attr_decking_material_assets.csv \
  --target attr_decking_material \
  --classifier logistic_regression_tuned
```

The tuned logistic regression searches these values inside each training fold:

```text
C = 0.01, 0.1, 1.0, 10.0, 100.0
class_weight = balanced, none
```

Expected outputs:

```text
results/dinov3_results/dinov3_logistic/dinov3_attr_decking_material_classification_results.csv
results/dinov3_results/dinov3_logistic/dinov3_attr_decking_material_classification_cv_folds.csv
data/predictions/dinov3_predictions/dinov3_logistic/dinov3_attr_decking_material_classification_predictions.csv
```

The prediction CSV contains one out-of-fold prediction per validation asset:
`asset_id`, `fold`, `y_true`, `y_pred`, `is_correct`, plus metadata such as the
target, feature file, classifier, and split settings. Because these are
out-of-fold predictions, each row is predicted by a model that did not train on
that asset.

The classifier uses grouped cross-validation by `asset_id`, matching the
baseline leakage-prevention strategy.

Classifier outputs are organized automatically:

```text
results/dinov3_results/dinov3_logistic/
results/dinov3_results/dinov3_logistic_tuned/
results/dinov3_results/dinov3_linear_svm/
results/dinov3_results/dinov3_random_forest/
results/dinov3_results/dinov3_gradient_boost/

data/predictions/dinov3_predictions/dinov3_logistic/
data/predictions/dinov3_predictions/dinov3_logistic_tuned/
data/predictions/dinov3_predictions/dinov3_linear_svm/
data/predictions/dinov3_predictions/dinov3_random_forest/
data/predictions/dinov3_predictions/dinov3_gradient_boost/
```

## Reusing for Other Attributes

Once `attr_decking_material` works, use the batch runner for the remaining
11 attributes:

```bash
python scripts/run_dinov3_remaining_attributes.py
```

The batch runner extracts one shared feature file from the union of the
remaining attributes, then runs the classifier for each target in this order:

```text
attr_abutment_material
attr_bridge_type
attr_has_edge_guard
attr_has_pedestrian_railing
attr_material_frame_tank_body
attr_structure_material
attr_structure_position
fall_height_bin
length_bin
steps_bin
width_bin
```

It logs each classifier run to DagsHub/MLflow by default. To skip logging:

```bash
python scripts/run_dinov3_remaining_attributes.py --no-mlflow
```

To run the remaining attributes with Linear SVM:

```bash
python scripts/run_dinov3_remaining_attributes.py --classifier linear_svm
```

To run the remaining attributes with tuned logistic regression:

```bash
python scripts/run_dinov3_remaining_attributes.py --classifier logistic_regression_tuned
```

To run the remaining attributes with Random Forest:

```bash
python scripts/run_dinov3_remaining_attributes.py --classifier random_forest
```

To run the remaining attributes with histogram-based gradient boosting:

```bash
python scripts/run_dinov3_remaining_attributes.py --classifier hist_gradient_boosting
```

To rerun feature extraction even if the shared feature file exists:

```bash
python scripts/run_dinov3_remaining_attributes.py --force-extract
```

To run all 12 baseline targets, including `attr_decking_material`:

```bash
python scripts/run_dinov3_remaining_attributes.py --include-decking
```

You can also repeat the original two-step pattern manually for one target:

```bash
python scripts/extract_dinov3_features.py \
  --input data/processed/train/attr_bridge_type_train.csv \
  --output data/features/dinov3_vitb16_attr_bridge_type_images.csv \
  --asset-output data/features/dinov3_vitb16_attr_bridge_type_assets.csv \
  --model dinov3_vitb16 \
  --weights models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --image-root data/processed/images_clean

python scripts/run_dinov3_classifier.py \
  --labels data/processed/train/attr_bridge_type_train.csv \
  --features data/features/dinov3_vitb16_attr_bridge_type_assets.csv \
  --target attr_bridge_type
```

To log a result that was already generated before MLflow logging was added,
rerun only the classifier step. You do not need to rerun DINOv3 feature
extraction if the asset feature CSV already exists.

Recommended next attributes:

```text
attr_bridge_type
attr_has_pedestrian_railing
attr_structure_material
steps_bin
```

## What Each File Does

```text
scripts/compare_dinov3_to_baseline.py
```

Command-line script for joining baseline and DINOv3 result summaries by
attribute and writing `results/dinov3_vs_baseline_comparison.csv`.

Run it after DINOv3 classifiers finish:

```bash
python scripts/compare_dinov3_to_baseline.py
```

For Linear SVM results:

```bash
python scripts/compare_dinov3_to_baseline.py \
  --dinov3-glob 'results/dinov3_results/dinov3_linear_svm/dinov3_*_linear_svm_classification_results.csv' \
  --output results/dinov3_linear_svm_vs_baseline_comparison.csv
```

For Random Forest results:

```bash
python scripts/compare_dinov3_to_baseline.py \
  --dinov3-glob 'results/dinov3_results/dinov3_random_forest/dinov3_*_random_forest_classification_results.csv' \
  --output results/dinov3_random_forest_vs_baseline_comparison.csv
```

For histogram-based gradient boosting results:

```bash
python scripts/compare_dinov3_to_baseline.py \
  --dinov3-glob 'results/dinov3_results/dinov3_gradient_boost/dinov3_*_hist_gradient_boosting_classification_results.csv' \
  --output results/dinov3_hist_gradient_boosting_vs_baseline_comparison.csv
```

```text
src/dinov3_features.py
```

Reusable functions for resolving image paths, loading DINOv3, extracting
image-level embeddings, and averaging them to asset-level features.

```text
src/dinov3_classifier.py
```

Reusable functions for joining labels to embeddings and evaluating the selected
classifier with grouped cross-validation. It returns summary metrics,
fold-level metrics, and out-of-fold predictions.

```text
scripts/extract_dinov3_features.py
```

Command-line script for the slow step: image embedding extraction.

```text
scripts/run_dinov3_classifier.py
```

Command-line script for one DINOv3 classification target. It writes summary and
fold CSVs under the classifier-specific `results/dinov3_results/...` folder and
prediction CSVs under the matching `data/predictions/dinov3_predictions/...`
folder unless custom directories are supplied.

Command-line script for the fast step: train/evaluate a classifier from saved
asset-level embeddings.
