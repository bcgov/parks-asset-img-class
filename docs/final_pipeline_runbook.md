# Final Pipeline Runbook

This guide explains how to run the final BC Parks image-attribute prediction pipeline and what each step does.

The pipeline is controlled by the project `Makefile`. It checks the project inputs, confirms the cleaned image set is ready, builds or reuses DINOv3 image features, trains or reuses saved final classifiers, and writes partner-facing prediction CSV files with confidence scores.

If you are starting from raw CityWide data or a new BC Parks bulk export, read
`docs/raw_data_to_pipeline.md` first. That guide explains where to place raw
files and which cleaning/preprocessing commands to run before using this
runbook.

## Which Starting Point Applies?

There are three common starting points.

| Starting point | Use when | Main command path |
| --- | --- | --- |
| Existing processed project data | You cloned the repo and want to reproduce the final project outputs. | `make pii`, then `make all` |
| CityWide API download | You have CityWide API credentials and want to refresh metadata or images directly from CityWide. | `make download-citywide-images`, then `make pii`, then `make all` |
| New CityWide bulk export | BC Parks gives you a flat folder of exported images plus a CSV mapping images to asset IDs. | `make sort-citywide-export`, then `make pii-batch`, then `make predict-new-images` |

The final model does not need the raw source files to be committed to Git.
Large raw images, model weights, generated features, and saved classifiers are
local files.

## Quick Start

Most users run **Path 1** (reproduce the final outputs from the processed data
already in the repo). From the repository root:

```bash
conda activate bcparks_capstone
make pii      
make all      
```

Outputs land in `results/final/`. For details and the other starting points
(CityWide API download, or a new bulk export), see the numbered paths below.

## Required Inputs

The final pipeline expects these files/directories:

```text
environment.yml
data/processed/master_dataset.csv
data/processed/train/
data/processed/attribute_applicability.csv
data/processed/images_clean/
data/processed/images_clean/.upload_set_complete
models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

The DINOv3 model weights are not committed to git. They must be placed locally at:

```text
models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

## What `make all` Does

When you run:

```bash
make all
```

the Makefile runs this target chain:

```text
make all
  -> final-dinov3
       -> test
       -> data-check
       -> pii-ready
       -> model-data-check
       -> features-dinov3-master
       -> train-final-models
       -> export-bcparks
```

In plain language:

1. `test`
   Runs the unit test suite to catch code regressions before producing outputs.

2. `data-check`
   Confirms required project inputs exist, including:
   - `environment.yml`
   - `data/processed/master_dataset.csv`
   - `data/processed/train/`
   - `data/processed/attribute_applicability.csv`

3. `pii-ready`
   Confirms the cleaned image set is already prepared and marked complete.

4. `model-data-check`
   Confirms the cleaned image directory exists.

5. `features-dinov3-master`
   Builds or reuses asset-level DINOv3 embeddings. If the feature files already exist, Make skips the expensive feature extraction step.

6. `train-final-models`
   Trains the final lightweight classifiers on the DINOv3 asset embeddings and saves them as a reusable local model bundle. This is fast because DINOv3 is frozen and the embeddings already exist.

7. `export-bcparks`
   Loads the saved final classifier bundle and exports the BC Parks prediction CSV files.

Each step prints a readable duration line, for example:

```text
Finished in: 4 sec
```

## Path 1: Reproduce The Final Project Outputs

Use this path when you want to run the final pipeline from the processed data
already included in the repository.

The repo already includes the processed inputs:

```text
data/processed/master_dataset.csv
data/processed/train/
data/processed/attribute_applicability.csv
```

The raw images are **not** committed. They must already exist locally under
`data/raw/citywide/images/`, or be downloaded via the CityWide API (Path 2)
before running this path.

**1. Build the cleaned image set:**

```bash
make pii
```

This screens raw images for PII, blurs flagged images, and writes the cleaned
set to `data/processed/images_clean/` along with the `.upload_set_complete`
marker that `make all` checks before inference.

**2. Run the final DINOv3 pipeline:**

```bash
make all
```

The final partner-facing outputs are:

```text
results/final/bcparks_asset_attribute_predictions_long.csv
results/final/bcparks_asset_attribute_predictions_wide.csv
```

## Path 2: Download Raw Data From The CityWide API

Use this path only if you have CityWide API credentials.

Create `.env` in the repository root:

```text
CITYWIDE_API_KEY=
CITYWIDE_DB=
CITYWIDE_USER=
CITYWIDE_API_URL=https://v4.citywidesolutions.com/v4_server/external/v1
```

Check credentials:

```bash
make citywide-check
```

Download metadata and linked attributes only:

```bash
make download-citywide-metadata
```

Download metadata, linked attributes, file metadata, and images:

```bash
make download-citywide-images
```

The downloader writes:

```text
data/raw/citywide/assets.csv
data/raw/citywide/attributes.csv
data/raw/citywide/files_manifest.csv
data/raw/citywide/images_manifest.csv
data/raw/citywide/images/<profile_id>/<asset_id>/<file_id>__<filename>
```

CityWide profile IDs map to asset types as follows:

```text
337 -> Boardwalk < 1.2m High
573 -> Boardwalk > 1.2m High
356 -> Stairs
253 -> Trail Bridge
359 -> Viewing Platform
```

After downloading images, build the cleaned image set:

```bash
make pii
```

Then run the final project pipeline:

```bash
make all
```

The CityWide download refreshes raw files. It does not automatically replace
the tracked processed training CSVs. To train on newly labelled assets, update
the processed files listed in "Retrain Models After Adding New Labelled Data"
below.

For more detail on how the API downloads linked attributes, see
`docs/citywide_api_runbook.md`.

## Path 3: Predict On A New CityWide Bulk Export

Use this path when BC Parks provides a one-off export with:

- a flat folder of image files
- a CSV export that maps image file names to asset IDs

The CSV must contain the columns:

```text
File Name
Source Page Link
```

Process one asset type at a time, because a flat export does not include enough
folder structure for the model to infer the asset type automatically.

### 1. Copy The Raw Export Locally

Use a local folder like this:

```text
data/raw/partner_exports/stairs/
  images/
    image_001.jpg
    image_002.jpg
  export.csv
```

Do not commit this folder to Git.

### 2. Sort Images Into Asset Folders

Run:

```bash
make sort-citywide-export \
  CITYWIDE_EXPORT_FOLDER=data/raw/partner_exports/stairs/images \
  CITYWIDE_EXPORT_CSV=data/raw/partner_exports/stairs/export.csv
```

This creates:

```text
data/raw/new_batch/<asset_id>/<image_file>
results/predictions/citywide_sort_report.csv
```

The sort report lists copied images, missing files, and images that were in the
folder but not referenced by the CSV.

### 3. Screen And Clean The New Batch

Run:

```bash
make pii-batch
```

This screens `data/raw/new_batch/`, blurs flagged images, and writes the cleaned
batch under:

```text
data/processed/images_clean/new_batch/
```

### 4. Predict Attributes For The New Batch

Run the prediction command with the exact asset type:

```bash
make predict-new-images NEW_IMAGE_ASSET_TYPE="Stairs"
```

Valid asset type values are:

```text
Boardwalk < 1.2m High
Boardwalk > 1.2m High
Stairs
Trail Bridge
Viewing Platform
```

The new-batch outputs are:

```text
results/final/new_image_predictions_long.csv
results/final/new_image_predictions_wide.csv
```

To limit a run for a quick check:

```bash
make predict-new-images NEW_IMAGE_ASSET_TYPE="Stairs" NEW_IMAGE_LIMIT=10
```

## Saved Final Classifiers

The final pipeline separates training from inference:

```text
train-final-models
  -> trains final sklearn classifiers on cached asset embeddings
  -> writes models/final/dinov3_vitb16_logistic_regression/final_classifiers.joblib

export-bcparks / predict-new-images
  -> loads the saved classifier bundle
  -> predicts attributes
```

This means DINOv3 is not retrained. DINOv3 is used as a frozen feature extractor. The saved model bundle contains the small classifiers trained on top of the DINOv3 embeddings.

The model manifest is human-readable:

```text
models/final/dinov3_vitb16_logistic_regression/manifest.json
```

It records the model family, DINOv3 model name, classifier type, feature columns, and the target attributes included in the bundle.

Run only the saved-model training step:

```bash
make train-final-models
```

This retrains the saved classifiers when the training CSVs, applicability matrix, or asset feature file are newer than the saved bundle.

## When Make Reuses Outputs

The pipeline uses normal Makefile timestamp logic. Each generated output is
reused when it already exists and is newer than the files it depends on. A step
reruns when the output is missing or older than one of its inputs.

### DINOv3 Embeddings

The final asset-level embedding file is:

```text
data/features/dinov3_vitb16_master_assets.csv
```

Make reuses this file when it is already present and up to date. It rebuilds it
when the file is missing or when an upstream dependency changes, such as the
master dataset, the image-level feature file, the DINOv3 extraction script, or
the configured cleaned-image inputs.

To force DINOv3 feature rebuilding:

```bash
make clean-dinov3
make features-dinov3-master
```

### Saved Sklearn Classifiers

The final saved classifier bundle is:

```text
models/final/dinov3_vitb16_logistic_regression/final_classifiers.joblib
```

Make reuses this bundle when it already exists and is newer than the training
CSV files, the asset-level DINOv3 feature file, the applicability matrix, and
the final-model training script. It retrains the classifiers when the bundle is
missing or stale.

To force final classifier retraining:

```bash
make clean-final-models
make train-final-models
```

### Final Prediction CSVs

The partner-facing CSVs are generated under:

```text
results/final/
```

These are export artifacts. They can be regenerated from the saved classifier
bundle and asset embeddings without retraining DINOv3.

## Attribute Applicability

The pipeline uses this matrix:

```text
data/processed/attribute_applicability.csv
```

This file controls which attributes should be predicted for each asset type.

For example:

- `attr_bridge_type` is predicted only for `Trail Bridge`
- `steps_bin` is predicted only for `Stairs`
- `attr_has_pedestrian_railing` is predicted for all supported asset types

The wide CSV leaves non-applicable asset/attribute combinations blank.

## Output Files

### Long Output

```text
results/final/bcparks_asset_attribute_predictions_long.csv
```

This file has one row per asset and predicted attribute.

Use this format when you want to filter, audit, or summarize predictions by attribute.

Important columns include:

- `asset_id`
- `profile_id`
- `profile_name`
- `description`
- `image_count`
- `attribute`
- `target_column`
- `predicted_value`
- `confidence_score`
- `confidence_level`
- `model_family`
- `model_name`
- `classifier`
- `applicable_profile_names`

### Wide Output

```text
results/final/bcparks_asset_attribute_predictions_wide.csv
```

This file has one row per asset.

Use this format when BC Parks wants one spreadsheet-style row per infrastructure asset.

For each attribute, it includes:

- `<attribute>_prediction`
- `<attribute>_confidence_score`
- `<attribute>_confidence_level`

## Confidence Levels

The pipeline converts model confidence scores into three readable levels:

```text
high    confidence score >= 0.80
medium  confidence score >= 0.60 and < 0.80
low     confidence score < 0.60
```

These are model confidence levels, not manual verification labels.

## Retrain Models After Adding New Labelled Data

Use this section only if new labelled training data is added, not just new
unlabelled images for prediction.

Update or add the processed files:

```text
data/processed/master_dataset.csv
data/processed/train/*_train.csv
data/processed/attribute_applicability.csv
```

If the new training data includes new images, place the raw images under
`data/raw/citywide/images/` or another raw folder that matches the paths in
`master_dataset.csv`, then rebuild the cleaned image set:

```bash
make pii
```

If only labels changed and the image/features files did not change, retrain the
saved lightweight classifier heads:

```bash
make clean-final-models
make train-final-models
```

If images or the master image table changed, rebuild DINOv3 features and saved
classifiers:

```bash
make clean-dinov3
make clean-final-models
make all
```

## Common Commands

### Show Available Commands

```bash
make help
```

### Train or Refresh Saved Final Classifiers

```bash
make train-final-models
```

This creates or refreshes:

```text
models/final/dinov3_vitb16_logistic_regression/final_classifiers.joblib
models/final/dinov3_vitb16_logistic_regression/manifest.json
```

### Run the Final Partner Deliverable Pipeline

```bash
make all
```

Equivalent command:

```bash
make final-dinov3
```

### Run a Fast Smoke Check

```bash
make smoke
```

This runs environment checks, data checks, tests, and baseline logic. It is useful before running heavier model steps.

### Run a Small New-Image Demo

```bash
make demo
```

`make demo` runs `demo-new-images` with:

```text
DEMO_ASSET_LIMIT=10
```

This is useful for showing the pipeline on a small sample.

### Predict on a Folder of New Images

```bash
make predict-new-images NEW_IMAGE_FOLDER=path/to/new/images
```

This command extracts DINOv3 embeddings for the new images and loads the saved final classifier bundle from:

```text
models/final/dinov3_vitb16_logistic_regression/final_classifiers.joblib
```

For a flat folder where all assets are the same type:

```bash
make predict-new-images \
  NEW_IMAGE_FOLDER=path/to/stair/images \
  NEW_IMAGE_ASSET_TYPE="Stairs"
```

For a small limited run:

```bash
make predict-new-images \
  NEW_IMAGE_FOLDER=path/to/new/images \
  NEW_IMAGE_LIMIT=10
```

### Optional VLM Run

The cloud VLM pipeline is optional and requires provider credentials.

Example:

```bash
make vlm-smoke VLM_PROVIDER=gemini VLM_MODEL=gemini-3-flash-preview
```

Supported provider options are shown by:

```bash
make help
```

## Files By Pipeline Stage

| Stage | Local location | Tracked in Git? | Notes |
| --- | --- | --- | --- |
| Raw CityWide API download | `data/raw/citywide/` | No | Contains raw assets, attributes, file manifests, and images. |
| Raw one-off partner export | `data/raw/partner_exports/<asset_type>/` | No | Suggested local folder for a flat image export and its mapping CSV. |
| Processed training metadata | `data/processed/master_dataset.csv` | Yes | Main asset/image table used by the final project pipeline. |
| Processed train labels | `data/processed/train/` | Yes | Label CSVs used to train the lightweight classifier heads. |
| Attribute applicability map | `data/processed/attribute_applicability.csv` | Yes | Defines which attributes apply to which asset types. |
| Cleaned image set | `data/processed/images_clean/` | No | Built by the PII pipeline from raw images. |
| DINOv3 feature files | `data/features/` | Usually no | Generated or reused by Makefile targets. |
| Saved final classifiers | `models/final/` | No | Created by `make train-final-models`; reused for inference. |
| Final prediction CSVs | `results/final/` | Usually no | Partner-facing outputs.

## Cleanup Commands

Remove generated final prediction CSVs:

```bash
make clean-final
```

Remove the saved final classifier artifact:

```bash
make clean-final-models
```

Remove generated DINOv3 feature CSVs for the current DINO model:

```bash
make clean-dinov3
```

Remove final prediction CSVs, saved final classifiers, and current DINOv3 feature CSVs:

```bash
make clean
```

The cleanup targets preserve model weights, raw data, training data, and cleaned images.

## What to Commit

Do commit:

- Makefile and script changes
- tests
- small documentation files
- small configuration or mapping files needed by the pipeline

Do not commit:

- generated prediction CSVs under `results/final/`
- saved final classifier artifacts under `models/final/`
- DINOv3 model weights under `models/downloaded_model/`
- raw downloaded image folders
- `.env`

## What Not To Commit

Do not commit:

- `.env`
- raw CityWide images or partner export folders
- DINOv3 `.pth` model weights
- generated DINOv3 features in `data/features/`
- saved classifiers in `models/final/`
- large generated prediction or report artifacts unless the team explicitly wants them in Git

The tracked processed CSVs are intentionally small enough to support
reproducible final pipeline runs.

## Troubleshooting

### Missing DINOv3 Weights

If the pipeline reports a missing model file, place the DINOv3 checkpoint here:

```text
models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

### Cleaned Images Not Ready

If the pipeline reports a missing upload marker:

```text
data/processed/images_clean/.upload_set_complete
```

run:

```bash
make pii
```

This screens images for PII, blurs flagged images, and assembles the cleaned image set.

### Final CSVs Are Missing

Run:

```bash
make all
```

Then check:

```text
results/final/
```

### Saved Final Classifier Bundle Is Missing

Run:

```bash
make train-final-models
```

Then check:

```text
models/final/dinov3_vitb16_logistic_regression/final_classifiers.joblib
models/final/dinov3_vitb16_logistic_regression/manifest.json
```

### The Pipeline Skips DINOv3 Feature Extraction

This is expected when feature CSVs already exist. Make only reruns steps whose outputs are missing or older than their inputs.

### The Pipeline Skips Final Classifier Training

This is expected when the saved model bundle already exists and is newer than the training CSVs, applicability matrix, and asset feature file. To force retraining:

```bash
make clean-final-models
make train-final-models
```

### 1. Copy The Raw Export Locally

Use a local folder like this:

```text
data/raw/partner_exports/stairs/
  images/
    image_001.jpg
    image_002.jpg
  export.csv
```

Do not commit this folder to Git.

### 2. Sort Images Into Asset Folders

Run:

```bash
make sort-citywide-export \
  CITYWIDE_EXPORT_FOLDER=data/raw/partner_exports/stairs/images \
  CITYWIDE_EXPORT_CSV=data/raw/partner_exports/stairs/export.csv
```

This creates:

```text
data/raw/new_batch/<asset_id>/<image_file>
results/predictions/citywide_sort_report.csv
```

The sort report lists copied images, missing files, and images that were in the
folder but not referenced by the CSV.

### 3. Screen And Clean The New Batch

Run:

```bash
make pii-batch
```

This screens `data/raw/new_batch/`, blurs flagged images, and writes the cleaned
batch under:

```text
data/processed/images_clean/new_batch/
```

### 4. Predict Attributes For The New Batch

Run the prediction command with the exact asset type:

```bash
make predict-new-images NEW_IMAGE_ASSET_TYPE="Stairs"
```

Valid asset type values are:

```text
Boardwalk < 1.2m High
Boardwalk > 1.2m High
Stairs
Trail Bridge
Viewing Platform
```

The new-batch outputs are:

```text
results/final/new_image_predictions_long.csv
results/final/new_image_predictions_wide.csv
```

To limit a run for a quick check:

```bash
make predict-new-images NEW_IMAGE_ASSET_TYPE="Stairs" NEW_IMAGE_LIMIT=10
```

