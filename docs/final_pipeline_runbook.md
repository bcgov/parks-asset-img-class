# Final Pipeline Runbook

This guide explains how to run the final BC Parks image-attribute prediction pipeline and what each step does.

The pipeline is controlled by the project `Makefile`. It checks the project inputs, confirms the cleaned image set is ready, builds or reuses DINOv3 image features, trains or reuses saved final classifiers, and writes partner-facing prediction CSV files with confidence scores.

## Quick Start

Run these commands from the repository root.

```bash
conda activate bcparks_capstone
make help
make pii
make all
```

`make all` is the main final pipeline command. It is currently an alias for the final DINOv3 partner-deliverable pipeline.
Run `make pii` once before `make all` on a fresh checkout. It screens the
images, creates the cleaned image set, and writes the completion marker that
`make all` checks before model inference.

The final outputs are written to:

```text
results/final/bcparks_asset_attribute_predictions_long.csv
results/final/bcparks_asset_attribute_predictions_wide.csv
```

The saved final classifier artifact is written to:

```text
models/final/dinov3_vitb16_logistic_regression/final_classifiers.joblib
models/final/dinov3_vitb16_logistic_regression/manifest.json
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
