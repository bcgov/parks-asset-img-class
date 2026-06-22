# Raw Data To Final Pipeline Guide

This guide explains how a new user can start from the files BC Parks provides,
prepare the local data folders, and run the final prediction pipeline.

Use this document when you are asking: "I have raw CityWide images or exports.
Where do I put them, which command do I run, and what output should I expect?"

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
| Final prediction CSVs | `results/final/` | Usually no | Partner-facing outputs. |

## Before Running Data Commands

Run these once after cloning the repository.

```bash
conda env create -f environment.yml
conda activate bcparks_capstone
make data-check
```

Place the DINOv3 weights here:

```text
models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

The model weights are not committed to Git. See
`docs/bcparks_software_installation_guide.md` for installation details.

## Path 1: Reproduce The Final Project Outputs

Use this path when you want to run the final pipeline from the processed data
already included in the repository.

The repo already includes:

```text
data/processed/master_dataset.csv
data/processed/train/
data/processed/attribute_applicability.csv
```

The raw images are not committed. They should either already exist locally under
`data/raw/citywide/images/`, or be downloaded with the CityWide API path below.

Build the cleaned image set:

```bash
make pii
```

This runs local PII screening, blurs flagged images, and writes clean copies to:

```text
data/processed/images_clean/
data/processed/images_clean/.upload_set_complete
```

Run the final DINOv3 pipeline:

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

## Quick Troubleshooting

If `make all` says the cleaned image marker is missing:

```bash
make pii
make all
```

If `make predict-new-images` cannot infer the asset type for a flat folder, add:

```bash
NEW_IMAGE_ASSET_TYPE="Trail Bridge"
```

If DINOv3 weights are missing, place this file exactly here:

```text
models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

If CityWide commands fail with missing credentials, check `.env` and rerun:

```bash
make citywide-check
```
