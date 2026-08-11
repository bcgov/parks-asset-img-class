# Raw Data To Pipeline Guide

This guide explains how to add the image data required before running the final
pipeline. The repository tracks processed CSV files, but raw images are not
stored in Git.

Start here if you see an error during `make pii` about missing image data.

## 1. Clone The Repository

```bash
git clone https://github.com/sgauth01/parks-asset-img-class.git
cd parks-asset-img-class
```

## 2. Create The Environment

```bash
conda env create -f environment.yml
conda activate bcparks_capstone
```

The environment includes `make`, which is required because the pipeline is
Makefile driven.

## 3. Add The DINOv3 Model Weights

To download the DINOv3 model locally, access must be requested by filling out [this form](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).

The full DINOv3 guide is available at [https://github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) with all available DINOv3 models listed in the `Pretrained models` section.

Once the form has been filled out, you will receive an email from Meta with the files to download. Download the `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` model.

Once downloaded, copy it to the following directory in the repository root:

```text
models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

## 4. Add Image Data

Choose one of the two paths below. This step must be completed before running
`make pii`.

### Option A: BC Parks SharePoint Zip, No API Required

BC Parks users can download the prepared image export from the internal
SharePoint folder:

[BC Parks SharePoint Round2_ImageDownload folder](https://bcgov.sharepoint.com/sites/ENV-amtraining/temp_ai_image/Forms/AllItems.aspx?id=%2Fsites%2FENV%2Damtraining%2Ftemp%5Fai%5Fimage%2FRound2%5FImageDownload&viewid=b969c8d6%2De7f5%2D465f%2Db3c3%2De32131305fd4)

Download:

```text
temp_ai_image/Round2_ImageDownload/bc_park-api-data.zip
```

Unzip the file locally. If the zip contains a `data/` folder, copy that folder
into the repository root so the paths merge with this project. After copying,
the repository should contain:

```text
data/raw/citywide/images/
```

The PII screen recursively searches `data/raw/citywide/images/`, so nested
folders are okay. Do not commit these raw images to Git.

### Option B: CityWide API Download

If you have CityWide API credentials, create `.env` in the repository root:

```text
CITYWIDE_API_KEY=
CITYWIDE_DB=
CITYWIDE_USER=
CITYWIDE_API_URL=https://v4.citywidesolutions.com/v4_server/external/v1
```

Then run:

```bash
make citywide-check
make download-citywide-images
```

The API downloader writes images to the same folder expected by `make pii`:

```text
data/raw/citywide/images/
```

The API path can take a long time for the full image set. For most BC Parks
handoff testing, the SharePoint zip is faster.

For more detail, see [`citywide_api_runbook.md`](citywide_api_runbook.md).

## 5. Run Privacy Screening

After raw images are in place, run:

```bash
make pii
```

If you previously ran a small sample or partial PII screen, force a full rebuild
of the PII outputs instead:

```bash
make -B pii
```

This screens raw images for PII, blurs flagged images, and writes the cleaned
image set to:

```text
data/processed/images_clean/
```

It also creates the marker required by the final pipeline:

```text
data/processed/images_clean/.upload_set_complete
```

### Optional: Run PII On A Small Sample

For a quick PII test, write sample outputs to separate file names so you do not
overwrite the default files used by `make pii`:

```bash
python scripts/screen_images_for_pii.py \
  --image-dir data/raw/citywide/images \
  --output-csv results/predictions/pii_screen_sample.csv \
  --max-images 20
```

Then blur and assemble a small cleaned image folder from that sample:

```bash
python scripts/blur_flagged_images.py \
  --input-csv results/predictions/pii_screen_sample.csv \
  --output-log data/pii_review/blur_log_sample.csv
```

```bash
python scripts/build_upload_set.py \
  --screen-csv results/predictions/pii_screen_sample.csv \
  --output-root data/processed/images_clean_sample
```

Do not write small sample runs to
`results/predictions/pii_screen.csv`. That default file is used by `make pii`
for the full pipeline.

## 6. Run The Final Pipeline

```bash
make all
```

Final partner-facing outputs are written to:

```text
results/final/bcparks_asset_attribute_predictions_long.csv
results/final/bcparks_asset_attribute_predictions_wide.csv
```

For the full final pipeline walkthrough, see
[`final_pipeline_runbook.md`](final_pipeline_runbook.md).

## What Not To Commit

Keep these local only:

- raw image folders under `data/raw/`
- cleaned image folders under `data/processed/images_clean/`
- downloaded DINOv3 model weights under `models/downloaded_model/`
- generated feature files under `data/features/`
- saved classifier artifacts under `models/final/`
