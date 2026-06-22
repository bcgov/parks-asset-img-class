[![Apache 2.0 License](https://img.shields.io/github/license/bcgov/nr-epd-aq-statements.svg)](/LICENSE) [![Lifecycle:Experimental](https://img.shields.io/badge/Lifecycle-Experimental-339999)](<Redirect-URL>)

# parks-asset-img-class

## Project overview

This repository contains a 2026 UBC MDS capstone project on using image analysis to classify BC Parks infrastructure assets and predict attributes such as asset type, material, railing presence, size ranges, structure position, and number of steps.

The main runnable artifact is the Makefile pipeline. It validates inputs, screens raw images for PII, builds cleaned image sets, runs baseline and DINOv3 embedding classifiers, and exports partner-facing prediction CSVs. The Quarto report is rendered separately.

## Repository structure

```text
.
├── Makefile
├── environment.yml
├── data/
│   ├── processed/
│   │   ├── master_dataset.csv
│   │   └── train/
│   └── raw/                  # optional CityWide download output
├── docs/
│   ├── raw_data_to_pipeline.md
│   ├── final_pipeline_runbook.md
│   ├── citywide_api_runbook.md
│   ├── dinov3_walkthrough.md
│   ├── siglip_walkthrough.md
│   └── vlm_walkthrough.md
├── scripts/                  # pipeline entry points used by the Makefile
├── src/                      # reusable package code
├── tests/
├── models/                   # where to store DINOv3 downloaded model
└── reports/
    ├── bcparks_capstone_final_report.qmd
    ├── report_images/
    └── references.bib
```

## Setup

For a printable BC Parks handoff guide, start with
[`docs/bcparks_software_installation_guide.md`](docs/bcparks_software_installation_guide.md)
or the matching PDF in the same folder.

**1. Create and activate the Conda environment**

```bash
conda env create -f environment.yml
conda activate bcparks_capstone
```
**2. Set up the VLM API keys**

Create an `.env` file in the project root with the required API keys for the models you want to use.

```bash
# Google AI Studio
GEMINI_API_KEY="your-key-here"

# GitHub Models (OpenAI, Llama, Phi)
GITHUB_TOKEN="your-token-here"
```

An example of how to set up API keys in an `.env` file is provided in `.env.example`.

- For **Google AI Studio** VLMs, get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).
- For **GitHub** VLMs, create a personal access token in [GitHub Settings](https://github.com/settings/tokens) with `read:packages` scope.

**3. Set up the DINOv3 model**

This project uses the `dinov3_vitb16` model. To download the model locally, access must be requested by filling out [this form](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/).

The full DINOv3 guide is available at [https://github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) with all available DINOv3 models listed in the `Pretrained models` section.

Once the form has been filled out, you will receive an email from Meta with the files to download. Download the `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` model.

Once downloaded, copy it to the following directory in the repository root:

```text
models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

## Makefile pipeline

From the repository root, list available targets:

```bash
make help
```

Documentation map:

- [`docs/bcparks_software_installation_guide.md`](docs/bcparks_software_installation_guide.md): first-time installation and handoff guide
- [`docs/raw_data_to_pipeline.md`](docs/raw_data_to_pipeline.md): start from raw BC Parks files and feed them through cleaning, preprocessing, and prediction
- [`docs/final_pipeline_runbook.md`](docs/final_pipeline_runbook.md): day-to-day final pipeline commands
- [`docs/citywide_api_runbook.md`](docs/citywide_api_runbook.md): optional CityWide API download details
- [`docs/vlm_walkthrough.md`](docs/vlm_walkthrough.md): optional cloud VLM workflow
- [`docs/dinov3_walkthrough.md`](docs/dinov3_walkthrough.md), [`docs/siglip_walkthrough.md`](docs/siglip_walkthrough.md): technical model experiment notes

For a full walkthrough of the final pipeline, see
[`docs/final_pipeline_runbook.md`](docs/final_pipeline_runbook.md).
If you are starting from raw CityWide files or a new BC Parks bulk export, start
with [`docs/raw_data_to_pipeline.md`](docs/raw_data_to_pipeline.md) first.

Run the final DINOv3 pipeline and export partner-facing prediction CSVs:

```bash
make pii
make all
```

Run `make pii` once before the final pipeline on a fresh checkout. It creates
the cleaned image set and the completion marker required by `make final-dinov3`
and `make all`.

Run a faster local validation pass:

```bash
make smoke
```

Run a small new-image demo from the cleaned training image folder:

```bash
make demo-new-images
```

Predict on a separate folder of already-cleaned new images:

```bash
make predict-new-images NEW_IMAGE_FOLDER=data/processed/images_clean/citywide/images NEW_IMAGE_LIMIT=5
```


### Predicting on a new CityWide bulk export

When BC Parks provides a bulk CityWide export — a flat folder of images plus a CSV mapping image file names to asset IDs — the pipeline sorts, screens, and predicts in three steps. Exports are processed one asset type at a time.

```bash
# 1. Sort the flat export into per-asset folders (data/raw/new_batch/<asset_id>/)
make sort-citywide-export \
  CITYWIDE_EXPORT_FOLDER=<path to the flat image folder> \
  CITYWIDE_EXPORT_CSV=<path to the export CSV>

# 2. Screen and blur for PII (writes cleaned images to data/processed/images_clean/new_batch/)
make pii-batch

# 3. Predict attributes (DINOv3 default)
make predict-new-images NEW_IMAGE_ASSET_TYPE="Stairs"

# or, optionally, with the cloud VLM
make vlm-predict NEW_IMAGE_ASSET_TYPE="Stairs" VLM_PROMPT=stairs_v1
```

Replace `"Stairs"` with the export's asset type — one of: `Stairs`, `Trail Bridge`, `Boardwalk < 1.2m High`, `Boardwalk > 1.2m High`, `Viewing Platform`.

To test on the existing training images instead of a new export, use `make demo`, or override the folder as shown above. Training data uses the `profile_id/asset_id` layout, so no `NEW_IMAGE_ASSET_TYPE` is needed — the type is inferred from the profile_id folder.

Optional cloud VLM predictions are available when provider credentials are set:

```bash
make vlm-smoke VLM_PROVIDER=gemini VLM_MODEL=gemini-3-flash-preview
make final-with-vlm VLM_PROVIDER=gemini VLM_MODEL=gemini-3-flash-preview
```

The partner-facing prediction exports are written to:

- `results/final/bcparks_asset_attribute_predictions_long.csv`
- `results/final/bcparks_asset_attribute_predictions_wide.csv`

## Optional CityWide download

The raw CityWide downloader is available as an upstream Makefile branch for BC Parks or graders who have API credentials. Required `.env` keys are:

```bash
CITYWIDE_API_KEY=
CITYWIDE_DB=
CITYWIDE_USER=
CITYWIDE_API_URL=https://v4.citywidesolutions.com/v4_server/external/v1
```

Download metadata only:

```bash
make download-citywide-metadata
```

Download metadata and images:

```bash
make download-citywide-images
```

Useful controls:

```bash
make download-citywide-sample
make download-citywide-images CITYWIDE_PROFILE="337 356" CITYWIDE_LIMIT=100
```

The downloader writes `assets.csv`, `attributes.csv`, `files_manifest.csv`, `images_manifest.csv`, and downloaded images under `data/raw/citywide/`.

For the full CityWide API flow, including how linked attributes are downloaded,
see [`docs/citywide_api_runbook.md`](docs/citywide_api_runbook.md).

## To use Vision Language Models (VLMs)

This project uses Vision Language Models (VLMs) to directly predict BC Parks asset attributes from images.
VLMs take asset images and return structured predictions (attribute values & confidence scores) in JSON format.

Supported provider families include:

- **Google Gemini** through `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- **OpenAI** through `OPENAI_API_KEY`
- **xAI Grok** through `XAI_API_KEY`
- **Anthropic Claude** through `ANTHROPIC_API_KEY`
- **GitHub Models** through `GITHUB_TOKEN`

### Quick Start

1. Set up API keys in `.env` file:

```bash
GEMINI_API_KEY="your-key-here"
GITHUB_TOKEN="your-token-here"
```

2. Run a Makefile smoke prediction:

```bash
make vlm-smoke VLM_PROVIDER=gemini VLM_MODEL=gemini-3-flash-preview
```

Or run batch predictions directly:

```bash
python scripts/run_vlm_predictor.py \
  --input data/processed/train/train_only_stairs.csv \
  --output results/vlm_stairs_gemini.csv \
  --provider gemini \
  --model gemini-3-flash-preview \
  --prompt stairs_v1
```

`--input` also accepts a folder of images, not just a CSV. For a flat folder of one asset type, pass `--asset-type`:

```bash
python scripts/run_vlm_predictor.py \
  --input data/processed/images_clean/new_batch \
  --asset-type "Stairs" \
  --output results/vlm_new_batch_gemini.csv \
  --provider gemini \
  --model gemini-3-flash-preview \
  --prompt stairs_v1
```

3. Evaluate predictions against ground truth:

```bash
python scripts/evaluate_predictions.py \
  --predictions results/vlm_stairs_gemini.csv \
  --ground_truth_dir data/processed/train \
  --attributes attr_number_of_steps \
  --model gemini-3-flash-preview
```

For comprehensive documentation on supported models, prompts, workflows, and extending the system with new models/prompts, see [`docs/vlm_walkthrough.md`](docs/vlm_walkthrough.md).

## Render the report

The report is built with [Quarto](https://quarto.org/). Install Quarto if it is not already available:

```bash
quarto --version
```

PDF rendering also requires a LaTeX installation. If PDF rendering fails because LaTeX is missing, install TinyTeX with:

```bash
quarto install tinytex
```

To render the report, from the repository root, run:

```bash
quarto render reports/bcparks_capstone_final_report.qmd
```

This command renders all formats listed in the report YAML, currently HTML and PDF.

To render only one format:

```bash
quarto render reports/bcparks_capstone_final_report.qmd --to html
quarto render reports/bcparks_capstone_final_report.qmd --to pdf
```

The older `reports/Image_analysis_of_park_infrastructure_report.qmd` is a
legacy draft that depends on raw CityWide files under `data/raw/citywide/`,
including `image_attributes_manifest.csv`. Use
`reports/bcparks_capstone_final_report.qmd` for the current final report.

## Experiment tracking with MLflow

All model runs are tracked with [MLflow](https://mlflow.org/). The
default tracking store is a local SQLite database at `./mlflow.db`
(gitignored), so no server is required and nothing leaves the machine.

End-to-end smoke test (synthetic data only, no SharePoint download needed):

```bash
python scripts/mlflow_smoke_test.py
```

This fits a `MajorityClassPredictor` on a synthetic 3-class target and a
`MedianRegressor` on a synthetic numeric target, and writes both runs
to `./mlflow.db` under the experiment **`parks-asset-img-class`**.

View the runs in your browser:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Programmatic logging:

```python
from src.baseline import MajorityClassPredictor
from src.mlflow_utils import (
    setup_mlflow, log_classification_run,
    make_run_name, make_standard_tags,
)

setup_mlflow()  # uses ./mlflow.db

clf = MajorityClassPredictor().fit(X_train, y_train)
y_pred = clf.predict(X_test)

log_classification_run(
    run_name=make_run_name("T2_decking_material", "majority_class"),
    tags=make_standard_tags(
        task="T2_decking_material",
        model_family="baseline",
        model_name="majority_class",
        data_version="2026-05-05",
        split_seed=42,
    ),
    params={"n_train": len(y_train), "majority_class": str(clf.fitted_value_)},
    y_true=y_test,
    y_pred=y_pred,
)
```

Standard tags every run carries: `task`, `model_family`, `model_name`,
`data_version`, `split_seed`. Standard metrics: `accuracy`, `macro_f1`,
`weighted_f1`, `per_class_f1.json`, and `confusion_matrix.json` for
classification; `mae`, `rmse`, `r2` for regression.

Run the unit tests:

```bash
pytest -q tests/test_mlflow_utils.py
```

## Current status

The final pipeline is Makefile-driven and centered on the DINOv3 embedding classifier, with optional cloud VLM and CityWide download branches for credentialed runs.

## Getting Help or Reporting an Issue

To report bugs/issues/feature requests, please file an [issue](https://github.com/bcgov/parks-asset-img-class/issues/new).

## How to Contribute

If you would like to contribute, please see our [CONTRIBUTING](CONTRIBUTING.md) guidelines.

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## License

    Copyright 2026 Province of British Columbia

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and limitations under the License.


------------------------------------------------------------------------

*This project was created using the [bcgovr](https://github.com/bcgov/bcgovr) package.*
