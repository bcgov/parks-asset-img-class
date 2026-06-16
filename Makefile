.DEFAULT_GOAL := help

PYTHON ?= conda run -n bcparks_capstone python
PYTEST ?= conda run -n bcparks_capstone pytest
QUARTO ?= quarto

SEED ?= 42
FOLDS ?= 5
CLASSIFIER ?= logistic_regression
DATA_VERSION ?= processed-train

TRAIN_DIR ?= data/processed/train
MASTER_DATA ?= data/processed/master_dataset.csv
IMAGE_ROOT ?= data/processed/images_clean
FEATURE_DIR ?= data/features
FINAL_DIR ?= results/final

DINO_MODEL ?= facebook_dinov3_vitl16_pretrain_lvd1689m
DINO_HUB_MODEL ?= dinov3_vitb16
DINO_WEIGHTS ?= models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
DINO_IMAGE_FEATURES ?= $(FEATURE_DIR)/$(DINO_MODEL).parquet
DINO_MASTER_FEATURES ?= $(FEATURE_DIR)/$(DINO_MODEL)_master_assets.csv
DINO_RUN_FEATURES ?= $(FEATURE_DIR)/$(DINO_MODEL)_all_attributes_assets.csv
PII_SCREEN_CSV ?= results/predictions/pii_screen.csv
PII_BLUR_LOG ?= data/pii_review/blur_log.csv
PII_UPLOAD_MARKER ?= $(IMAGE_ROOT)/.upload_set_complete

VLM_PROVIDER ?= gemini
VLM_MODEL ?= gemini-3-flash-preview
VLM_PROMPT ?= pedestrian_railing_v1
VLM_INPUT ?= $(TRAIN_DIR)/attr_has_pedestrian_railing_train.csv
VLM_OUTPUT_DIR ?= results/vlm_predictions
VLM_OUTPUT ?= $(VLM_OUTPUT_DIR)/$(VLM_PROVIDER)_$(VLM_PROMPT)_$(VLM_MODEL).csv
VLM_IMAGE_ROOT ?= $(IMAGE_ROOT)
VLM_LIMIT ?=
VLM_OFFSET ?= 0
VLM_DELAY ?= 0
VLM_BUFFER_SIZE ?= 20
VLM_MAX_TOKENS ?= 4096
VLM_LIMIT_ARG = $(if $(VLM_LIMIT),--limit $(VLM_LIMIT),)

CITYWIDE_OUTPUT_DIR ?= data/raw/citywide
CITYWIDE_PROFILE ?=
CITYWIDE_LIMIT ?=
CITYWIDE_WORKERS ?= 4
CITYWIDE_MAX_CALLS_PER_HOUR ?= 900
CITYWIDE_PROFILE_ARG = $(foreach profile,$(CITYWIDE_PROFILE),--profile $(profile))
CITYWIDE_LIMIT_ARG = $(if $(CITYWIDE_LIMIT),--limit $(CITYWIDE_LIMIT),)

.PHONY: help all final-dinov3 smoke env-check data-check model-data-check test \
	pii pii-screen pii-blur pii-upload-set baseline \
	features-dinov3-master features-dinov3-extract-master train-dinov3 train-siglip train-openclip models \
	vlm-check vlm-predict vlm-smoke final-with-vlm \
	citywide-check download-citywide-probe download-citywide-metadata download-citywide-images download-citywide-sample \
	compare-dinov3 compare-siglip compare figures export-bcparks \
	report report-html report-pdf

help:
	@echo "Final pipeline targets:"
	@echo "  make smoke                 Fast local validation: env/data checks + tests + baseline"
	@echo "  make final-dinov3          Reproducible final DINOv3 pipeline + BC Parks CSV + report"
	@echo "  make all                   Full final pipeline target alias"
	@echo "  make model-data-check      Check cleaned images and DINOv3 feature file"
	@echo "  make pii                   Screen, blur, and assemble cleaned image set"
	@echo "  make baseline              Run grouped baseline strategies"
	@echo "  make features-dinov3-master Build asset features from precomputed DINOv3 parquet"
	@echo "  make train-dinov3          Run grouped CV for DINOv3 final classifier"
	@echo "  make vlm-predict           Optional cloud VLM predictions; requires provider credentials"
	@echo "  make vlm-smoke             Optional 5-asset VLM credential/path smoke run"
	@echo "  make final-with-vlm        Run final DINOv3 pipeline plus optional VLM branch"
	@echo "    VLM variables: VLM_PROVIDER, VLM_MODEL, VLM_PROMPT, VLM_INPUT, VLM_OUTPUT, VLM_LIMIT"
	@echo "    VLM providers: gemini, openai, grok, claude, github"
	@echo "    VLM credentials: GEMINI_API_KEY/GOOGLE_API_KEY, OPENAI_API_KEY, XAI_API_KEY, ANTHROPIC_API_KEY, GITHUB_TOKEN"
	@echo "  make download-citywide-images  Optional CityWide raw image/metadata download"
	@echo "  make download-citywide-metadata Optional CityWide metadata-only download"
	@echo "  make download-citywide-sample  Optional 50-file CityWide download smoke run"
	@echo "    CityWide variables: CITYWIDE_PROFILE, CITYWIDE_LIMIT, CITYWIDE_WORKERS, CITYWIDE_OUTPUT_DIR"
	@echo "    CityWide credentials: CITYWIDE_API_KEY, CITYWIDE_DB, CITYWIDE_USER"
	@echo "  make export-bcparks        Write partner-facing prediction CSVs"
	@echo "  make report                Render Quarto HTML/PDF report"

all: final-dinov3

final-dinov3: test data-check pii model-data-check baseline features-dinov3-master train-dinov3 compare-dinov3 figures export-bcparks report

final-with-vlm: final-dinov3 vlm-predict

smoke: env-check data-check test baseline

env-check:
	$(PYTHON) -c "import pandas, sklearn, torch, PIL; print('Environment imports OK')"

citywide-check:
	$(PYTHON) scripts/check_pipeline_inputs.py --require-citywide-credentials

download-citywide-probe: citywide-check
	$(PYTHON) scripts/download_citywide_images.py \
	  --output-dir $(CITYWIDE_OUTPUT_DIR) \
	  --max-calls-per-hour $(CITYWIDE_MAX_CALLS_PER_HOUR) \
	  --probe

download-citywide-metadata: citywide-check
	$(PYTHON) scripts/download_citywide_images.py \
	  --output-dir $(CITYWIDE_OUTPUT_DIR) \
	  --workers $(CITYWIDE_WORKERS) \
	  --max-calls-per-hour $(CITYWIDE_MAX_CALLS_PER_HOUR) \
	  --metadata-only \
	  $(CITYWIDE_PROFILE_ARG)

download-citywide-images: citywide-check
	$(PYTHON) scripts/download_citywide_images.py \
	  --output-dir $(CITYWIDE_OUTPUT_DIR) \
	  --workers $(CITYWIDE_WORKERS) \
	  --max-calls-per-hour $(CITYWIDE_MAX_CALLS_PER_HOUR) \
	  $(CITYWIDE_PROFILE_ARG) \
	  $(CITYWIDE_LIMIT_ARG)

download-citywide-sample: CITYWIDE_LIMIT = 50
download-citywide-sample: download-citywide-images

data-check:
	$(PYTHON) scripts/check_pipeline_inputs.py

model-data-check:
	$(PYTHON) scripts/check_pipeline_inputs.py \
	  --require-images \
	  --feature-file $(DINO_IMAGE_FEATURES)

test:
	$(PYTEST) -q tests

$(PII_SCREEN_CSV): scripts/screen_images_for_pii.py
	$(PYTHON) scripts/screen_images_for_pii.py

pii-screen: $(PII_SCREEN_CSV)

$(PII_BLUR_LOG): scripts/blur_flagged_images.py $(PII_SCREEN_CSV)
	$(PYTHON) scripts/blur_flagged_images.py

pii-blur: $(PII_BLUR_LOG)

$(PII_UPLOAD_MARKER): scripts/build_upload_set.py $(PII_SCREEN_CSV) $(PII_BLUR_LOG)
	$(PYTHON) scripts/build_upload_set.py
	touch $(PII_UPLOAD_MARKER)

pii-upload-set: $(PII_UPLOAD_MARKER)

pii: pii-screen pii-blur pii-upload-set

baseline:
	$(PYTHON) scripts/run_baseline.py \
	  --train-dir $(TRAIN_DIR) \
	  --output-dir results/baseline_results \
	  --folds $(FOLDS) \
	  --seed $(SEED) \
	  --data-version $(DATA_VERSION) \
	  --no-mlflow

$(DINO_MASTER_FEATURES): scripts/build_asset_features_from_image_features.py $(MASTER_DATA) $(DINO_IMAGE_FEATURES)
	$(PYTHON) scripts/build_asset_features_from_image_features.py \
	  --master $(MASTER_DATA) \
	  --image-features $(DINO_IMAGE_FEATURES) \
	  --asset-output $(DINO_MASTER_FEATURES)

$(DINO_RUN_FEATURES): $(DINO_MASTER_FEATURES)
	cp $(DINO_MASTER_FEATURES) $(DINO_RUN_FEATURES)

features-dinov3-master: $(DINO_MASTER_FEATURES) $(DINO_RUN_FEATURES)

features-dinov3-extract-master:
	$(PYTHON) scripts/extract_dinov3_features.py \
	  --input $(MASTER_DATA) \
	  --output $(FEATURE_DIR)/$(DINO_HUB_MODEL)_master_images.csv \
	  --asset-output $(DINO_MASTER_FEATURES) \
	  --model $(DINO_HUB_MODEL) \
	  --weights $(DINO_WEIGHTS) \
	  --image-root $(IMAGE_ROOT)

train-dinov3:
	$(PYTHON) scripts/run_dinov3_remaining_attributes.py \
	  --train-dir $(TRAIN_DIR) \
	  --feature-dir $(FEATURE_DIR) \
	  --model $(DINO_MODEL) \
	  --weights $(DINO_WEIGHTS) \
	  --image-root $(IMAGE_ROOT) \
	  --folds $(FOLDS) \
	  --seed $(SEED) \
	  --classifier $(CLASSIFIER) \
	  --data-version $(DATA_VERSION) \
	  --no-mlflow

train-siglip:
	$(PYTHON) scripts/run_siglip_attributes.py \
	  --train-dir $(TRAIN_DIR) \
	  --feature-dir $(FEATURE_DIR) \
	  --image-root $(IMAGE_ROOT) \
	  --folds $(FOLDS) \
	  --seed $(SEED) \
	  --classifier $(CLASSIFIER) \
	  --data-version $(DATA_VERSION) \
	  --include-decking \
	  --no-mlflow

train-openclip:
	$(PYTHON) scripts/run_openclip_attributes.py \
	  --train-dir $(TRAIN_DIR) \
	  --feature-dir $(FEATURE_DIR) \
	  --image-root $(IMAGE_ROOT) \
	  --folds $(FOLDS) \
	  --seed $(SEED) \
	  --classifier $(CLASSIFIER) \
	  --data-version $(DATA_VERSION) \
	  --no-mlflow

models: train-dinov3 train-siglip train-openclip

vlm-check:
	$(PYTHON) scripts/check_pipeline_inputs.py \
	  --require-images \
	  --require-vlm-credentials \
	  --vlm-provider $(VLM_PROVIDER) \
	  --vlm-model $(VLM_MODEL)

vlm-predict: pii vlm-check
	$(PYTHON) scripts/run_vlm_predictor.py \
	  --input $(VLM_INPUT) \
	  --output $(VLM_OUTPUT) \
	  --provider $(VLM_PROVIDER) \
	  --model $(VLM_MODEL) \
	  --prompt $(VLM_PROMPT) \
	  --image_root $(VLM_IMAGE_ROOT) \
	  --offset $(VLM_OFFSET) \
	  --delay $(VLM_DELAY) \
	  --buffer_size $(VLM_BUFFER_SIZE) \
	  --max_tokens $(VLM_MAX_TOKENS) \
	  $(VLM_LIMIT_ARG)

vlm-smoke: VLM_LIMIT = 5
vlm-smoke: vlm-predict

compare-dinov3:
	$(PYTHON) scripts/compare_dinov3_to_baseline.py \
	  --classifier $(CLASSIFIER)

compare-siglip:
	$(PYTHON) scripts/compare_siglip_to_baseline.py \
	  --classifier $(CLASSIFIER)

compare: compare-dinov3 compare-siglip

figures:
	$(PYTHON) scripts/create_model_comparison_figures.py

export-bcparks:
	$(PYTHON) scripts/export_bcparks_predictions.py \
	  --master $(MASTER_DATA) \
	  --features $(DINO_MASTER_FEATURES) \
	  --train-dir $(TRAIN_DIR) \
	  --classifier $(CLASSIFIER) \
	  --model-family dinov3 \
	  --model-name $(DINO_MODEL) \
	  --seed $(SEED) \
	  --output-long $(FINAL_DIR)/bcparks_asset_attribute_predictions_long.csv \
	  --output-wide $(FINAL_DIR)/bcparks_asset_attribute_predictions_wide.csv

report:
	$(QUARTO) render reports/Image_analysis_of_park_infrastructure_report.qmd

report-html:
	$(QUARTO) render reports/Image_analysis_of_park_infrastructure_report.qmd --to html

report-pdf:
	$(QUARTO) render reports/Image_analysis_of_park_infrastructure_report.qmd --to pdf
