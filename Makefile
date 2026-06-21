.DEFAULT_GOAL := help

PYTHON ?= python
PYTEST ?= pytest
TIME ?= @sh scripts/run_with_timer.sh

SEED ?= 42
FOLDS ?= 5
CLASSIFIER ?= logistic_regression
DATA_VERSION ?= processed-train

TRAIN_DIR ?= data/processed/train
MASTER_DATA ?= data/processed/master_dataset.csv
IMAGE_ROOT ?= data/processed/images_clean
FEATURE_DIR ?= data/features
FINAL_DIR ?= results/final
ATTRIBUTE_APPLICABILITY ?= data/processed/attribute_applicability.csv

DINO_MODEL ?= dinov3_vitb16
DINO_HUB_MODEL ?= dinov3_vitb16
DINO_WEIGHTS ?= models/downloaded_model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
GENERATED_DINO_IMAGE_FEATURES ?= $(FEATURE_DIR)/$(DINO_MODEL)_master_images.csv
DINO_IMAGE_FEATURES ?= $(GENERATED_DINO_IMAGE_FEATURES)
DINO_MASTER_FEATURES ?= $(FEATURE_DIR)/$(DINO_MODEL)_master_assets.csv
DINO_RUN_FEATURES ?= $(FEATURE_DIR)/$(DINO_MODEL)_all_attributes_assets.csv
PII_SCREEN_CSV ?= results/predictions/pii_screen.csv
PII_BLUR_LOG ?= data/pii_review/blur_log.csv
PII_UPLOAD_MARKER ?= $(IMAGE_ROOT)/.upload_set_complete

VLM_PROVIDER ?= gemini
VLM_MODEL ?= gemini-3-flash-preview
VLM_PROMPT ?= stairs_v1
VLM_INPUT ?= $(NEW_BATCH_CLEAN)
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

CITYWIDE_EXPORT_FOLDER ?=
CITYWIDE_EXPORT_CSV ?=
NEW_BATCH_RAW ?= data/raw/new_batch
NEW_BATCH_CLEAN ?= $(IMAGE_ROOT)/new_batch
CITYWIDE_SORT_OUTPUT ?= $(NEW_BATCH_RAW)
CITYWIDE_SORT_REPORT ?= results/predictions/citywide_sort_report.csv

NEW_IMAGE_FOLDER ?= $(NEW_BATCH_CLEAN)
NEW_IMAGE_ASSET_TYPE ?=
NEW_IMAGE_OUTPUT ?= $(FINAL_DIR)/new_image_predictions_long.csv
NEW_IMAGE_OUTPUT_WIDE ?= $(FINAL_DIR)/new_image_predictions_wide.csv
NEW_IMAGE_LIMIT ?=
NEW_IMAGE_LIMIT_ARG = $(if $(NEW_IMAGE_LIMIT),--limit-assets $(NEW_IMAGE_LIMIT),)
DEMO_ASSET_LIMIT ?= 5

.PHONY: help all final-dinov3 smoke env-check data-check model-data-check test \
	pii pii-ready pii-screen pii-blur pii-upload-set baseline \
	features-dinov3-master features-dinov3-extract-master train-dinov3 train-siglip train-openclip models \
	vlm-check vlm-predict vlm-smoke final-with-vlm \
	citywide-check download-citywide-probe download-citywide-metadata download-citywide-images download-citywide-sample \
	compare-dinov3 compare-siglip compare figures export-bcparks predict-new-images demo demo-new-images \
	all-start final-dinov3-start evaluate-start smoke-start clean-final clean-dinov3 clean-pipeline clean \
	sort-citywide-export pii-batch pii-batch-screen

help:
	@echo "Final pipeline targets:"
	@echo "  make smoke                 Fast local validation: env/data checks + tests + baseline"
	@echo "  make final-dinov3          Reproducible final DINOv3 pipeline + BC Parks CSV"
	@echo "  make all                   Full final pipeline target alias; run 'make pii' once first"
	@echo "    DINO variables: DINO_WEIGHTS, DINO_IMAGE_FEATURES, DINO_MASTER_FEATURES"
	@echo "    Attribute map: ATTRIBUTE_APPLICABILITY"
	@echo "  make model-data-check      Check cleaned image inputs"
	@echo "  make pii                   Screen, blur, and assemble cleaned image set"
	@echo "  make pii-ready             Check that the cleaned image set already exists"
	@echo "  make baseline              Run grouped baseline strategies"
	@echo "  make features-dinov3-master Build asset features from DINOv3 image features"
	@echo "  make train-dinov3          Run grouped CV for DINOv3 final classifier"
	@echo "  make predict-new-images    Predict attributes for a folder of new images"
	@echo "  make demo                  Alias for demo-new-images with DEMO_ASSET_LIMIT=10"
	@echo "  make demo-new-images       Small new-image prediction run from cleaned training images"
	@echo "    New image variables: NEW_IMAGE_FOLDER, NEW_IMAGE_ASSET_TYPE, NEW_IMAGE_LIMIT, NEW_IMAGE_OUTPUT"
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
	@echo "  make clean-final           Remove generated partner-facing final CSVs"
	@echo "  make clean-dinov3          Remove generated DINOv3 feature CSVs for current DINO_MODEL"
	@echo "  make clean-pipeline        Remove generated final CSVs + current DINOv3 feature CSVs"
	@echo "  make clean                 Alias for clean-pipeline"
	@echo "    Timing: command steps print plain-language durations, for example 'Finished in: 4 sec'"

all: all-start final-dinov3

all-start:
	@printf "\n==> make all: running the final DINOv3 partner-deliverable pipeline\n"

# Partner deliverable: prediction + confidence CSVs only
final-dinov3: final-dinov3-start test data-check pii-ready model-data-check features-dinov3-master export-bcparks
	@printf "\n==> final-dinov3 complete: prediction CSVs are in $(FINAL_DIR)\n"

final-dinov3-start:
	@printf "\n==> final-dinov3: tests -> data checks -> PII-ready check -> DINOv3 features -> BC Parks CSV export\n"
	@printf "    DINO_MODEL=$(DINO_MODEL)\n"
	@printf "    DINO_WEIGHTS=$(DINO_WEIGHTS)\n"
	@printf "    DINO_IMAGE_FEATURES=$(DINO_IMAGE_FEATURES)\n"
	@printf "    DINO_MASTER_FEATURES=$(DINO_MASTER_FEATURES)\n"

# Analysis/evaluation (run manually when regenerating report numbers/figures)
evaluate: evaluate-start baseline train-dinov3 compare-dinov3 figures

evaluate-start:
	@printf "\n==> evaluate: baseline -> DINOv3 CV training -> baseline comparison -> figures\n"

final-with-vlm: final-dinov3 vlm-predict

smoke: smoke-start env-check data-check test baseline

smoke-start:
	@printf "\n==> smoke: environment imports -> data checks -> tests -> baseline\n"

env-check:
	@printf "\n==> env-check: importing core Python packages\n"
	$(TIME) $(PYTHON) -c "import pandas, sklearn, torch, PIL; print('Environment imports OK')"

citywide-check:
	@printf "\n==> citywide-check: checking CityWide credentials\n"
	$(TIME) $(PYTHON) scripts/check_pipeline_inputs.py --require-citywide-credentials

download-citywide-probe: citywide-check
	@printf "\n==> download-citywide-probe: probing CityWide API without downloading images\n"
	$(TIME) $(PYTHON) scripts/download_citywide_images.py \
	  --output-dir $(CITYWIDE_OUTPUT_DIR) \
	  --max-calls-per-hour $(CITYWIDE_MAX_CALLS_PER_HOUR) \
	  --probe

download-citywide-metadata: citywide-check
	@printf "\n==> download-citywide-metadata: downloading CityWide metadata only\n"
	$(TIME) $(PYTHON) scripts/download_citywide_images.py \
	  --output-dir $(CITYWIDE_OUTPUT_DIR) \
	  --workers $(CITYWIDE_WORKERS) \
	  --max-calls-per-hour $(CITYWIDE_MAX_CALLS_PER_HOUR) \
	  --metadata-only \
	  $(CITYWIDE_PROFILE_ARG)

download-citywide-images: citywide-check
	@printf "\n==> download-citywide-images: downloading CityWide metadata and images\n"
	$(TIME) $(PYTHON) scripts/download_citywide_images.py \
	  --output-dir $(CITYWIDE_OUTPUT_DIR) \
	  --workers $(CITYWIDE_WORKERS) \
	  --max-calls-per-hour $(CITYWIDE_MAX_CALLS_PER_HOUR) \
	  $(CITYWIDE_PROFILE_ARG) \
	  $(CITYWIDE_LIMIT_ARG)

download-citywide-sample: CITYWIDE_LIMIT = 50
download-citywide-sample: download-citywide-images

data-check:
	@printf "\n==> data-check: checking required project data inputs\n"
	$(TIME) $(PYTHON) scripts/check_pipeline_inputs.py

model-data-check:
	@printf "\n==> model-data-check: checking cleaned image directory\n"
	$(TIME) $(PYTHON) scripts/check_pipeline_inputs.py \
	  --require-images

test:
	@printf "\n==> test: running unit tests\n"
	$(TIME) $(PYTEST) -q tests

$(PII_SCREEN_CSV): scripts/screen_images_for_pii.py
	@printf "\n==> pii-screen: scanning raw images for PII and writing $(PII_SCREEN_CSV)\n"
	$(TIME) $(PYTHON) scripts/screen_images_for_pii.py

pii-screen: $(PII_SCREEN_CSV)

$(PII_BLUR_LOG): scripts/blur_flagged_images.py $(PII_SCREEN_CSV)
	@printf "\n==> pii-blur: blurring flagged images and writing $(PII_BLUR_LOG)\n"
	$(TIME) $(PYTHON) scripts/blur_flagged_images.py

pii-blur: $(PII_BLUR_LOG)

$(PII_UPLOAD_MARKER): scripts/build_upload_set.py $(PII_SCREEN_CSV) $(PII_BLUR_LOG)
	@printf "\n==> pii-upload-set: assembling cleaned image set under $(IMAGE_ROOT)\n"
	$(TIME) $(PYTHON) scripts/build_upload_set.py
	$(TIME) touch $(PII_UPLOAD_MARKER)

pii-upload-set: $(PII_UPLOAD_MARKER)

pii-ready: model-data-check
	@printf "\n==> pii-ready: checking cleaned image-set marker $(PII_UPLOAD_MARKER)\n"
	$(TIME) test -f $(PII_UPLOAD_MARKER) || (echo "Missing $(PII_UPLOAD_MARKER). Run 'make pii' before final-dinov3."; exit 1)

pii: pii-screen pii-blur pii-upload-set

pii-batch-screen:
	@printf "\n==> pii-batch-screen: screening $(NEW_BATCH_RAW) for PII\n"
	$(TIME) test -d $(NEW_BATCH_RAW) || (echo "Missing $(NEW_BATCH_RAW). Run 'make sort-citywide-export' first."; exit 1)
	$(TIME) $(PYTHON) scripts/screen_images_for_pii.py --image-dir $(NEW_BATCH_RAW)

pii-batch: pii-batch-screen pii-blur pii-upload-set
	@printf "\n==> pii-batch complete: cleaned batch under $(NEW_BATCH_CLEAN)\n"

baseline:
	@printf "\n==> baseline: running grouped majority-class baselines\n"
	$(TIME) $(PYTHON) scripts/run_baseline.py \
	  --train-dir $(TRAIN_DIR) \
	  --output-dir results/baseline_results \
	  --folds $(FOLDS) \
	  --seed $(SEED) \
	  --data-version $(DATA_VERSION) \
	  --no-mlflow

$(GENERATED_DINO_IMAGE_FEATURES): scripts/extract_dinov3_features.py $(MASTER_DATA)
	@printf "\n==> features-dinov3-extract: extracting image embeddings to $(GENERATED_DINO_IMAGE_FEATURES)\n"
	@printf "    This is the slow DINOv3 step when features are not already present.\n"
	$(TIME) $(PYTHON) scripts/check_pipeline_inputs.py \
	  --require-dinov3-weights \
	  --dinov3-weights $(DINO_WEIGHTS)
	$(TIME) $(PYTHON) scripts/extract_dinov3_features.py \
	  --input $(MASTER_DATA) \
	  --output $(GENERATED_DINO_IMAGE_FEATURES) \
	  --asset-output $(DINO_MASTER_FEATURES) \
	  --model $(DINO_HUB_MODEL) \
	  --weights $(DINO_WEIGHTS) \
	  --image-root $(IMAGE_ROOT)

$(DINO_MASTER_FEATURES): scripts/build_asset_features_from_image_features.py $(MASTER_DATA) $(DINO_IMAGE_FEATURES)
	@printf "\n==> features-dinov3-master: aggregating image embeddings to asset embeddings\n"
	@printf "    Input:  $(DINO_IMAGE_FEATURES)\n"
	@printf "    Output: $(DINO_MASTER_FEATURES)\n"
	$(TIME) $(PYTHON) scripts/build_asset_features_from_image_features.py \
	  --master $(MASTER_DATA) \
	  --image-features $(DINO_IMAGE_FEATURES) \
	  --asset-output $(DINO_MASTER_FEATURES)

$(DINO_RUN_FEATURES): $(DINO_MASTER_FEATURES)
	@printf "\n==> features-dinov3-master: copying asset features for all-attribute runs\n"
	$(TIME) cp $(DINO_MASTER_FEATURES) $(DINO_RUN_FEATURES)

features-dinov3-master: $(DINO_MASTER_FEATURES) $(DINO_RUN_FEATURES)
	@printf "\n==> features-dinov3-master complete\n"

features-dinov3-extract-master: $(DINO_IMAGE_FEATURES) $(DINO_MASTER_FEATURES)
	@printf "\n==> features-dinov3-extract-master complete\n"

train-dinov3:
	@printf "\n==> train-dinov3: running grouped CV classifiers on DINOv3 embeddings\n"
	$(TIME) $(PYTHON) scripts/run_dinov3_remaining_attributes.py \
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
	@printf "\n==> train-siglip: running grouped CV classifiers on SigLIP embeddings\n"
	$(TIME) $(PYTHON) scripts/run_siglip_attributes.py \
	  --train-dir $(TRAIN_DIR) \
	  --feature-dir $(FEATURE_DIR) \
	  --image-root $(IMAGE_ROOT) \
	  --folds $(FOLDS) \
	  --seed $(SEED) \
	  --classifier $(CLASSIFIER) \
	  --data-version $(DATA_VERSION) \
	  --no-mlflow

train-openclip:
	@printf "\n==> train-openclip: running grouped CV classifiers on OpenCLIP embeddings\n"
	$(TIME) $(PYTHON) scripts/run_openclip_attributes.py \
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
	@printf "\n==> vlm-check: checking VLM credentials for $(VLM_PROVIDER)/$(VLM_MODEL)\n"
	$(TIME) $(PYTHON) scripts/check_pipeline_inputs.py \
	  --require-images \
	  --require-vlm-credentials \
	  --vlm-provider $(VLM_PROVIDER) \
	  --vlm-model $(VLM_MODEL)

vlm-predict: pii-ready vlm-check
	@printf "\n==> vlm-predict: running optional cloud VLM predictions\n"
	$(TIME) $(PYTHON) scripts/run_vlm_predictor.py \
	  --input $(VLM_INPUT) \
	  --output $(VLM_OUTPUT) \
	  --provider $(VLM_PROVIDER) \
	  --model $(VLM_MODEL) \
	  --prompt $(VLM_PROMPT) \
	  --image_root $(VLM_IMAGE_ROOT) \
	  $(if $(NEW_IMAGE_ASSET_TYPE), --asset-type "$(NEW_IMAGE_ASSET_TYPE)") \
	  --offset $(VLM_OFFSET) \
	  --delay $(VLM_DELAY) \
	  --buffer_size $(VLM_BUFFER_SIZE) \
	  --max_tokens $(VLM_MAX_TOKENS) \
	  $(VLM_LIMIT_ARG)

vlm-smoke: VLM_LIMIT = 5
vlm-smoke: vlm-predict

compare-dinov3:
	@printf "\n==> compare-dinov3: comparing DINOv3 results to baseline\n"
	$(TIME) $(PYTHON) scripts/compare_dinov3_to_baseline.py \
	  --classifier $(CLASSIFIER)

compare-siglip:
	@printf "\n==> compare-siglip: comparing SigLIP results to baseline\n"
	$(TIME) $(PYTHON) scripts/compare_siglip_to_baseline.py \
	  --classifier $(CLASSIFIER)

compare: compare-dinov3 compare-siglip

figures:
	@printf "\n==> figures: creating model-comparison figures\n"
	$(TIME) $(PYTHON) scripts/create_model_comparison_figures.py

export-bcparks:
	@printf "\n==> export-bcparks: training final classifiers and exporting partner CSVs\n"
	@printf "    Long CSV: $(FINAL_DIR)/bcparks_asset_attribute_predictions_long.csv\n"
	@printf "    Wide CSV: $(FINAL_DIR)/bcparks_asset_attribute_predictions_wide.csv\n"
	@printf "    Attribute map: $(ATTRIBUTE_APPLICABILITY)\n"
	$(TIME) $(PYTHON) scripts/export_bcparks_predictions.py \
	  --master $(MASTER_DATA) \
	  --features $(DINO_MASTER_FEATURES) \
	  --train-dir $(TRAIN_DIR) \
	  --applicability $(ATTRIBUTE_APPLICABILITY) \
	  --classifier $(CLASSIFIER) \
	  --model-family dinov3 \
	  --model-name $(DINO_MODEL) \
	  --seed $(SEED) \
	  --output-long $(FINAL_DIR)/bcparks_asset_attribute_predictions_long.csv \
	  --output-wide $(FINAL_DIR)/bcparks_asset_attribute_predictions_wide.csv
	  
sort-citywide-export:
	@printf "\n==> sort-citywide-export: sorting flat export into per-asset folders\n"
	$(TIME) $(PYTHON) scripts/sort_citywide_export.py \
	  --input-folder $(CITYWIDE_EXPORT_FOLDER) \
	  --mapping-csv $(CITYWIDE_EXPORT_CSV) \
	  --output-dir $(CITYWIDE_SORT_OUTPUT) \
	  --report $(CITYWIDE_SORT_REPORT)

predict-new-images: data-check model-data-check features-dinov3-master
	@printf "\n==> predict-new-images: predicting attributes for $(NEW_IMAGE_FOLDER)\n"
	$(TIME) $(PYTHON) scripts/check_pipeline_inputs.py \
	  --require-dinov3-weights \
	  --dinov3-weights $(DINO_WEIGHTS) \
	  --feature-file $(DINO_MASTER_FEATURES)
	$(TIME) $(PYTHON) scripts/predict_new_images.py \
	  --image-folder $(NEW_IMAGE_FOLDER)$(if $(NEW_IMAGE_ASSET_TYPE), --asset-type "$(NEW_IMAGE_ASSET_TYPE)") \
	  --training-features $(DINO_MASTER_FEATURES) \
	  --train-dir $(TRAIN_DIR) \
	  --weights $(DINO_WEIGHTS) \
	  --model $(DINO_HUB_MODEL) \
	  --image-root . \
	  --classifier $(CLASSIFIER) \
	  --seed $(SEED) \
	  --output $(NEW_IMAGE_OUTPUT) \
	  --output-wide $(NEW_IMAGE_OUTPUT_WIDE) \
	  $(NEW_IMAGE_LIMIT_ARG)

demo-new-images: NEW_IMAGE_FOLDER = $(IMAGE_ROOT)/citywide/images
demo-new-images: NEW_IMAGE_LIMIT = $(DEMO_ASSET_LIMIT)
demo-new-images: NEW_IMAGE_OUTPUT = $(FINAL_DIR)/demo_new_image_predictions_long.csv
demo-new-images: NEW_IMAGE_OUTPUT_WIDE = $(FINAL_DIR)/demo_new_image_predictions_wide.csv
demo-new-images: predict-new-images

demo: DEMO_ASSET_LIMIT = 10
demo: demo-new-images

clean-final:
	@printf "\n==> clean-final: removing generated final prediction CSVs from $(FINAL_DIR)\n"
	$(TIME) rm -f $(FINAL_DIR)/bcparks_asset_attribute_predictions_long.csv
	$(TIME) rm -f $(FINAL_DIR)/bcparks_asset_attribute_predictions_wide.csv
	$(TIME) rm -f $(FINAL_DIR)/new_image_predictions_long.csv
	$(TIME) rm -f $(FINAL_DIR)/new_image_predictions_wide.csv
	$(TIME) rm -f $(FINAL_DIR)/demo_new_image_predictions_long.csv
	$(TIME) rm -f $(FINAL_DIR)/demo_new_image_predictions_wide.csv

clean-dinov3:
	@printf "\n==> clean-dinov3: removing generated DINOv3 feature CSVs for $(DINO_MODEL)\n"
	$(TIME) rm -f $(GENERATED_DINO_IMAGE_FEATURES)
	$(TIME) rm -f $(DINO_MASTER_FEATURES)
	$(TIME) rm -f $(DINO_RUN_FEATURES)
	$(TIME) rm -f $(FEATURE_DIR)/$(DINO_MODEL)_master_images_skipped.csv
	$(TIME) rm -f $(FEATURE_DIR)/$(DINO_MODEL)_all_attributes_images.csv
	$(TIME) rm -f $(FEATURE_DIR)/$(DINO_MODEL)_all_attributes_assets.csv
	$(TIME) rm -f $(FEATURE_DIR)/$(DINO_MODEL)_all_attributes_union_input.csv

clean-pipeline: clean-final clean-dinov3
	@printf "\n==> clean-pipeline complete. Model weights, raw data, train data, and cleaned images were preserved.\n"

clean: clean-pipeline
