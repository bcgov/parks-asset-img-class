# VLM Walkthrough

This project uses Vision Language Models (VLMs) to directly predict BC Parks asset attribute labels
from asset images. VLMs take the images associated with each asset along with a prompt describing
what attributes to predict and their possible values, then output predicted attribute values and
confidence scores for each asset in a structured JSON format.

## Supported VLM Models

Models are organized by family in `src/vlm/config.py`. Each family uses a different API backend.

### Google AI Studio Models

```text
gemini-3-flash-preview
gemini-3.1-flash-lite
gemma-4-26b-a4b-it
```

Use Google Gemini API backend. Set authentication in an `.env` file via:

```bash
GEMINI_API_KEY="your-gemini-api-key"
```

Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### GitHub Models (OpenAI, Llama, Phi)

These families all use the same GitHub Models API backend and the same authentication method.

**OpenAI Models:**

```text
gpt-4o
```

**Llama Models:**

```text
Llama-3.2-11B-Vision-Instruct
```

**Phi Models:**

```text
Phi-4-multimodal-instruct
```

Use GitHub Models API backend. Set authentication in an `.env` file via:

```bash
GITHUB_TOKEN="your-github-token"
```

Create a personal access token in [GitHub Settings](https://github.com/settings/tokens) with `read:packages` scope.

## Setup

### Environment

Create and activate the project's Conda environment:

```bash
conda env update -f environment.yml --prune
conda activate bcparks_capstone
```

### API Keys

Create an `.env` file in the project root with the required API keys for the models you want to use.

```bash
# Google AI Studio
GEMINI_API_KEY="your-key-here"

# GitHub Models (OpenAI, Llama, Phi)
GITHUB_TOKEN="your-token-here"
```

An example of how to set up API keys in an `.env` file is provided in `.env.example`.

## Workflow

1. Read an input CSV from `data/processed/train/`.
2. For each unique `asset_id`:
   - Load all images for that asset from disk.
   - Encode images to base64.
   - Send images and prompt to VLM API.
   - Parse JSON response to extract attribute values and confidence scores.
3. Write results to output CSV with schema inferred from first successful parse.
4. Handle retries and exponential backoff for API robustness.

## Step 1: Run Batch Predictions

```bash
python scripts/run_vlm_predictor.py \
  --input data/processed/train/train_only_stairs.csv \
  --output results/vlm_stairs_gemini-3-flash.csv \
  --model gemini-3-flash-preview \
  --prompt stairs_v1
```

#### Required Arguments

- `--input`: CSV with `asset_id`, `image_path`, and `profile_name` columns.
- `--output`: Output CSV path (created if not present).
- `--model`: Model name (must be in `MODEL_FAMILIES` in `src/vlm/config.py`).
- `--prompt`: Prompt key from `PROMPT_REGISTRY` in `src/vlm/prompts.py`.

#### Optional Arguments

- `--image_root`: Image directory (default: `data/processed/images_clean`).
- `--limit`: Process only the first $N$ assets.
- `--offset`: Start from the Nth asset (useful for resuming interrupted runs).
- `--delay`: Delay in seconds between API calls (default: `0`).
- `--buffer_size`: How many results to batch before writing to CSV (default: `20`).

#### Output Schema

The output CSV is created dynamically based on the first successful model response.
Typical columns include:

```text
asset_id
timestamp
model
response                (raw JSON from model)
latency_s
<attribute>_value
<attribute>_confidence
parse_error             (1 if JSON parsing failed)
raw_response            (if parse_error = 1)
error                   (if API call failed)
traceback               (if API call failed)
```

For example, with the `stairs_v1` prompt, the output CSV would have columns like:

```text
asset_id, timestamp, model, response, latency_s, fall_height_value, fall_height_confidence, number_of_steps_value, number_of_steps_confidence, has_pedestrian_railing_value, has_pedestrian_railing_confidence, ...
```

### Prompt Registry

Prompts are defined in `src/vlm/prompts.py` and registered in `PROMPT_REGISTRY`.
Available prompt keys:

#### Asset-Type Prompts (Multi-Attribute)

Predict all attributes for an asset type in one call:

- `stairs_v1`: Stairs (fall_height, has_pedestrian_railing, material_frame_tank_body,
              number_of_steps, structure_position)
- `trail_bridge_v1`: Trail bridge (abutment_material, bridge_type, decking_material,
                     fall_height, has_pedestrian_railing, length, width, structure_material)

#### Attribute-Specific Prompts (Single Attribute)

Predict only one attribute:

- `structure_position_v1`: Structure position (`Elevated | At-Grade | Other`)
- `pedestrian_railing_v1`: Has pedestrian railing (`2 railings | 1 railing | No railings`)
- `steps_bin_v1`: Number of steps for stairs (`few (<10) | medium (10-20) | many (>20)`)
- `material_v1`: Material for stairs (`Timber/Wood | Concrete | Metal | Steel | etc.`)

#### Dynamic Prompts (Asset-Type Dependent)

Bins adjust based on `profile_name` (asset type):

- `fall_height_v1`: Fall height (bins vary by asset type)
- `length_v1`: Length (bins vary by asset type)
- `width_v1`: Width (bins vary by asset type)

### Example Workflows

#### Trail Bridge with GPT-4o

Multi-attribute prediction for all bridge assets using GPT-4o:

```bash
python scripts/run_vlm_predictor.py \
  --input data/processed/train/attr_bridge_type_train.csv \
  --output results/vlm_bridges_gpt4o.csv \
  --model gpt-4o \
  --prompt trail_bridge_v1
```

#### Single Attribute Across Models

Compare model performance on one attribute:

```bash
for model in gemini-3-flash-preview gpt-4o Llama-3.2-11B-Vision-Instruct; do
  python scripts/run_vlm_predictor.py \
    --input data/processed/train/attr_structure_position_train.csv \
    --output results/vlm_structure_position_${model}.csv \
    --model "$model" \
    --prompt structure_position_v1
done
```

#### Resuming Interrupted Runs

If a batch run fails partway through, resume from a specific offset
(skips first 50 assets that were already processed):

```bash
python scripts/run_vlm_predictor.py \
  --input data/processed/train/attr_number_of_steps_train.csv \
  --output results/vlm_stairs_gemini.csv \
  --model gemini-3-flash-preview \
  --prompt stairs_v1 \
  --offset 50
```

#### Rate-Limited Model with Delays

If hitting API rate limits, add delays (in seconds) between requests:

```bash
python scripts/run_vlm_predictor.py \
  --input data/processed/train/attr_number_of_steps_train.csv \
  --output results/vlm_stairs_slow.csv \
  --model gemini-3-flash-preview \
  --prompt stairs_v1 \
  --delay 0.5
```

## Step 2: Evaluate Predictions

Compare VLM predictions against ground truth labels:

```bash
python scripts/evaluate_predictions.py \
    --predictions results/vlm_stairs_gemini.csv \
    --ground_truth_dir data/processed/train \
    --attributes attr_structure_position attr_has_pedestrian_railing "attr_material_frame,_tank,_body" fall_height_bin steps_bin \
    --model gemini-3-flash-preview \
    --asset_type Stairs \
    --prompt_version stairs_v1
```

This computes accuracy, macro F1, and weighted F1 scores for the specified target attribute (e.g., `steps_bin`).

#### Required Arguments

- `--predictions`: CSV with predictions from `run_vlm_predictor.py`.
- `--ground_truth_dir`: Directory containing ground truth CSV files.
- `--attributes`: List of attributes to evaluate. Must match column names in ground truth CSVs (e.g., `attr_structure_position`).
- `--model`: Model name (must be in `MODEL_FAMILIES` in `src/vlm/config.py`).

#### Optional Arguments

- `--asset_type`: Type of asset being evaluated (default: `"unknown"`).
- `--prompt_version`: Version of the prompt used (default: `"v1"`).

## Adding New VLM Models

To add a new VLM to the supported model families:

#### 1. Register the Model

Add the model name to `MODEL_FAMILIES` in `src/vlm/config.py`:

```python
MODEL_FAMILIES = {
    "new_family": ["model-name-here", "another-model"],
    # ... existing models ...
}
```

#### 2. Add API Client (if new family)

If the model uses a new API provider not already supported, initialize the client in
`src/vlm/config.py`:

```python
from new_api import NewAPIClient

new_api_client = NewAPIClient(
    api_key=os.getenv("NEW_API_KEY")
)
```

#### 3. Add Message Formatter

In `src/vlm/model_router.py`, create a formatter function for the new API:

```python
def build_new_messages(prompt, images):
    content = []
    for img in images:
        content.append({
            "type": "image_url",
            "url": f"data:{img['mime']};base64,{img['b64']}"
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]
```

#### 4. Update Model Router

Add detection and routing logic to `run_model()` in `src/vlm/model_router.py`:

```python
def run_model(model_name, prompt, images):
    family = detect_model_family(model_name)

    if family == "gemini":
        # ... existing gemini logic ...
        
    elif family == "new_family":
        messages = build_new_messages(prompt, images)
        response = new_api_client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response.choices[0].message.content

    else:
        # ... existing openai default logic ...
```

#### 5. Test

Test the new model with a smoke run:

```bash
python scripts/run_vlm_predictor.py \
  --input data/processed/train/attr_number_of_steps_train.csv \
  --output results/test_new_model.csv \
  --model model-name-here \
  --prompt stairs_v1 \
  --limit 5
```

## Adding New Prompts

Create prompts for new attributes or asset types in `src/vlm/prompts.py`.

### Static Prompts

For attributes with fixed options across all asset types:

```python
MY_ATTRIBUTE_PROMPT = """
    You are an expert in park infrastructure analysis.
    
    Using ALL provided images of this single asset, identify the attribute value.
    
    Predict exactly ONE value from the listed options:
    - Option 1
    - Option 2
    - Option 3
    
    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {
        "my_attribute": {
            "value": "<predicted value or 'unable to determine'>",
            "confidence": <float 0.0-1.0>
        }
    }
    
    If you cannot determine the attribute from the images, set value to
    "unable to determine" and confidence to 0.0.
"""

PROMPT_REGISTRY = {
    "my_attribute_v1": MY_ATTRIBUTE_PROMPT,
    # ... existing prompts ...
}
```

### Dynamic Prompts

For prompts that need asset-type-specific options (e.g., different fall height bins):

```python
def make_my_dynamic_prompt(asset_type):
    if asset_type == "Stairs":
        bins = "low (<0.5m) | medium (0.5-1.2m) | high (>1.2m)"
    elif asset_type == "Trail Bridge":
        bins = "low (<1.2m) | medium (1.2-5m) | high (>5m)"
    else:
        bins = "low | medium | high"
    
    return f"""
        You are an expert in park infrastructure analysis.
        
        Using ALL provided images of this single {asset_type} asset, estimate the attribute.
        
        Predict exactly ONE value from the listed options for this asset type:
        {bins}
        
        Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
        {{
            "my_attribute": {{"value": "<bin label>", "confidence": <float 0.0-1.0>}}
        }}
        
        If you cannot determine the attribute, set value to "unable to determine" 
        and confidence to 0.0.
    """

PROMPT_REGISTRY = {
    "my_dynamic_prompt_v1": make_my_dynamic_prompt,
    # ... existing prompts ...
}
```

## What Each File Does

```text
src/vlm/config.py
```

API client initialization and model family registry. Add new VLM families and
API keys here.

```text
src/vlm/model_router.py
```

Model detection and message formatting for each API backend (Gemini, OpenAI,
Llama, Phi). Add new model families and backend handling here.

```text
src/vlm/predictors.py
```

High-level prediction function (`predict_asset_attributes`). Orchestrates image
loading, model calling, and error handling.

```text
src/vlm/image_loader.py
```

Image file resolution and base64 encoding. Handles PNG and JPEG files.

```text
src/vlm/prompts.py
```

Prompt templates and registry. Add all new prompts here (static or dynamic).

```text
scripts/run_vlm_predictor.py
```

Command-line entry point for batch prediction. Handles CSV I/O, asset iteration,
retries, and result CSV schema inference.

```text
scripts/evaluate_predictions.py
```

Evaluate predictions against ground truth labels. Computes accuracy, macro F1,
and weighted F1 scores.
