"""
Batch VLM predictor script

Usage:
    python scripts/run_vlm_predictor.py \
        --input data/processed/train/attr_number_of_steps_train.csv \
        --output results/vlm_stairs_gemini-3-flash.csv \
        --model gemini-3-flash-preview \
        --prompt stairs_v1
"""

import argparse
import os
import sys
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time

# ---------------------------------------------------------------------
# Ensure project root is in import path
# ---------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.vlm.predictors import predict_asset_attributes
from src.vlm.prompts import PROMPT_REGISTRY

DEFAULT_IMAGE_ROOT = "data/processed/images_clean"

BIN_COL_MAPPING = {
    "fall_height": "fall_height_bin",
    "number_of_steps": "steps_bin",
    "length": "length_bin",
    "width": "width_bin",
}

# JSON parsing (handles markdown and bad outputs)
def safe_json_loads(text: str):
    if not text:
        return None

    text = text.strip()

    # strip markdown code blocks
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

# Results CSV header inference
def get_output_columns(parsed_attrs: list[str]) -> list[str]:
    parsed_cols = [
        f"{BIN_COL_MAPPING.get(attr, attr)}_{suffix}"
        for attr in parsed_attrs
        for suffix in ("value", "confidence")
    ]
    return (
        ["asset_id", "timestamp", "model", "response", "latency_s"]
        + parsed_cols
        + ["parse_error", "raw_response", "error", "traceback"]
    )

FALLBACK_COLUMNS = ["asset_id", "timestamp", "model", "error", "traceback"]


# Retry wrapper for API robustness
def with_retry(fn, retries=2, delay=1.5):
    for i in range(retries + 1):
        try:
            return fn()
        except Exception:
            if i == retries:
                raise
            time.sleep(delay * (2 ** i))  # exponential backoff

# Buffer flush
def flush_buffer(buffer: list, output_path: str, output_columns: list):
    pd.DataFrame(buffer) \
      .reindex(columns=output_columns) \
      .to_csv(output_path, mode="a", header=False, index=False)


# ---------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------
def run_batch(
    input_path,
    output_path,
    model_name,
    prompt_or_fn,
    image_root=DEFAULT_IMAGE_ROOT,
    limit=None,
    offset=0,
    delay=0,
    buffer_size=20): # batch writes to results CSV instead of row-by-row

    print(f"Loading input from: {input_path}")
    df = pd.read_csv(input_path)

    # early validation
    if "asset_id" not in df.columns:
        raise ValueError("Input file must contain an 'asset_id' column")
    if "profile_name" not in df.columns:
        raise ValueError("Input file must contain a 'profile_name' column")

    unique_asset_ids = df["asset_id"].unique()
    unique_asset_ids = unique_asset_ids[offset:]
    if limit:
        unique_asset_ids = unique_asset_ids[:limit]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Running model: {model_name}")
    print(f"Total assets to process: {len(unique_asset_ids)}")
    print(f"Offset: {offset}")
    print(f"Writing results to: {output_path}")

    write_header = not os.path.exists(output_path)

    output_columns = None  # unknown until first successful parse
    buffer = []

    def process_asset(asset_id):
        asset_df = df[df["asset_id"] == asset_id]
        asset_type = asset_df["profile_name"].iloc[0]

        prompt = (
            prompt_or_fn(asset_type)
            if callable(prompt_or_fn)
            else prompt_or_fn
        )

        start_time = time.time()

        try:
            # retry wrapper
            result = with_retry(
                lambda: predict_asset_attributes(
                    asset_id=int(asset_id),
                    df=asset_df,
                    model_name=model_name,
                    prompt=prompt,
                    image_root=image_root
                )
            )

            out = {
                "asset_id": int(asset_id),
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "response": result.get("response"),
                "latency_s": round(time.time() - start_time, 3)
            }

        except Exception as e:
            out = {
                "asset_id": int(asset_id),
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "error": str(e)
            }
            return out, None

        # parsing pipeline
        parsed = safe_json_loads(result.get("response"))

        if not parsed:
            out["parse_error"] = True
            out["raw_response"] = result.get("response")
            return out, None

        if isinstance(parsed, dict):

            for attr, val in parsed.items():
                col_name = BIN_COL_MAPPING.get(attr, attr)

                # safe access (avoid crash if val malformed)
                if isinstance(val, dict):
                    out[f"{col_name}_value"] = val.get("value")
                    out[f"{col_name}_confidence"] = val.get("confidence")

        return out, list(parsed.keys())

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
    for asset_id in tqdm(unique_asset_ids):

        out, parsed_attrs = process_asset(asset_id)

        # infer schema and write header from first successful parse
        if output_columns is None and parsed_attrs is not None:
            output_columns = get_output_columns(parsed_attrs)
            pd.DataFrame(columns=output_columns).to_csv(output_path, index=False)

        buffer.append(out)

        if len(buffer) >= buffer_size:
            if output_columns is not None:
                flush_buffer(buffer, output_path, output_columns)
                buffer.clear()
            # if still None, keep buffering until we get a successful parse

        time.sleep(delay)

    # if we never got a successful parse, fall back to minimal schema
    if output_columns is None:
        output_columns = FALLBACK_COLUMNS
        pd.DataFrame(columns=output_columns).to_csv(output_path, index=False)

    # flush remaining buffer
    if buffer:
        flush_buffer(buffer, output_path, output_columns)

    print("✅ Done!", len(unique_asset_ids), "assets processed.")
    print(f"Next offset: {offset + len(unique_asset_ids)}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch VLM predictor script.")

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt",
        required=True,
        help=f"Prompt key. Available: {list(PROMPT_REGISTRY.keys())}",
    )
    parser.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0)
    parser.add_argument("--buffer_size", type=int, default=20)

    args = parser.parse_args()

    prompt_or_fn = PROMPT_REGISTRY.get(args.prompt)
    if prompt_or_fn is None:
        raise ValueError(
            f"Unknown prompt key: {args.prompt}. "
            f"Available: {list(PROMPT_REGISTRY.keys())}"
        )

    run_batch(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        prompt_or_fn=prompt_or_fn,
        image_root=args.image_root,
        limit=args.limit,
        offset=args.offset,
        delay=args.delay,
        buffer_size=args.buffer_size
    )