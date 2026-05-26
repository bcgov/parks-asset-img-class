#!/usr/bin/env python3
"""
Batch VLM predictor script.

Usage:
    python scripts/run_vlm_predictor.py \
        --input data/processed/train/attr_number_of_steps_train.csv \
        --output results/vlm_predictions_stairs_gemma.csv \
        --model gemini-3-flash-preview \
        --prompt stairs_v1 \
        --limit 10 \
        --offset 0 \
        --delay 5
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

import signal

def timeout_handler(signum, frame):
    raise TimeoutError("API call timed out after 60s")

# ---------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------
def run_batch(input_path, output_path, model_name, prompt_or_fn, limit=None, offset=0, delay=5.0):
    print(f"Loading input from: {input_path}")
    df = pd.read_csv(input_path)

    if "asset_id" not in df.columns:
        raise ValueError("Input file must contain an 'asset_id' column")
    
    unique_asset_ids = df['asset_id'].unique()

    unique_asset_ids = unique_asset_ids[offset:]

    if limit:
        unique_asset_ids = unique_asset_ids[:limit]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Running model: {model_name}")
    print(f"Total assets to process: {len(unique_asset_ids)}")
    print(f"Offset: {offset}")
    print(f"Writing results to: {output_path}")

    for asset_id in tqdm(unique_asset_ids):
        asset_df = df[df["asset_id"] == asset_id]

        # get asset type for dynamic prompts
        asset_type = asset_df["profile_name"].iloc[0]

        # resolve prompt — function or static string
        if callable(prompt_or_fn):
            prompt = prompt_or_fn(asset_type)
        else:
            prompt = prompt_or_fn
        
        result = None

        try:
            result = predict_asset_attributes(
                asset_id=int(asset_id),
                df=asset_df,
                model_name=model_name,
                prompt=prompt
            )

            out = {
                "asset_id": int(asset_id),
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "response": result.get("response"),
            }

        except Exception as e:
            out = {
                "asset_id": int(asset_id),
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "error": str(e),
            }
        
        parsed = None
        if result is not None:
            parsed = result.get("response")
    
            if not parsed:  # catches None and empty string
                out["parse_error"] = True
                pd.DataFrame([out]).to_csv(output_path, mode="a", header=not os.path.exists(output_path), index=False)
                continue

            if isinstance(parsed, str):
                # strip markdown code blocks
                parsed = parsed.strip()
                if parsed.startswith("```"):
                    parsed = parsed.split("```")[1]
                    if parsed.startswith("json"):
                        parsed = parsed[4:]
                    parsed = parsed.strip()
                try:
                    parsed = json.loads(parsed)
                except json.JSONDecodeError:
                    out["parse_error"] = True
                    pd.DataFrame([out]).to_csv(output_path, mode="a", header=not os.path.exists(output_path), index=False)
                    continue
                    
            if isinstance(parsed, dict):
            # map numeric attribute keys to their binned column names
                BIN_COL_MAPPING = {
                    "fall_height": "fall_height_bin",
                    "number_of_steps": "steps_bin",
                    "length": "length_bin",
                    "width": "width_bin",
                }
                
                for attr, val in parsed.items():
                    col_name = BIN_COL_MAPPING.get(attr, attr)
                    out[f"{col_name}_value"] = val.get("value")
                    out[f"{col_name}_confidence"] = val.get("confidence")
            
        pd.DataFrame([out]).to_csv(output_path, mode="a", header=not os.path.exists(output_path), index=False)

        time.sleep(delay)

    print("✅ Done! Processed", len(unique_asset_ids), "assets.")
    print(f"Next offset: {offset + len(unique_asset_ids)}")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch VLM predictor script.")
    parser.add_argument("--input", required=True, help="Path to training data (CSV)")
    parser.add_argument("--output", required=True, help="Path to store JSONL results")
    parser.add_argument("--model", required=True, help="VLM model name")
    parser.add_argument("--prompt", required=True, 
                        help=f"Prompt key from registry. Available: {list(PROMPT_REGISTRY.keys())}")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging")
    parser.add_argument("--offset", type=int, default=0, 
                        help="Skip first N assets (for resuming after rate limit)")
    parser.add_argument("--delay", type=float, default=5.0,
                    help="Seconds to wait between API calls (default 5s = safe for 15 req/min )")

    args = parser.parse_args()

    prompt_or_fn = PROMPT_REGISTRY.get(args.prompt)
    if prompt_or_fn is None:
        raise ValueError(f"Unknown prompt key: {args.prompt}. Available: {list(PROMPT_REGISTRY.keys())}")

    run_batch(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        prompt_or_fn=prompt_or_fn,
        limit=args.limit,
        offset=args.offset,
        delay=args.delay
    )