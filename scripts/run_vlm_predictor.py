"""
Batch VLM predictor script

Usage:
    python scripts/run_vlm_predictor.py \
        --input data/processed/train/attr_number_of_steps_train.csv \
        --output results/vlm_stairs_gemini-3-flash.csv \
        --provider gemini \
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

from pathlib import Path
from scripts.download_citywide_images import PROFILES

VALID_ASSET_TYPES = set(PROFILES.values())
PROFILE_ID_TO_NAME = {str(pid): name for pid, name in PROFILES.items()}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def build_input_table_from_folder(
    image_folder: Path, asset_type: str | None, image_root: str
) -> pd.DataFrame:
    """Walk a folder of images into rows of asset_id, image_path, profile_name.

    Mirrors predict_new_images.build_input_table so the VLM reads the same
    structure DINOv3 does. image_path is written as ``data/<path relative to
    image_root>`` so src/vlm/image_loader.py resolves it via its
    ``.replace("data/", image_root + "/")`` step.

    Flat mode (asset_type given):  <folder>/<asset_id>/<images>
    Profile mode (no asset_type):  <folder>/<profile_id>/<asset_id>/<images>
    """
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    root = Path(ROOT)
    image_root_abs = (root / image_root).resolve()

    def rel_image_path(img: Path) -> str:
        # path the loader expects: "data/" + (image relative to image_root)
        rel = img.resolve().relative_to(image_root_abs)
        return "data/" + rel.as_posix()

    rows: list[dict[str, object]] = []

    if asset_type is not None:
        if asset_type not in VALID_ASSET_TYPES:
            raise ValueError(
                f"--asset-type {asset_type!r} is not a known asset type. "
                f"Valid: {sorted(VALID_ASSET_TYPES)}"
            )
        asset_dirs = [p for p in sorted(image_folder.iterdir()) if p.is_dir()]
        if not asset_dirs:
            for img in sorted(image_folder.iterdir()):
                if _is_image(img):
                    rows.append({
                        "asset_id": img.stem,
                        "image_path": rel_image_path(img),
                        "profile_name": asset_type,
                    })
        else:
            for asset_dir in asset_dirs:
                for img in sorted(asset_dir.iterdir()):
                    if _is_image(img):
                        rows.append({
                            "asset_id": int(asset_dir.name),
                            "image_path": rel_image_path(img),
                            "profile_name": asset_type,
                        })
    else:
        profile_dirs = [p for p in sorted(image_folder.iterdir()) if p.is_dir()]
        if not profile_dirs:
            raise ValueError(
                "No profile_id subfolders found and no --asset-type given. "
                "Either pass --asset-type for a flat folder, or use the "
                "<profile_id>/<asset_id>/<images> layout."
            )
        for profile_dir in profile_dirs:
            profile_name = PROFILE_ID_TO_NAME.get(profile_dir.name)
            if profile_name is None:
                print(f"  [skip] '{profile_dir.name}' is not a known profile_id; skipping.")
                continue
            for asset_dir in sorted(profile_dir.iterdir()):
                if not asset_dir.is_dir():
                    continue
                for img in sorted(asset_dir.iterdir()):
                    if _is_image(img):
                        rows.append({
                            "asset_id": int(asset_dir.name),
                            "image_path": rel_image_path(img),
                            "profile_name": profile_name,
                        })

    if not rows:
        raise ValueError(f"No images found under {image_folder}.")
    return pd.DataFrame(rows)

DEFAULT_IMAGE_ROOT = "data/processed/images_clean"

BIN_COL_MAPPING = {
    "fall_height": "fall_height_bin",
    "number_of_steps": "steps_bin",
    "length": "length_bin",
    "width": "width_bin",
}

# JSON parsing (handles markdown and bad outputs)
def safe_json_loads(text: str):
    """Parse a JSON string and return None when parsing fails."""
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
    """Return the output columns expected for a prompt template."""
    parsed_cols = [
        f"{BIN_COL_MAPPING.get(attr, attr)}_{suffix}"
        for attr in parsed_attrs
        for suffix in ("value", "confidence")
    ]
    return (
        ["asset_id", "timestamp", "provider", "model", "response", "latency_s"]
        + parsed_cols
        + ["parse_error", "raw_response", "error", "traceback"]
    )

FALLBACK_COLUMNS = ["asset_id", "timestamp", "provider", "model", "error", "traceback"]


# Retry wrapper for API robustness
def with_retry(fn, retries=2, delay=1.5):
    """Call a provider function with simple retry handling."""
    for i in range(retries + 1):
        try:
            return fn()
        except Exception:
            if i == retries:
                raise
            time.sleep(delay * (2 ** i))  # exponential backoff

# Buffer flush
def flush_buffer(buffer: list, output_path: str, output_columns: list):
    """Append buffered prediction rows to disk."""
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
    provider="auto",
    image_root=DEFAULT_IMAGE_ROOT,
    limit=None,
    offset=0,
    delay=0,
    buffer_size=20,
    max_tokens=4096,
    asset_type=None): # batch writes to results CSV instead of row-by-row

    """Run VLM prediction over an input batch and write outputs incrementally."""
    print(f"Loading input from: {input_path}")
    input_path_obj = Path(input_path)
    if input_path_obj.is_dir():
        print(f"Detected folder input; walking {input_path_obj}")
        df = build_input_table_from_folder(input_path_obj, asset_type, image_root)
    else:
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
    
    # skip already processed assets
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            existing = pd.read_csv(output_path)
            already_done = set(existing["asset_id"].tolist())
            unique_asset_ids = [a for a in unique_asset_ids if a not in already_done]
            print(f"Skipping {len(already_done)} already processed assets")
            print(f"Remaining: {len(unique_asset_ids)} assets")
        except Exception:
            pass  # if we can't read it, just proceed normally

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Running provider/model: {provider} / {model_name}")
    print(f"Total assets to process: {len(unique_asset_ids)}")
    print(f"Offset: {offset}")
    print(f"Writing results to: {output_path}")

    write_header = not os.path.exists(output_path)

    output_columns = None  # unknown until first successful parse
    last_parsed_attrs = None
    buffer = []

    def process_asset(asset_id):
        """Run one VLM prediction request and return a row for one asset."""
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
                    image_root=image_root,
                    provider=provider,
                    max_tokens=max_tokens
                )
            )

            out = {
                "asset_id": int(asset_id),
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
                "model": model_name,
                "response": result.get("response"),
                "latency_s": round(time.time() - start_time, 3)
            }

        except Exception as e:
            out = {
                "asset_id": int(asset_id),
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
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
        if parsed_attrs is not None:
            last_parsed_attrs = parsed_attrs

        # infer schema and write header from first successful parse
        if output_columns is None and parsed_attrs is not None:
            output_columns = get_output_columns(parsed_attrs)
            if not os.path.exists(output_path):
                pd.DataFrame(columns=output_columns).to_csv(output_path, index=False)

        buffer.append(out)

        if len(buffer) >= buffer_size:
            if output_columns is not None:
                flush_buffer(buffer, output_path, output_columns)
                buffer.clear()
            # if still None, keep buffering until we get a successful parse

        time.sleep(delay)

    # if we never got a successful parse, fall back to minimal schema
    # infer schema and write header from first successful parse
    if output_columns is None and last_parsed_attrs is not None:
        output_columns = get_output_columns(last_parsed_attrs)
        # only write header if file doesn't exist OR is empty
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            pd.DataFrame(columns=output_columns).to_csv(output_path, index=False)
        else:
            # file exists with content - check if schema matches
            existing = pd.read_csv(output_path, nrows=0)
            if list(existing.columns) != output_columns:
                # schema mismatch - overwrite with new header
                pd.DataFrame(columns=output_columns).to_csv(output_path, index=False)

    if output_columns is None:
        output_columns = FALLBACK_COLUMNS
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
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
    parser.add_argument(
        "--provider",
        default="auto",
        help="VLM provider: auto, gemini, openai, grok, claude, or github.",
    )
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
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--asset-type", default=None,
                        help="Asset type for flat folder input, e.g. 'Stairs'. "
                             "Omit for CSV input or profile_id/asset_id folder layout.")

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
        provider=args.provider,
        limit=args.limit,
        offset=args.offset,
        delay=args.delay,
        buffer_size=args.buffer_size,
        max_tokens=args.max_tokens,
        image_root=args.image_root,
        asset_type=args.asset_type,
    )
