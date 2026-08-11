"""
Merge attribute-specific prediction CSVs column-wise for heatmap plotting.

Usage:
    python scripts/merge_attribute_predictions.py \
        --input_files results/vlm_structure_position_gemini_preview_all.csv \
                      results/vlm_pedestrian_railing_gemini_preview_all.csv \
                      results/vlm_material_gemini_preview_all.csv \
                      results/vlm_fall_height_gemini_preview_all.csv \
                      results/vlm_steps_bin_gemini_preview_all.csv \
        --output results/vlm_stairs_gemini_attribute_specific_combined.csv
"""

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_files", required=True, nargs="+")
parser.add_argument("--output", required=True)
args = parser.parse_args()

# metadata columns to keep from first file only
METADATA_COLS = ["asset_id", "timestamp", "model", "response", "latency_s"]

frames = [pd.read_csv(f) for f in args.input_files]

# start with first file
merged = frames[0]

# column-wise merge — add new attribute columns from each subsequent file
for df in frames[1:]:
    new_cols = ["asset_id"] + [c for c in df.columns if c not in merged.columns]
    merged = merged.merge(df[new_cols], on="asset_id", how="outer")

merged.to_csv(args.output, index=False)
print(f"Merged {len(frames)} files -> {args.output}")
print(f"Total assets: {merged['asset_id'].nunique()}")
print(f"Columns: {merged.columns.tolist()}")