"""
Merge predictions into one CSV file

Example usage:
    python scripts/merge_predictions.py \
        --input_files results/vlm_stairs_attributes/*_gemini*.csv \
        --output results/vlm_stairs_attributes/merged_stairs_attributes_gemini.csv

"""


import glob
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_files", required=True, nargs="+")
parser.add_argument("--output", required=True)
args = parser.parse_args()

frames = [pd.read_csv(f) for f in args.input_files]
merged = pd.concat(frames, ignore_index=True).drop_duplicates("asset_id")
merged.to_csv(args.output, index=False)
print(f"Merged {len(frames)} files -> {args.output}")