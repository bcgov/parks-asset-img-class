#!/usr/bin/env bash
# Regenerate every eval-results plot with the larger slide-friendly fonts.
# Run from repo root:  bash scripts/regenerate_all_plots.sh
#
# Requires the MULTI_ATTRIBUTE_SERIES edit in plot_eval_results.py
# (boardwalk_low_v1 added) so Boardwalk folds into the gemini
# multi-attribute average for the gemini-vs-dinov3 plots.

set -euo pipefail

PLOT=scripts/plot_eval_results.py
OUT=results/eval_results_plots
VLM=results/vlm_eval_results/
DINO=results/dinov3_results
SIGLIP=results/siglip_results
OPENCLIP=results/openclip_results

mkdir -p "$OUT"

# Run both metrics for a given base set of args.
# $1 = output filename stem (no metric, no extension)
# rest = args
run_both() {
  local stem="$1"; shift
  for metric in macro_f1 weighted_f1; do
    python "$PLOT" --metric "$metric" \
      --output "$OUT/${stem}_${metric}.png" "$@"
  done
}

# ---------------------------------------------------------------
# 1. Pure VLM plots (per asset type, both prompt strategies shown)
# ---------------------------------------------------------------
run_both stairs        --input_dir "$VLM" --asset_type "Stairs" \
  --title "VLM attribute evaluation"
run_both trail_bridge  --input_dir "$VLM" --asset_type "Trail Bridge" \
  --title "VLM attribute evaluation"
run_both boardwalk_low --input_dir "$VLM" --asset_type "Boardwalk < 1.2m High" \
  --title "VLM attribute evaluation"

# ---------------------------------------------------------------
# 2. VLM + embedding comparison: DINOv3 vs gemini multi-attribute
#    (no --asset_type, so all 3 assets average together;
#     boardwalk_low_v1 folds in via MULTI_ATTRIBUTE_SERIES)
# ---------------------------------------------------------------
run_both gemini_vs_dinov3 \
  --dinov3_dir "$DINO" --input_dir "$VLM" \
  --include-series \
    "DINOv3 + Logistic Regression" \
    $'gemini-3-flash-preview\n(multi-attribute prompts)' \
  --title "DINOv3 vs. gemini-3-flash-preview (multi-attribute prompts)"

# ---------------------------------------------------------------
# 3. Pure embedding plots
# ---------------------------------------------------------------
# DINOv3 all classifiers, per attribute
run_both dinov3_comparison \
  --dinov3_dir "$DINO" \
  --title "DINOv3 classifiers comparison"

# DINOv3 all classifiers, aggregated across attributes
run_both agg_dinov3 \
  --dinov3_dir "$DINO" --aggregate --figsize 7 7 \
  --title "DINOv3 classifiers — mean across all attributes"

# 3-model logistic regression comparison, per attribute
run_both logistic_comparison \
  --dinov3_dir "$DINO" --siglip-dir "$SIGLIP" --openclip-dir "$OPENCLIP" \
  --include-series \
    "DINOv3 + Logistic Regression" \
    "SigLIP + Logistic Regression" \
    "OpenCLIP + Logistic Regression" \
  --title "DINOv3 vs. SigLIP vs. OpenCLIP"

# 3-model logistic regression comparison, aggregated
run_both agg_logistic_comparison \
  --dinov3_dir "$DINO" --siglip-dir "$SIGLIP" --openclip-dir "$OPENCLIP" \
  --aggregate --figsize 7 7 \
  --include-series \
    "DINOv3 + Logistic Regression" \
    "SigLIP + Logistic Regression" \
    "OpenCLIP + Logistic Regression" \
  --title "DINOv3 vs. SigLIP vs. OpenCLIP — mean across all attributes"

echo ""
echo "All plots regenerated in $OUT"