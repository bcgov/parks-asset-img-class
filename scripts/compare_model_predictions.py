"""Compare VLM (gemini) vs DINOv3 embedding predictions on the SAME assets.

The two approaches were evaluated on different asset pools, so this script
inner-joins on asset_id per attribute and reports the OVERLAP first, then
buckets each overlapping asset into:

    both_right   - VLM correct AND embedding correct
    vlm_only     - VLM correct, embedding wrong   (justifies routing to VLM)
    embed_only   - embedding correct, VLM wrong   (justifies routing to DINOv3)
    both_wrong   - both wrong                      (honest gaps / future work)

Step 1 (default): print a table across all attributes so you can see which
attributes have enough overlap to make good comparison slides.

    python scripts/compare_model_predictions.py

Step 2: generate base64-image HTML reports for chosen attribute + bucket(s):

    python scripts/compare_model_predictions.py \
        --attribute attr_decking_material --buckets both_right \
        --limit 20

Open the printed HTML in a browser (images are embedded, so they always render).
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse helpers from the existing scripts (unchanged).
from src.prediction_inspection import find_ground_truth_file  # noqa: E402
from scripts.inspect_wrong_predictions import (  # noqa: E402
    create_error_report_html,
)

# ---------------------------------------------------------------------------
# Per-attribute configuration
#   vlm_col      : column in the VLM CSV holding the predicted label (no attr_ prefix)
#   embed_stem   : stem of the dinov3 predictions file (attr_ prefix kept for attr_* ones)
#   asset_types  : which per-asset-type VLM files contain this attribute (in-scope only)
#   gt_attribute : name passed to find_ground_truth_file (locates <name>_train.csv)
# ---------------------------------------------------------------------------
ATTRIBUTES = {
    "attr_abutment_material": dict(
        vlm_col="abutment_material_value",
        embed_stem="attr_abutment_material",
        asset_types=["bridge"],
        gt_attribute="attr_abutment_material",
    ),
    "attr_bridge_type": dict(
        vlm_col="bridge_type_value",
        embed_stem="attr_bridge_type",
        asset_types=["bridge"],
        gt_attribute="attr_bridge_type",
    ),
    "attr_decking_material": dict(
        vlm_col="decking_material_value",
        embed_stem="attr_decking_material",
        asset_types=["boardwalk_low", "bridge"],
        gt_attribute="attr_decking_material",
    ),
    "fall_height_bin": dict(
        vlm_col="fall_height_bin_value",
        embed_stem="fall_height_bin",
        asset_types=["boardwalk_low", "stairs", "bridge"],
        gt_attribute="fall_height_bin",
    ),
    "attr_has_edge_guard": dict(
        vlm_col="has_edge_guard_value",
        embed_stem="attr_has_edge_guard",
        asset_types=["boardwalk_low"],
        gt_attribute="attr_has_edge_guard",
    ),
    "attr_has_pedestrian_railing": dict(
        vlm_col="has_pedestrian_railing_value",
        embed_stem="attr_has_pedestrian_railing",
        asset_types=["boardwalk_low", "stairs", "bridge"],
        gt_attribute="attr_has_pedestrian_railing",
    ),
    "length_bin": dict(
        vlm_col="length_bin_value",
        embed_stem="length_bin",
        asset_types=["boardwalk_low", "bridge"],
        gt_attribute="length_bin",
    ),
    "attr_material_frame_tank_body": dict(
        vlm_col="material_frame_tank_body_value",
        embed_stem="attr_material_frame_tank_body",
        asset_types=["stairs"],
        gt_attribute="attr_material_frame,_tank,_body",
    ),
    "steps_bin": dict(
        vlm_col="steps_bin_value",
        embed_stem="steps_bin",
        asset_types=["stairs"],
        gt_attribute="steps_bin",
    ),
    "attr_structure_material": dict(
        vlm_col="structure_material_value",
        embed_stem="attr_structure_material",
        asset_types=["boardwalk_low", "bridge"],
        gt_attribute="attr_structure_material",
    ),
    "attr_structure_position": dict(
        vlm_col="structure_position_value",
        embed_stem="attr_structure_position",
        asset_types=["stairs"],
        gt_attribute="attr_structure_position",
    ),
    "width_bin": dict(
        vlm_col="width_bin_value",
        embed_stem="width_bin",
        asset_types=["boardwalk_low", "bridge"],
        gt_attribute="width_bin",
    ),
}

# Glob tokens used to locate each asset type's VLM predictions CSV.
ASSET_TYPE_GLOB = {
    "stairs": "*stairs*gemini*complete*.csv",
    "bridge": "*bridge*gemini*complete*.csv",
    "boardwalk_low": "*boardwalk_low*gemini*complete*.csv",
}

PRED_COLUMN = "predicted_value"   # what create_error_report_html reads
GT_COLUMN = "true_value"

COLORS = {  # just for console readability
    "both_right": "both_right",
    "vlm_only": "vlm_only",
    "embed_only": "embed_only",
    "both_wrong": "both_wrong",
}


def _norm(series: pd.Series) -> pd.Series:
    """Normalize a label value for robust string comparisons."""
    return series.astype(str).str.strip().str.lower()


def find_vlm_files(results_root: str) -> dict[str, Path]:
    """Locate one VLM predictions CSV per in-scope asset type."""
    found = {}
    for atype, pattern in ASSET_TYPE_GLOB.items():
        matches = glob.glob(str(Path(results_root) / "**" / pattern), recursive=True)
        if matches:
            found[atype] = Path(sorted(matches)[0])
    return found


def load_vlm_for_attribute(cfg: dict, vlm_files: dict[str, Path]) -> pd.DataFrame:
    """Concat the relevant per-asset-type VLM files; return asset_id + vlm value."""
    frames = []
    for atype in cfg["asset_types"]:
        path = vlm_files.get(atype)
        if path is None:
            continue
        df = pd.read_csv(path)
        if cfg["vlm_col"] not in df.columns or "asset_id" not in df.columns:
            continue
        sub = df[["asset_id", cfg["vlm_col"]]].copy()
        sub = sub.rename(columns={cfg["vlm_col"]: "vlm_pred"})
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["asset_id", "vlm_pred"])
    out = pd.concat(frames, ignore_index=True)
    out["asset_id"] = out["asset_id"].astype(str).str.strip()
    out = out.dropna(subset=["vlm_pred"]).drop_duplicates("asset_id")
    return out


def load_embed_for_attribute(cfg: dict, embed_dir: str) -> pd.DataFrame:
    """Load the dinov3 predictions file for this attribute."""
    fname = f"dinov3_{cfg['embed_stem']}_classification_predictions.csv"
    path = Path(embed_dir) / fname
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    # Drop junk rows with no prediction/label, then keep one row per asset
    df = df.dropna(subset=["true_label", "predicted_label", "correct"])
    df = df.drop_duplicates("asset_id", keep="first")
    return df


def resolve_truth(cfg: dict, gt_dir: str) -> pd.DataFrame:
    """Asset-level ground-truth label from the train CSV."""
    gt_path = find_ground_truth_file(cfg["gt_attribute"], gt_dir)
    if gt_path is None:
        return pd.DataFrame()
    gt = pd.read_csv(gt_path)
    # find the label column (same logic family as the inspection helper)
    candidates = [
        cfg["embed_stem"],
        cfg["embed_stem"].replace("attr_", ""),
        cfg["gt_attribute"],
    ]
    label_col = next((c for c in candidates if c in gt.columns), None)
    if label_col is None:
        return pd.DataFrame()
    gt["asset_id"] = gt["asset_id"].astype(str).str.strip()
    out = gt[["asset_id", label_col]].dropna().drop_duplicates("asset_id")
    out = out.rename(columns={label_col: "true_label"})
    return out


def build_comparison(attr: str, cfg: dict, vlm_files, embed_dir, gt_dir):
    """Return a per-asset DataFrame with vlm_correct, embed_correct, bucket."""
    vlm = load_vlm_for_attribute(cfg, vlm_files)
    embed = load_embed_for_attribute(cfg, embed_dir)
    truth = resolve_truth(cfg, gt_dir)

    if vlm.empty or embed.empty or truth.empty:
        return None

    # VLM correctness vs ground truth
    vlm = vlm.merge(truth, on="asset_id", how="inner")
    vlm["vlm_correct"] = _norm(vlm["vlm_pred"]) == _norm(vlm["true_label"])

    # Embedding already has correctness + predicted label
    embed_small = embed[["asset_id", "predicted_label", "correct"]].rename(
        columns={"predicted_label": "embed_pred", "correct": "embed_correct"}
    )

    merged = vlm.merge(embed_small, on="asset_id", how="inner")
    if merged.empty:
        return merged

    def bucket(row):
        """Assign an overlap row to a VLM/embedding correctness bucket."""
        if row["vlm_correct"] and row["embed_correct"]:
            return "both_right"
        if row["vlm_correct"] and not row["embed_correct"]:
            return "vlm_only"
        if not row["vlm_correct"] and row["embed_correct"]:
            return "embed_only"
        return "both_wrong"

    merged["bucket"] = merged.apply(bucket, axis=1)
    return merged


def print_overview(vlm_files, embed_dir, gt_dir):
    """Print overlap and correctness summaries for compared predictions."""
    print("\n=== VLM files found ===")
    for atype, path in vlm_files.items():
        print(f"  {atype:14s} -> {path}")
    missing = [a for a in ASSET_TYPE_GLOB if a not in vlm_files]
    if missing:
        print(f"  (not found: {missing})")

    rows = []
    for attr, cfg in ATTRIBUTES.items():
        merged = build_comparison(attr, cfg, vlm_files, embed_dir, gt_dir)
        if merged is None or merged.empty:
            rows.append((attr, 0, 0, 0, 0, 0, 0))
            continue
        counts = merged["bucket"].value_counts()
        rows.append((
            attr,
            len(merged),  # overlap
            counts.get("both_right", 0),
            counts.get("vlm_only", 0),
            counts.get("embed_only", 0),
            counts.get("both_wrong", 0),
            0,
        ))

    print("\n=== Overlap + bucket counts (assets in BOTH models' eval sets) ===")
    hdr = f"{'attribute':32s} {'overlap':>7s} {'both_R':>7s} {'vlm_R':>6s} {'embed_R':>8s} {'both_W':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for (attr, overlap, br, vo, eo, bw, _) in rows:
        print(f"{attr:32s} {overlap:7d} {br:7d} {vo:6d} {eo:8d} {bw:7d}")
    print("\nvlm_R = VLM right / embed wrong (route to VLM)")
    print("embed_R = embed right / VLM wrong (route to DINOv3)")


def generate_html(attr, cfg, vlm_files, embed_dir, gt_dir, train_dir,
                  buckets, output_dir, limit):
    """Generate an HTML report showing compared model predictions and images."""
    merged = build_comparison(attr, cfg, vlm_files, embed_dir, gt_dir)
    if merged is None or merged.empty:
        print(f"No overlapping assets for {attr}; nothing to render.")
        return

    # Attach all image paths per asset from the train CSV.
    gt_path = find_ground_truth_file(cfg["gt_attribute"], train_dir)
    train = pd.read_csv(gt_path)
    train["asset_id"] = train["asset_id"].astype(str).str.strip()
    image_rows = train[["asset_id", "image_path"]].dropna().drop_duplicates()

    out_base = Path(output_dir) / attr
    for bucket in buckets:
        sub = merged[merged["bucket"] == bucket]
        if sub.empty:
            print(f"  [{bucket}] empty, skipping")
            continue
        if limit:
            keep_ids = sub["asset_id"].unique()[:limit]
            sub = sub[sub["asset_id"].isin(keep_ids)]

        # Expand to one row per image; set the columns the renderer reads.
        expanded = sub.merge(image_rows, on="asset_id", how="left")
        expanded["filename"] = expanded["image_path"].apply(
            lambda p: Path(str(p)).name if pd.notna(p) else ""
        )
        expanded[PRED_COLUMN] = (
            "VLM: " + expanded["vlm_pred"].astype(str)
            + "   |   DINOv3: " + expanded["embed_pred"].astype(str)
        )
        expanded[GT_COLUMN] = expanded["true_label"]

        create_error_report_html(
            expanded,            # treated as the "wrong_preds" rows to show
            expanded,            # merged (used for totals)
            f"{attr} [{bucket}]",
            PRED_COLUMN,
            GT_COLUMN,
            out_base / bucket,
            model_name="VLM vs DINOv3",
            asset_type="",
            limit=limit,
        )



# ---------------------------------------------------------------------------
# Matched per-attribute F1 on the SHARED assets (the rigorous comparison)
# ---------------------------------------------------------------------------
def compute_matched_f1(vlm_files, embed_dir, gt_dir, min_overlap=30):
    """For each attribute, compute weighted & macro F1 for BOTH models on the
    identical set of shared assets. Returns a tidy DataFrame.
    """
    from sklearn.metrics import f1_score

    records = []
    for attr, cfg in ATTRIBUTES.items():
        merged = build_comparison(attr, cfg, vlm_files, embed_dir, gt_dir)
        if merged is None or merged.empty:
            continue
        n = len(merged)
        y_true = merged["true_label"].astype(str)
        y_vlm = merged["vlm_pred"].astype(str)
        y_embed = merged["embed_pred"].astype(str)

        for model, y_pred in [("VLM (gemini)", y_vlm),
                              ("DINOv3 + LogReg", y_embed)]:
            records.append({
                "attribute": attr,
                "model": model,
                "n_shared": n,
                "weighted_f1": f1_score(y_true, y_pred, average="weighted",
                                        zero_division=0),
                "macro_f1": f1_score(y_true, y_pred, average="macro",
                                     zero_division=0),
                "low_overlap": n < min_overlap,
            })
    return pd.DataFrame(records)


def plot_matched_f1(df, metric, output_path, min_overlap=30):
    """Grouped bar chart of matched F1 per attribute, VLM vs DINOv3."""
    import matplotlib.pyplot as plt
    import numpy as np

    attrs = sorted(df["attribute"].unique())
    models = ["VLM (gemini)", "DINOv3 + LogReg"]
    colors = {"VLM (gemini)": "#1f4e79", "DINOv3 + LogReg": "#2e9e8f"}

    x = np.arange(len(attrs))
    width = 0.38
    fig, ax = plt.subplots(figsize=(15, 7))

    for i, model in enumerate(models):
        vals, ns = [], []
        for a in attrs:
            row = df[(df["attribute"] == a) & (df["model"] == model)]
            vals.append(row[metric].values[0] if len(row) else 0)
            ns.append(row["n_shared"].values[0] if len(row) else 0)
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label=model, color=colors[model])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    # annotate n_shared under each attribute (small)
    ns_by_attr = [df[df["attribute"] == a]["n_shared"].iloc[0] for a in attrs]
    labels = [a.replace("attr_", "").replace("_", "\n") for a in attrs]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns_by_attr)],
                       fontsize=11)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel(metric.replace("_", " "), fontsize=16)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"VLM vs. DINOv3 on SHARED assets \u2014 {metric}",
                 fontsize=19, pad=16, fontweight="semibold")
    ax.legend(title="model", fontsize=12, title_fontsize=14)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved matched plot: {output_path}")


def parse_args():
    """Parse command-line arguments for this script."""
    ap = argparse.ArgumentParser(description="Compare VLM vs DINOv3 per asset.")
    ap.add_argument("--results-root", default="results",
                    help="Root to search for VLM prediction CSVs.")
    ap.add_argument("--embed-dir",
                    default="results/dinov3_results/dinov3_logistic/predictions",
                    help="Folder with dinov3_<attr>_classification_predictions.csv")
    ap.add_argument("--train-dir", default="data/processed/train",
                    help="Folder with <attr>_train.csv (ground truth + image paths)")
    ap.add_argument("--attribute", default=None,
                    help="Attribute to render HTML for (omit to just print the table).")
    ap.add_argument("--buckets", nargs="+",
                    default=["both_right", "vlm_only", "embed_only", "both_wrong"],
                    help="Which buckets to render.")
    ap.add_argument("--output-dir", default="results/model_comparison",
                    help="Where to write HTML reports.")
    ap.add_argument("--limit", type=int, default=20,
                    help="Max assets per bucket in HTML.")
    ap.add_argument("--matched-plot", action="store_true",
                    help="Compute matched per-attribute F1 on shared assets, "
                         "save CSV + bar plots, then exit.")
    ap.add_argument("--matched-out", default="results/model_comparison/matched",
                    help="Output dir for matched F1 CSV + plots.")
    ap.add_argument("--min-overlap", type=int, default=30,
                    help="Flag attributes with fewer shared assets as low-overlap.")
    return ap.parse_args()


def main():
    """Run the script from parsed command-line arguments."""
    args = parse_args()
    vlm_files = find_vlm_files(args.results_root)
    if not vlm_files:
        print(f"No VLM prediction CSVs found under '{args.results_root}'. "
              f"Searched for patterns: {list(ASSET_TYPE_GLOB.values())}",
              file=sys.stderr)
        return 1

    if args.matched_plot:
        from pathlib import Path as _P
        out = _P(args.matched_out)
        out.mkdir(parents=True, exist_ok=True)
        df = compute_matched_f1(vlm_files, args.embed_dir, args.train_dir,
                                args.min_overlap)
        if df.empty:
            print("No shared assets found for any attribute.", file=sys.stderr)
            return 1
        csv_path = out / "matched_f1_shared_assets.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved matched F1 table: {csv_path}\n")
        # quick console summary
        wide = df.pivot(index="attribute", columns="model", values="weighted_f1")
        print("Weighted F1 on shared assets (per attribute):")
        print(wide.round(3).to_string())
        for metric in ("weighted_f1", "macro_f1"):
            plot_matched_f1(df, metric, out / f"matched_{metric}.png",
                            args.min_overlap)
        return 0

    if args.attribute is None:
        print_overview(vlm_files, args.embed_dir, args.train_dir)
        return 0

    if args.attribute not in ATTRIBUTES:
        print(f"Unknown attribute '{args.attribute}'. "
              f"Choices: {list(ATTRIBUTES)}", file=sys.stderr)
        return 1

    generate_html(
        args.attribute, ATTRIBUTES[args.attribute], vlm_files,
        args.embed_dir, args.train_dir, args.train_dir,
        args.buckets, args.output_dir, args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
