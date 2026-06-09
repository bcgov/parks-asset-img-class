"""Helpers for visualizing and analyzing VLM prediction errors."""

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from src.dinov3_features import resolve_image_path
except ImportError:
    resolve_image_path = None


# ---------------------------------------------------------------------------
# Attribute name normalization
# ---------------------------------------------------------------------------

def normalize_attribute_name(attr: str) -> str:
    """Normalize attribute names for file path / column matching.

    Strips the leading ``attr_`` prefix, removes punctuation that is illegal
    in file-names, and collapses runs of underscores so that
    ``attr_material_frame,_tank,_body`` → ``material_frame_tank_body``.
    """
    name = attr
    name = re.sub(r"^attr_", "", name)        # strip leading attr_
    name = name.replace(",", "")               # remove commas
    name = name.replace(" ", "_")
    name = name.replace("(", "").replace(")", "")
    name = name.replace("<", "lt").replace(">", "gt")
    name = name.replace("/", "_")
    name = re.sub(r"_+", "_", name)           # collapse repeated underscores
    name = name.strip("_")
    return name


def _candidate_normalized_names(attribute: str) -> list[str]:
    """Return several normalized forms of *attribute* to try during lookup."""
    norm = normalize_attribute_name(attribute)
    candidates = [norm]
    # also try with the attr_ prefix re-added, in case the file kept it
    if not norm.startswith("attr_"):
        candidates.append(f"attr_{norm}")
    # original attribute string (might already be the exact column name)
    if attribute not in candidates:
        candidates.append(attribute)
    return candidates


# ---------------------------------------------------------------------------
# Ground-truth file discovery
# ---------------------------------------------------------------------------

def find_ground_truth_file(
    attribute: str,
    ground_truth_dir: str = "data/processed/train",
) -> Optional[Path]:
    """Locate the ground-truth CSV for *attribute*.

    Mirrors the naming logic in ``evaluate_predictions.py``:

    * ``attr_structure_position`` → ``{dir}/attr_structure_position_train.csv``
    * ``steps_bin``               → ``{dir}/steps_bin_train.csv``

    Searches *ground_truth_dir* first, then any sibling split directory
    (train ↔ test), trying several filename variants so that callers
    passing the wrong split still find the file.
    """
    base_dir = Path(ground_truth_dir)

    # Build ordered list of directories to search (prefer the given dir)
    search_dirs: list[Path] = [base_dir]

    # Reproduce evaluate_predictions.py filename construction exactly:
    #   normalised = attribute with special chars stripped, attr_ prefix kept
    #   for attr_* attributes  → attr_{normalised}_train.csv
    #   for everything else    → {normalised}_train.csv
    norm = normalize_attribute_name(attribute)  # strips attr_ prefix
    if attribute.startswith("attr_"):
        primary_stem = f"attr_{norm}"           # re-add prefix, as evaluate_predictions does
    else:
        primary_stem = norm

    # Additional stems to try as fallbacks (cover both with- and without-prefix)
    extra_stems = [s for s in _candidate_normalized_names(attribute) if s != primary_stem]

    for search_dir in search_dirs:
        split_tag = search_dir.name  # "train" or "test"
        for stem in [primary_stem] + extra_stems:
            for fname in [
                f"{stem}_{split_tag}.csv",   # e.g. attr_structure_position_train.csv
                f"{stem}.csv",               # bare, no split tag
            ]:
                p = search_dir / fname
                if p.exists():
                    return p

    return None


# ---------------------------------------------------------------------------
# Prediction column discovery  (THE MAIN BUG FIX)
# ---------------------------------------------------------------------------

def get_prediction_column(
    attribute: str,
    predictions_df: pd.DataFrame,
) -> Optional[str]:
    """Find the prediction value column for *attribute* in *predictions_df*.

    Tries several naming strategies so that e.g. ``attr_structure_position``
    resolves to ``structure_position_value`` even when the column is named
    differently.
    """
    columns = predictions_df.columns.tolist()

    for candidate_name in _candidate_normalized_names(attribute):
        # Most common pattern: {normalized}_value
        for col in [f"{candidate_name}_value", candidate_name]:
            if col in columns:
                return col

    # Last-resort: scan all columns for a partial match on the normalised stem
    norm = normalize_attribute_name(attribute)
    for col in columns:
        col_norm = normalize_attribute_name(col)
        if col_norm == norm or col_norm == f"{norm}_value":
            return col

    return None


def get_confidence_column(
    attribute: str,
    predictions_df: pd.DataFrame,
) -> Optional[str]:
    """Find the confidence column for *attribute*, if present."""
    for candidate_name in _candidate_normalized_names(attribute):
        col = f"{candidate_name}_confidence"
        if col in predictions_df.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# Ground-truth column discovery inside a DataFrame
# ---------------------------------------------------------------------------

def _find_gt_column(attribute: str, gt_df: pd.DataFrame) -> Optional[str]:
    """Return the column in *gt_df* that holds the ground-truth for *attribute*.

    Excludes confidence / metadata columns so we don't accidentally use those.
    """
    EXCLUDE_SUFFIXES = ("_confidence", "_path", "_id", "_url", "_file", "_name")
    columns = gt_df.columns.tolist()

    for candidate in _candidate_normalized_names(attribute):
        if candidate in columns:
            return candidate

    # Fuzzy: normalise every column and compare
    norm_target = normalize_attribute_name(attribute)
    for col in columns:
        if any(col.endswith(s) for s in EXCLUDE_SUFFIXES):
            continue
        if normalize_attribute_name(col) == norm_target:
            return col

    return None


# ---------------------------------------------------------------------------
# Core load + merge
# ---------------------------------------------------------------------------

def load_predictions_and_ground_truth(
    predictions_csv: str,
    attribute: str,
    ground_truth_dir: str = "data/processed/train",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Load predictions and ground truth and inner-merge on ``asset_id``.

    Returns
    -------
    merged_df, predictions_df, pred_column, gt_column
    """
    predictions_df = pd.read_csv(predictions_csv)
    if verbose:
        print(f"[load] Predictions: {len(predictions_df)} rows, "
              f"{len(predictions_df.columns)} columns")
        print(f"[load] Prediction columns: {predictions_df.columns.tolist()}")

    # --- locate prediction column -------------------------------------------
    pred_col = get_prediction_column(attribute, predictions_df)
    if pred_col is None:
        raise ValueError(
            f"Could not find a prediction column for attribute '{attribute}'.\n"
            f"Available columns: {predictions_df.columns.tolist()}\n"
            f"Tried normalized forms: {_candidate_normalized_names(attribute)}"
        )
    if verbose:
        print(f"[load] Using prediction column: '{pred_col}'")

    # --- locate ground-truth CSV --------------------------------------------
    gt_path = find_ground_truth_file(attribute, ground_truth_dir)
    if gt_path is None:
        raise ValueError(
            f"Could not find ground-truth CSV for attribute '{attribute}' "
            f"under '{ground_truth_dir}'.\n"
            f"Tried name variants: {_candidate_normalized_names(attribute)}"
        )
    if verbose:
        print(f"[load] Ground-truth file: {gt_path}")

    gt_df = pd.read_csv(gt_path)
    if verbose:
        print(f"[load] Ground-truth: {len(gt_df)} rows, "
              f"columns: {gt_df.columns.tolist()}")

    # --- locate ground-truth column -----------------------------------------
    gt_col = _find_gt_column(attribute, gt_df)
    if gt_col is None:
        raise ValueError(
            f"Could not find ground-truth column for attribute '{attribute}' "
            f"in {gt_path}.\n"
            f"Available columns: {gt_df.columns.tolist()}"
        )
    if verbose:
        print(f"[load] Using ground-truth column: '{gt_col}'")

    # --- normalise asset_id dtype so the join works -------------------------
    for df in (predictions_df, gt_df):
        df["asset_id"] = df["asset_id"].astype(str).str.strip()

    # --- merge --------------------------------------------------------------
    extra_gt_cols = ["asset_id", gt_col]
    for optional in ("image_path", "filename"):
        if optional in gt_df.columns:
            extra_gt_cols.append(optional)

    merged = predictions_df.merge(
        gt_df[extra_gt_cols],
        on="asset_id",
        how="inner",
    )
    if verbose:
        print(f"[load] After merge: {len(merged)} rows")

    pre_drop = len(merged)
    merged = merged.dropna(subset=[pred_col, gt_col])
    if verbose and len(merged) < pre_drop:
        print(f"[load] Dropped {pre_drop - len(merged)} rows with NaN in "
              f"'{pred_col}' or '{gt_col}'")
        print(f"[load] Rows available for comparison: {len(merged)}")

    return merged, predictions_df, pred_col, gt_col


# ---------------------------------------------------------------------------
# Wrong-prediction filtering  (BUG FIX: normalise before comparing)
# ---------------------------------------------------------------------------

def _coerce_to_str(series: pd.Series) -> pd.Series:
    """Stringify, lowercase, strip whitespace — for robust equality checks."""
    return series.astype(str).str.strip().str.lower()


def get_wrong_predictions(
    merged_df: pd.DataFrame,
    pred_column: str,
    gt_column: str,
    exclude_parse_errors: bool = True,
    normalize_for_compare: bool = True,
) -> pd.DataFrame:
    """Return rows where the prediction does not match the ground truth.

    Parameters
    ----------
    normalize_for_compare:
        If True (default) comparisons are case-insensitive and whitespace-
        insensitive.  The *values in the returned DataFrame are not changed*;
        only the filtering uses the normalised forms.
    """
    df = merged_df.copy()

    if exclude_parse_errors and "parse_error" in df.columns:
        df = df[df["parse_error"].ne(True)]

    if normalize_for_compare:
        pred_norm = _coerce_to_str(df[pred_column])
        gt_norm   = _coerce_to_str(df[gt_column])
        mask_wrong = pred_norm != gt_norm
    else:
        mask_wrong = df[pred_column] != df[gt_column]

    return df[mask_wrong].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def extract_response_value(response_json: str, attribute: str) -> Optional[dict]:
    """Extract value/confidence dict from a raw JSON response string."""
    if pd.isna(response_json):
        return None
    try:
        data = json.loads(response_json)
    except (json.JSONDecodeError, TypeError):
        return None

    for key in _candidate_normalized_names(attribute) + [attribute]:
        if key in data:
            return data[key]
    return None


def get_image_path(row: pd.Series, repo_root: Path = None) -> Optional[Path]:
    """Resolve the image path for a DataFrame row."""
    if "image_path" not in row or pd.isna(row["image_path"]):
        return None
    if resolve_image_path is None:
        return Path(row["image_path"])
    try:
        return resolve_image_path(row["image_path"], repo_root=repo_root)
    except Exception:
        return None