"""Shared label-cleaning helpers used by every pipeline.

Every pipeline (baseline, CLIP zero-shot, DINOv3 head, k-NN, VLM, YOLO, …)
historically defined its own ``_clean_labels`` + ``_MISSING_TOKENS`` copy
to drop blank / TBD / Unknown / NaN values before fitting or scoring.
This module is the single source of truth so the de-duplication policy is
uniform across the leaderboard.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

MISSING_TOKENS: Final[frozenset[str]] = frozenset(
    {"", "nan", "none", "null", "tbd", "unknown"}
)
"""Lower-cased tokens treated as missing labels by every pipeline.

Anything else (including ``"N/A"`` written in mixed case, which lower-cases
to ``"n/a"``) is left in place — callers should normalise their own domain
vocabulary via the schema alias table before invoking these helpers.
"""


def is_missing(value: object) -> bool:
    """Return ``True`` if ``value`` is one of the canonical missing tokens."""

    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in MISSING_TOKENS


def clean_labels(series: pd.Series) -> pd.Series:
    """Strip whitespace and drop missing tokens from a label column.

    Returns a copy with ``string`` dtype that preserves the original index so
    callers can re-align with feature matrices via ``loc``.
    """

    out = series.astype("string").str.strip()
    mask = out.notna() & ~out.str.lower().isin(MISSING_TOKENS)
    return out.loc[mask]


__all__ = ["MISSING_TOKENS", "clean_labels", "is_missing"]
