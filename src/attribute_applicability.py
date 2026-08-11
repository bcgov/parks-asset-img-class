"""Helpers for BC Parks asset-attribute applicability matrices."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


NON_ASSET_COLUMNS = {"Attribute", "Want AI to Determine", "Weight Priority"}


def _wants_ai(row: pd.Series) -> bool:
    """Return whether an applicability value means the attribute should be predicted."""
    if "Want AI to Determine" not in row:
        return True
    return str(row["Want AI to Determine"]).strip().lower() == "yes"


def load_applicability(path: Path) -> dict[str, set[str]]:
    """Return {asset_type: set(internal_target_names)} from the matrix CSV."""
    matrix = pd.read_csv(path)
    if "Attribute" not in matrix.columns:
        raise ValueError(
            f"Applicability CSV must contain an 'Attribute' column. "
            f"Got {matrix.columns.tolist()}."
        )

    asset_type_columns = [
        column for column in matrix.columns if column not in NON_ASSET_COLUMNS
    ]
    if not asset_type_columns:
        raise ValueError(
            "Applicability CSV must contain one or more asset-type columns."
        )

    applicable: dict[str, set[str]] = {
        asset_type: set() for asset_type in asset_type_columns
    }
    for _, row in matrix.iterrows():
        if not _wants_ai(row):
            continue
        target = str(row["Attribute"]).strip()
        for asset_type in asset_type_columns:
            cell = row[asset_type]
            if pd.notna(cell) and str(cell).strip():
                applicable[asset_type].add(target)
    return applicable


def applicable_profiles_for_target(
    applicability: dict[str, set[str]],
    target: str,
) -> list[str]:
    """Return asset types where a target should be predicted."""
    return sorted(
        asset_type
        for asset_type, targets in applicability.items()
        if target in targets
    )
