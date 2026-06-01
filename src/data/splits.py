"""Asset-level train/test loading and cross-validation.

Every pipeline reads splits through `load_split()` so they share exactly
the same view of the data and the same asset_id partition.

Source of truth (in priority order):
1. `data/processed/train.csv` + `data/processed/test.csv` if both exist
   (this is the consolidated split produced on branch VLM-exploration-2
   and merged into main; uses asset_id-disjoint partitioning).
2. Otherwise, fall back to an in-memory asset-level split of
   `data/processed/master_dataset.csv` (deterministic via `split_seed`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

DEFAULT_TEST_SIZE = 0.2
DEFAULT_SPLIT_SEED = 42
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


@dataclass(frozen=True)
class SplitPaths:
    """Resolved on-disk locations for the train/test CSVs."""

    train: Path | None
    test: Path | None
    master: Path

    def has_explicit_split(self) -> bool:
        return self.train is not None and self.test is not None


def resolve_split_paths(processed_dir: str | Path | None = None) -> SplitPaths:
    """Find the train/test/master CSVs."""
    base = Path(processed_dir) if processed_dir is not None else DEFAULT_PROCESSED_DIR
    train = base / "train.csv"
    test = base / "test.csv"
    master = base / "master_dataset.csv"
    return SplitPaths(
        train=train if train.exists() else None,
        test=test if test.exists() else None,
        master=master,
    )


def load_split(
    *,
    processed_dir: str | Path | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    split_seed: int = DEFAULT_SPLIT_SEED,
    drop_missing_files: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the project-wide ``(train_df, test_df)``.

    Parameters
    ----------
    processed_dir
        Directory containing ``train.csv`` / ``test.csv`` / ``master_dataset.csv``.
        Defaults to ``parks-asset-img-class/data/processed``.
    test_size, split_seed
        Used only as a fallback when ``train.csv`` / ``test.csv`` are not
        present and the master dataset has to be split on the fly.
    drop_missing_files
        When True, rows whose ``file_exists`` column is False are dropped
        (~250 of 5,562 rows in the current data drop).
    """
    paths = resolve_split_paths(processed_dir)

    if paths.has_explicit_split():
        train_df = pd.read_csv(paths.train)
        test_df = pd.read_csv(paths.test)
    else:
        if not paths.master.exists():
            raise FileNotFoundError(
                f"No split CSVs and master file missing: {paths.master}"
            )
        master = pd.read_csv(paths.master)
        train_df, test_df = _split_by_asset(
            master, test_size=test_size, split_seed=split_seed
        )

    if drop_missing_files and "file_exists" in train_df.columns:
        train_df = train_df[train_df["file_exists"].astype(bool)].reset_index(drop=True)
        test_df = test_df[test_df["file_exists"].astype(bool)].reset_index(drop=True)

    overlap = set(train_df["asset_id"]) & set(test_df["asset_id"])
    if overlap:
        raise ValueError(
            f"Train/test split is leaking: {len(overlap)} asset_ids appear in both."
        )

    return train_df, test_df


def _split_by_asset(
    df: pd.DataFrame, *, test_size: float, split_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset_ids = pd.Series(df["asset_id"].dropna().unique())
    if len(asset_ids) < 2:
        raise ValueError("Need at least two assets to create a train/test split.")
    train_assets, test_assets = train_test_split(
        asset_ids, test_size=test_size, random_state=split_seed
    )
    train_df = df[df["asset_id"].isin(train_assets)].reset_index(drop=True)
    test_df = df[df["asset_id"].isin(test_assets)].reset_index(drop=True)
    return train_df, test_df


def asset_grouped_kfold(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield (train_idx, val_idx) splits that never share an ``asset_id``.

    Used by every cross-validated head + by the stacking meta-learner so
    that out-of-fold predictions cannot leak across the same asset's
    images.
    """
    groups = df["asset_id"].values
    kf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in kf.split(df, groups=groups):
        yield df.index[train_idx], df.index[val_idx]


def absolute_image_path(
    image_path: str,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve the on-disk image path from a ``image_path`` cell.

    The CSV stores paths like ``data/citywide/images/337/48117/86997__file.jpeg``
    while the actual images live under ``data/raw/citywide/images/...``.
    """
    base = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    rel = str(image_path)
    if rel.startswith("data/"):
        rel = rel[len("data/"):]
    return base / "data" / "raw" / rel
