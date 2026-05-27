"""Assemble a single clean image set ready for cloud-VLM upload.

For each of the 5,310 images in ``data/predictions/pii_screen.csv``:
- If flagged → copy the **blurred** version from ``data/pii_review/blurred/``.
- If not flagged → copy the **original** from ``data/raw/...``.

Output goes to ``data/processed/images_clean/`` mirroring the source path layout.
That directory is the only thing we ship to OpenAI / Claude / Gemini / Grok.

Pre-requisites:
- ``scripts/screen_images_for_pii.py`` was run (creates pii_screen.csv).
- ``scripts/blur_flagged_images.py`` was run (creates pii_review/blurred/).

Usage:
    python scripts/build_upload_set.py
    python scripts/build_upload_set.py --max-images 50    # smoke
    python scripts/build_upload_set.py --dry-run          # counts only
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)


def _rel_under_raw(image_path: str) -> Path:
    """Map `data/raw/citywide/...` → `citywide/...` (drop data/raw/ prefix)."""
    parts = Path(image_path).parts
    if parts[:2] == ("data", "raw"):
        return Path(*parts[2:])
    return Path(image_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen-csv", type=Path, default=Path("data/predictions/pii_screen.csv"))
    p.add_argument("--blurred-root", type=Path, default=Path("data/pii_review/blurred"))
    p.add_argument("--source-root", type=Path, default=Path("data/raw"))
    p.add_argument("--output-root", type=Path, default=Path("data/processed/images_clean"))
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Report counts but don't copy.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if not args.screen_csv.exists():
        raise SystemExit(
            f"{args.screen_csv} not found. Run scripts/screen_images_for_pii.py first."
        )
    df = pd.read_csv(args.screen_csv)
    if args.max_images is not None:
        df = df.head(args.max_images)
    logger.info(
        "Loaded %d rows from %s (%d flagged)",
        len(df),
        args.screen_csv,
        int(df["flagged"].sum()),
    )

    n_copied_original = 0
    n_copied_blurred = 0
    n_missing_source = 0
    n_missing_blurred = 0

    for _, row in df.iterrows():
        rel_raw = _rel_under_raw(row["image_path"])  # citywide/images/.../X.jpeg
        dst = args.output_root / rel_raw

        if row["flagged"]:
            src = args.blurred_root / rel_raw
            kind = "blurred"
        else:
            src = args.source_root / rel_raw
            kind = "original"

        if not src.exists():
            if kind == "blurred":
                n_missing_blurred += 1
                logger.warning("missing blurred for flagged image: %s", src)
            else:
                n_missing_source += 1
                logger.warning("missing source: %s", src)
            continue

        if args.dry_run:
            (n_copied_blurred if kind == "blurred" else n_copied_original)
            if kind == "blurred":
                n_copied_blurred += 1
            else:
                n_copied_original += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if kind == "blurred":
            n_copied_blurred += 1
        else:
            n_copied_original += 1

    print(
        f"\nUpload set summary"
        f"\n------------------"
        f"\n  originals copied : {n_copied_original}"
        f"\n  blurred copied   : {n_copied_blurred}"
        f"\n  missing source   : {n_missing_source}"
        f"\n  missing blurred  : {n_missing_blurred}"
        f"\n  total written    : {n_copied_original + n_copied_blurred}"
        f"\n  output root      : {args.output_root}"
        f"\n  dry-run          : {args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
