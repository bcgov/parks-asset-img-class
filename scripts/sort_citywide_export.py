"""Sort a flat CityWide image export into per-asset folders.

BCPark's CityWide bulk export is a flat folder of images plus a CSV that maps
each image's file name to its asset id (the "Source Page Link" column). This
script reorganises the flat folder into the per-asset layout the rest of the
pipeline expects:

    <output-dir>/<asset_id>/<file_name>

Multiple images belonging to the same asset are placed in the same asset folder,
matching how the CityWide API download is structured. The sorted folder is the
raw input to PII screening/blurring; the cleaned images then feed DINOv3 (or the
optional VLM path).

The export's header row is auto-detected (the CSV often has a title row or two
before the real header), by locating the row that contains the expected column
names. Images referenced in the CSV but missing from the folder, and images in
the folder not referenced by the CSV, are skipped and reported rather than
causing a hard failure.

Usage:
    python scripts/sort_citywide_export.py \
        --input-folder data/raw/hilary_export/images \
        --mapping-csv data/raw/hilary_export/export.csv \
        --output-dir data/raw/citywide/images

    # move instead of copy (leaves no duplicate of the original export)
    python scripts/sort_citywide_export.py ... --move
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

FILENAME_COLUMN = "File Name"
ASSET_ID_COLUMN = "Source Page Link"
# Columns we expect on the real header row (used to auto-detect it).
EXPECTED_HEADER_TOKENS = {FILENAME_COLUMN, ASSET_ID_COLUMN}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def detect_header_row(csv_path: Path, max_scan: int = 20) -> int:
    """Return the 0-based index of the row that holds the real header.

    Scans the first ``max_scan`` rows for one containing the expected column
    names, so leading title/blank rows are handled automatically.
    """
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for index, row in enumerate(reader):
            cells = {cell.strip() for cell in row}
            if EXPECTED_HEADER_TOKENS.issubset(cells):
                return index
            if index >= max_scan:
                break
    raise ValueError(
        f"Could not find a header row containing {sorted(EXPECTED_HEADER_TOKENS)} "
        f"in the first {max_scan} rows of {csv_path}. Check the export format."
    )


def load_mapping(csv_path: Path) -> list[tuple[str, str]]:
    """Return a list of (file_name, asset_id) pairs from the export CSV."""
    header_row = detect_header_row(csv_path)
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for _ in range(header_row):
            next(reader, None)
        header = [cell.strip() for cell in next(reader)]
        try:
            name_idx = header.index(FILENAME_COLUMN)
            asset_idx = header.index(ASSET_ID_COLUMN)
        except ValueError as exc:
            raise ValueError(
                f"Header row is missing an expected column: {exc}. "
                f"Found columns: {header}"
            ) from exc

        pairs: list[tuple[str, str]] = []
        for row in reader:
            if len(row) <= max(name_idx, asset_idx):
                continue
            file_name = row[name_idx].strip()
            asset_id = row[asset_idx].strip()
            if file_name and asset_id:
                pairs.append((file_name, asset_id))
    return pairs


def index_folder(input_folder: Path) -> dict[str, Path]:
    """Map available image file names to their paths (exact and stem keys)."""
    index: dict[str, Path] = {}
    for path in input_folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.name, path)             # exact name
            index.setdefault(path.name.lower(), path)     # case-insensitive
            index.setdefault(path.stem.lower(), path)     # stem (no extension)
    return index


def resolve_image(file_name: str, folder_index: dict[str, Path]) -> Path | None:
    """Find the actual image file for a CSV file name, tolerating small diffs."""
    for key in (file_name, file_name.lower(), Path(file_name).stem.lower()):
        if key in folder_index:
            return folder_index[key]
    return None


def safe_asset_dirname(asset_id: str) -> str:
    """Sanitise an asset id for use as a folder name."""
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in asset_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-folder", type=Path, required=True,
                        help="Flat folder of exported images.")
    parser.add_argument("--mapping-csv", type=Path, required=True,
                        help="CityWide export CSV mapping File Name to Source Page Link (asset id).")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/raw/citywide/images"),
                        help="Destination root; images land in <output-dir>/<asset_id>/.")
    parser.add_argument("--move", action="store_true",
                        help="Move files instead of copying (default: copy).")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional path to write a CSV log of the sort result.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input_folder.exists():
        print(f"Input folder not found: {args.input_folder}", file=sys.stderr)
        return 1
    if not args.mapping_csv.exists():
        print(f"Mapping CSV not found: {args.mapping_csv}", file=sys.stderr)
        return 1

    pairs = load_mapping(args.mapping_csv)
    if not pairs:
        print(f"No (file name, asset id) rows found in {args.mapping_csv}.", file=sys.stderr)
        return 1

    folder_index = index_folder(args.input_folder)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sorted_count = 0
    missing: list[tuple[str, str]] = []     # in CSV, no file found
    matched_files: set[Path] = set()
    report_rows: list[dict[str, str]] = []

    for file_name, asset_id in pairs:
        source = resolve_image(file_name, folder_index)
        if source is None:
            missing.append((file_name, asset_id))
            report_rows.append({
                "file_name": file_name, "asset_id": asset_id,
                "status": "missing_file", "destination": "",
            })
            continue

        asset_dir = args.output_dir / safe_asset_dirname(asset_id)
        asset_dir.mkdir(parents=True, exist_ok=True)
        destination = asset_dir / source.name

        if destination.exists():
            report_rows.append({
                "file_name": file_name, "asset_id": asset_id,
                "status": "already_present", "destination": str(destination),
            })
            matched_files.add(source)
            sorted_count += 1
            continue

        if args.move:
            shutil.move(str(source), str(destination))
        else:
            shutil.copy2(source, destination)
        matched_files.add(source)
        sorted_count += 1
        report_rows.append({
            "file_name": file_name, "asset_id": asset_id,
            "status": "moved" if args.move else "copied",
            "destination": str(destination),
        })

    # Images present in the folder but never referenced by the CSV.
    all_images = {p for p in folder_index.values()}
    orphans = sorted(all_images - matched_files)
    for orphan in orphans:
        report_rows.append({
            "file_name": orphan.name, "asset_id": "",
            "status": "orphan_not_in_csv", "destination": "",
        })

    n_assets = len({safe_asset_dirname(a) for _, a in pairs})
    print("CityWide export sort complete.")
    print(f"  images sorted:        {sorted_count}")
    print(f"  unique assets:        {n_assets}")
    print(f"  CSV rows missing file: {len(missing)}")
    print(f"  folder orphans:       {len(orphans)}")
    print(f"  output:               {args.output_dir}")
    if missing:
        print("  (missing files are listed in the report; they were skipped)")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["file_name", "asset_id", "status", "destination"]
            )
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"  report written to:    {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())