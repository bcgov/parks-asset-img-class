"""Download CityWide asset records, attributes, file metadata, and images.

This imports the focused CityWide downloader from the companion
``bcpark-api`` project into the capstone repo. It defaults to the raw-data
layout used by the modelling pipeline:

``data/raw/citywide/images/<profile_id>/<asset_id>/<file_id>__<filename>``

The run is resume-safe:
- profile-level metadata snapshots are cached under ``by_profile/``
- existing non-empty image files are skipped
- ``images_manifest.csv`` records per-file status
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.citywide_client import CitywideClient  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("data/raw/citywide")

PROFILES: dict[int, str] = {
    337: "Boardwalk < 1.2m High",
    573: "Boardwalk > 1.2m High",
    356: "Stairs",
    253: "Trail Bridge",
    359: "Viewing Platform",
}


def selected_profiles(profile_ids: list[int] | None) -> dict[int, str]:
    """Return requested CityWide profile ids and names."""
    if not profile_ids:
        return dict(PROFILES)
    invalid = [profile_id for profile_id in profile_ids if profile_id not in PROFILES]
    if invalid:
        raise ValueError(f"Unknown CityWide profile id(s): {invalid}. Known: {sorted(PROFILES)}")
    return {profile_id: PROFILES[profile_id] for profile_id in profile_ids}


def _profile_dir(output_dir: Path, profile_id: int) -> Path:
    return output_dir / "by_profile" / str(profile_id)


def _save_profile(
    output_dir: Path,
    profile_id: int,
    assets: list[dict],
    attrs: list[dict],
    files: list[dict],
) -> None:
    """Persist per-profile metadata so partial runs are not lost."""
    profile_dir = _profile_dir(output_dir, profile_id)
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "assets.json").write_text(json.dumps(assets, indent=2, default=str))
    (profile_dir / "attributes.json").write_text(json.dumps(attrs, indent=2, default=str))
    (profile_dir / "files.json").write_text(json.dumps(files, indent=2, default=str))


def fetch_metadata(
    client: CitywideClient,
    output_dir: Path,
    profiles: dict[int, str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Fetch assets, attributes, and attached-file metadata for target profiles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    assets: list[dict] = []
    attributes: list[dict] = []
    files: list[dict] = []

    for profile_id, profile_name in profiles.items():
        profile_dir = _profile_dir(output_dir, profile_id)
        if (profile_dir / "files.json").exists():
            profile_assets = json.loads((profile_dir / "assets.json").read_text())
            profile_attrs = json.loads((profile_dir / "attributes.json").read_text())
            profile_files = json.loads((profile_dir / "files.json").read_text())
            assets.extend(profile_assets)
            attributes.extend(profile_attrs)
            files.extend(profile_files)
            print(
                f"\n[{profile_id}] {profile_name} "
                f"(cached: {len(profile_assets)} assets)",
                flush=True,
            )
            continue

        print(f"\n[{profile_id}] {profile_name}", flush=True)
        profile_assets: list[dict] = []
        profile_attrs: list[dict] = []
        profile_files: list[dict] = []

        for asset in client.list_all(
            "/bulk/assets",
            {"profile_id": profile_id, "$linked": "Attributes"},
        ):
            asset_id = asset["id"]
            asset["profile_id_used"] = profile_id
            asset["profile_name"] = profile_name
            linked = asset.pop("linked", {}) or {}
            for attribute in linked.get("Attributes") or []:
                attribute["asset_id"] = asset_id
                attribute["profile_id"] = profile_id
                profile_attrs.append(attribute)
            profile_assets.append(asset)

        print(
            f"  fetched {len(profile_assets)} assets, {len(profile_attrs)} attrs; "
            "fetching attached_files metadata...",
            flush=True,
        )
        for index, asset in enumerate(profile_assets, start=1):
            response = client.get(f"/assets/{asset['id']}/attached_files")
            if response.status_code == 200:
                for file_record in response.json():
                    file_record["asset_id"] = asset["id"]
                    file_record["profile_id"] = profile_id
                    profile_files.append(file_record)
            if index % 100 == 0 or index == len(profile_assets):
                print(
                    f"    files: {index}/{len(profile_assets)} "
                    f"total_files={len(profile_files)}",
                    flush=True,
                )

        _save_profile(output_dir, profile_id, profile_assets, profile_attrs, profile_files)
        assets.extend(profile_assets)
        attributes.extend(profile_attrs)
        files.extend(profile_files)
        print(
            f"  done: {len(profile_assets)} assets, {len(profile_attrs)} attrs, "
            f"{len(profile_files)} files -> {_profile_dir(output_dir, profile_id)}",
            flush=True,
        )

    return assets, attributes, files


def save_metadata(
    output_dir: Path,
    assets: list[dict],
    attributes: list[dict],
    files: list[dict],
) -> None:
    """Write consolidated metadata JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("assets", assets),
        ("attributes", attributes),
        ("files_manifest", files),
    ]:
        (output_dir / f"{name}.json").write_text(json.dumps(rows, indent=2, default=str))
        if rows:
            pd.json_normalize(rows).to_csv(output_dir / f"{name}.csv", index=False)

    print(f"\nMetadata saved to {output_dir}/")
    print(f"  assets.csv         {len(assets):>5} rows")
    print(f"  attributes.csv     {len(attributes):>5} rows")
    print(f"  files_manifest.csv {len(files):>5} rows")


def load_cached_files(output_dir: Path, profiles: dict[int, str]) -> list[dict]:
    """Load cached per-profile attached-file metadata."""
    files: list[dict] = []
    for profile_id in profiles:
        profile_dir = _profile_dir(output_dir, profile_id)
        if (profile_dir / "files.json").exists():
            files.extend(json.loads((profile_dir / "files.json").read_text()))
    print(f"Loaded {len(files)} file metadata rows from cache")
    return files


def consolidate_metadata(output_dir: Path, profiles: dict[int, str]) -> list[dict]:
    """Read cached per-profile snapshots and write consolidated metadata files."""
    assets: list[dict] = []
    attributes: list[dict] = []
    files: list[dict] = []
    for profile_id in profiles:
        profile_dir = _profile_dir(output_dir, profile_id)
        if not (profile_dir / "files.json").exists():
            continue
        assets.extend(json.loads((profile_dir / "assets.json").read_text()))
        attributes.extend(json.loads((profile_dir / "attributes.json").read_text()))
        files.extend(json.loads((profile_dir / "files.json").read_text()))
    if assets:
        save_metadata(output_dir, assets, attributes, files)
    return files


def safe_filename(name: str) -> str:
    """Return a filesystem-safe filename with the original extension retained."""
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)[:120]


def image_destination(output_dir: Path, file_record: dict) -> Path:
    """Return the output path for one CityWide attached file."""
    asset_id = file_record["asset_id"]
    file_id = file_record["id"]
    profile_id = file_record["profile_id"]
    filename = safe_filename(file_record.get("filename") or f"file_{file_id}")
    return output_dir / "images" / str(profile_id) / str(asset_id) / f"{file_id}__{filename}"


def download_one(
    client: CitywideClient,
    output_dir: Path,
    file_record: dict,
) -> tuple[dict, str, int, Path]:
    """Download one attached file and return status metadata."""
    asset_id = file_record["asset_id"]
    file_id = file_record["id"]
    destination = image_destination(output_dir, file_record)
    if destination.exists() and destination.stat().st_size > 0:
        return file_record, "skip", destination.stat().st_size, destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = client.get_binary(f"/assets/{asset_id}/attached_files/{file_id}/content")
        if response.status_code != 200:
            return file_record, f"err:HTTP{response.status_code}", 0, destination
        destination.write_bytes(response.content)
        return file_record, "ok", len(response.content), destination
    except Exception as exc:
        return file_record, f"err:{type(exc).__name__}", 0, destination


def download_images(
    client: CitywideClient,
    output_dir: Path,
    files: list[dict],
    *,
    workers: int = 4,
    only_images: bool = True,
) -> None:
    """Download attached image binaries and write an image manifest."""
    targets = [
        file_record
        for file_record in files
        if (not only_images) or "image" in (file_record.get("mime_type") or "")
    ]
    print(f"\nDownloading {len(targets)} files (images_only={only_images}, workers={workers})...")

    rows: list[dict] = []
    ok = skip = err = 0
    bytes_total = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(download_one, client, output_dir, file_record)
            for file_record in targets
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            file_record, status, size, destination = future.result()
            if status == "ok":
                ok += 1
                bytes_total += size
            elif status == "skip":
                skip += 1
                bytes_total += size
            else:
                err += 1

            try:
                image_path = destination.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                image_path = destination.as_posix()
            rows.append(
                {
                    "asset_id": file_record["asset_id"],
                    "profile_id": file_record["profile_id"],
                    "file_id": file_record["id"],
                    "filename": file_record.get("filename"),
                    "mime_type": file_record.get("mime_type"),
                    "status": status,
                    "bytes": size,
                    "image_path": image_path,
                }
            )
            if index % 50 == 0 or index == len(futures):
                print(
                    f"  [{index:>5}/{len(futures)}] ok={ok} skip={skip} "
                    f"err={err} {bytes_total / 1e6:.0f} MB"
                )

    manifest_path = output_dir / "images_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"\nManifest -> {manifest_path}")


def probe_quota(client: CitywideClient) -> None:
    """Spend one API call to print rate-limit headers."""
    response = client.get("/users", params={"$limit": 1})
    print(f"  status: {response.status_code}")
    for header in (
        "X-Total",
        "X-Rate-Limit-Limit",
        "X-Rate-Limit-Remaining",
        "X-Rate-Limit-Reset",
        "Retry-After",
    ):
        value = response.headers.get(header)
        if value:
            print(f"  {header}: {value}")

    print("\n  all X-/Retry- headers:")
    for key, value in response.headers.items():
        if key.lower().startswith(("x-", "retry-")):
            print(f"    {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Raw CityWide output directory.",
    )
    parser.add_argument(
        "--profile",
        type=int,
        action="append",
        default=None,
        help=(
            "Restrict to one profile id. May be repeated. Known profiles: "
            + ", ".join(f"{key}={value}" for key, value in PROFILES.items())
        ),
    )
    parser.add_argument("--metadata-only", action="store_true", help="Skip image download.")
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Skip metadata fetch and use cached by_profile JSON files.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only download first N files after metadata filtering.",
    )
    parser.add_argument(
        "--max-calls-per-hour",
        type=int,
        default=900,
        help="Client-side rate limit. CityWide server cap is 1000/hr.",
    )
    parser.add_argument("--probe", action="store_true", help="Print rate-limit headers and exit.")
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download all attachments, not only image MIME types.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    profiles = selected_profiles(args.profile)
    output_dir.mkdir(parents=True, exist_ok=True)

    with CitywideClient(max_calls_per_hour=args.max_calls_per_hour) as client:
        if args.probe:
            probe_quota(client)
            return 0

        if args.images_only:
            files = load_cached_files(output_dir, profiles)
        else:
            _assets, _attrs, files = fetch_metadata(client, output_dir, profiles)

        files = consolidate_metadata(output_dir, profiles) or files

        if args.metadata_only:
            return 0

        if args.limit:
            files = files[: args.limit]
        download_images(
            client,
            output_dir,
            files,
            workers=args.workers,
            only_images=not args.all_files,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
