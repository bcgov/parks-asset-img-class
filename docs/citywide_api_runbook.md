# CityWide API Runbook

This guide explains how the project downloads CityWide asset metadata,
attributes, attached-file metadata, and image files for the BC Parks image
classification pipeline.

The CityWide download is optional. It is intended for BC Parks users or graders
who have CityWide API credentials. The modelling pipeline can run from the
tracked processed data without re-downloading CityWide data.

## Required Credentials

Create a local `.env` file from `.env.example` and set:

```bash
CITYWIDE_API_KEY=
CITYWIDE_DB=
CITYWIDE_USER=
CITYWIDE_API_URL=https://v4.citywidesolutions.com/v4_server/external/v1
```

`CITYWIDE_API_URL` is optional if the default CityWide URL is correct.

Check credentials before downloading:

```bash
make citywide-check
```

## Download Commands

Download metadata and attributes only:

```bash
make download-citywide-metadata
```

Download metadata, attributes, attached-file metadata, and image files:

```bash
make download-citywide-images
```

Run a small 50-file download smoke test:

```bash
make download-citywide-sample
```

Restrict the download to selected profile IDs:

```bash
make download-citywide-images CITYWIDE_PROFILE="337 356"
```

Limit downloaded files:

```bash
make download-citywide-images CITYWIDE_LIMIT=100
```

## Asset Type Mapping

CityWide asset types are selected by profile ID in
`scripts/download_citywide_images.py`:

```text
337 -> Boardwalk < 1.2m High
573 -> Boardwalk > 1.2m High
356 -> Stairs
253 -> Trail Bridge
359 -> Viewing Platform
```

The downloader writes these values into the metadata as `profile_id` and
`profile_name`. Later pipeline steps use `profile_name` as the asset type.

## How Attributes Are Downloaded

Attributes are downloaded as linked records on the asset request. They are not
downloaded from a separate source CSV.

For each selected profile, the downloader calls the CityWide bulk asset
endpoint with linked attributes:

```text
GET /bulk/assets?profile_id=<profile_id>&$linked=Attributes
```

In code, this is:

```python
for asset in client.list_all(
    "/bulk/assets",
    {"profile_id": profile_id, "$linked": "Attributes"},
):
```

The response includes the asset record plus linked `Attributes`. The downloader
extracts those linked attributes and adds the asset/profile identifiers:

```python
linked = asset.pop("linked", {}) or {}
for attribute in linked.get("Attributes") or []:
    attribute["asset_id"] = asset_id
    attribute["profile_id"] = profile_id
```

## Output Files

The consolidated metadata is written under:

```text
data/raw/citywide/
```

Important outputs:

```text
data/raw/citywide/assets.csv
data/raw/citywide/attributes.csv
data/raw/citywide/files_manifest.csv
data/raw/citywide/images_manifest.csv
```

Downloaded image files are written under:

```text
data/raw/citywide/images/<profile_id>/<asset_id>/<file_id>__<filename>
```

Per-profile cached API snapshots are written under:

```text
data/raw/citywide/by_profile/<profile_id>/assets.json
data/raw/citywide/by_profile/<profile_id>/attributes.json
data/raw/citywide/by_profile/<profile_id>/files.json
```

The cache makes the downloader resume-safe. If the per-profile JSON files
already exist, the downloader reuses them instead of fetching the same metadata
again.

## How Images Are Downloaded

After asset and attribute metadata are fetched, the downloader calls the
attached-file endpoint for each asset:

```text
GET /assets/<asset_id>/attached_files
```

For image binaries, it then calls:

```text
GET /assets/<asset_id>/attached_files/<file_id>/content
```

By default, only attachments with an image MIME type are downloaded. Use the
script-level `--all-files` option if every attachment type is needed.

## How This Connects To The Model Pipeline

The model pipeline does not train directly from `data/raw/citywide/`. The raw
download is an upstream data source. The current final pipeline reads processed
inputs such as:

```text
data/processed/master_dataset.csv
data/processed/train/
data/processed/attribute_applicability.csv
```

The final image pipeline reads cleaned local image files from:

```text
data/processed/images_clean/
```

Run this once to build the cleaned image set:

```bash
make pii
```

Then run the final pipeline:

```bash
make all
```
