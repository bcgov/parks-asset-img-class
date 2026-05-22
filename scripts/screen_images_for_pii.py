"""Screen every image in data/raw/.../images for likely PII before any cloud upload.

Runs locally — no data leaves the machine. Uses YOLOv8 (general) to detect
person / vehicle bounding boxes. The "person" class is conservative — if a
human is visible at all (face, body, partial torso) the image is flagged.

Optional face detector: pass ``--face-model path/to/yolov8n-face.pt`` to
also flag close-up faces that the person detector might miss (e.g., faces
behind windshields). The face model is *not* auto-downloaded — get it from
`https://github.com/akanametov/yolov8-face` if you want it.

Usage:
    python scripts/screen_images_for_pii.py                       # all images
    python scripts/screen_images_for_pii.py --max-images 50       # smoke test
    python scripts/screen_images_for_pii.py --conf 0.3            # cautious

Output:
    data/predictions/pii_screen.csv         — one row per image
    reports/pii_screen_summary.md           — counts + flagged-image table
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("YOLO_VERBOSE", "False")

import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

# Classes in YOLOv8 (COCO) that indicate identifiable subjects.
PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

DEFAULT_GENERAL_MODEL = "yolov8n.pt"  # auto-downloaded by ultralytics

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def iter_image_paths(image_dir: Path) -> Iterable[Path]:
    for p in sorted(image_dir.rglob("*")):
        if p.suffix in IMAGE_SUFFIXES and p.is_file():
            yield p


def _select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _classes_above(results, class_names: set[str], conf: float) -> tuple[int, float]:
    """Return (count, max_conf) of detections matching any class name above conf."""
    if results.boxes is None or len(results.boxes) == 0:
        return 0, 0.0
    names = results.names
    confs = results.boxes.conf.cpu().tolist()
    cls_ids = results.boxes.cls.cpu().tolist()
    n = 0
    max_c = 0.0
    for cid, c in zip(cls_ids, confs, strict=False):
        if c < conf:
            continue
        if names[int(cid)] in class_names:
            n += 1
            if c > max_c:
                max_c = c
    return n, max_c


def _face_count(results, conf: float) -> tuple[int, float]:
    """yolov8n-face emits a single class (0=face), so we count above conf."""
    if results.boxes is None or len(results.boxes) == 0:
        return 0, 0.0
    confs = results.boxes.conf.cpu().tolist()
    above = [c for c in confs if c >= conf]
    return len(above), max(above, default=0.0)


def screen_images(
    image_paths: list[Path],
    *,
    general_model_id: str,
    face_model_id: str | None,
    conf: float,
    device: str,
    batch_size: int,
    repo_root: Path,
) -> pd.DataFrame:
    from ultralytics import YOLO

    logger.info("Loading YOLO general model %s on %s", general_model_id, device)
    general = YOLO(general_model_id)
    face = None
    if face_model_id:
        logger.info("Loading YOLO face model %s on %s", face_model_id, device)
        face = YOLO(face_model_id)

    rows: list[dict] = []

    def _batches(items: list[Path]) -> Iterable[list[Path]]:
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    for batch_idx, batch in enumerate(_batches(image_paths)):
        gen_results = general(
            [str(p) for p in batch], conf=conf, device=device, verbose=False
        )
        face_results = (
            face([str(p) for p in batch], conf=conf, device=device, verbose=False)
            if face is not None
            else [None] * len(batch)
        )
        for path, gr, fr in zip(batch, gen_results, face_results, strict=True):
            n_persons, persons_conf = _classes_above(gr, PERSON_CLASSES, conf)
            n_vehicles, vehicles_conf = _classes_above(gr, VEHICLE_CLASSES, conf)
            n_faces, faces_conf = (_face_count(fr, conf) if fr is not None else (0, 0.0))
            rel = str(path.relative_to(repo_root)) if path.is_absolute() else str(path)
            rows.append(
                {
                    "image_path": rel,
                    "n_persons": n_persons,
                    "persons_max_conf": round(persons_conf, 3),
                    "n_vehicles": n_vehicles,
                    "vehicles_max_conf": round(vehicles_conf, 3),
                    "n_faces": n_faces,
                    "faces_max_conf": round(faces_conf, 3),
                }
            )
        if (batch_idx + 1) % 20 == 0:
            logger.info("Screened %d / %d images", len(rows), len(image_paths))

    df = pd.DataFrame(rows)
    df["flagged_reason"] = df.apply(_flag_reason, axis=1)
    df["flagged"] = df["flagged_reason"].astype(bool)
    return df


def _flag_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if row["n_faces"] > 0:
        reasons.append(f"face×{row['n_faces']}")
    if row["n_persons"] > 0:
        reasons.append(f"person×{row['n_persons']}")
    if row["n_vehicles"] > 0:
        reasons.append(f"vehicle×{row['n_vehicles']}")
    return ", ".join(reasons)


def render_summary(
    df: pd.DataFrame, out_path: Path, *, conf: float, face_model_used: bool
) -> None:
    n_total = len(df)
    n_flagged = int(df["flagged"].sum())
    n_with_face = int((df["n_faces"] > 0).sum())
    n_with_person = int((df["n_persons"] > 0).sum())
    n_with_vehicle = int((df["n_vehicles"] > 0).sum())

    flagged_sample = (
        df[df["flagged"]]
        .sort_values(["n_faces", "n_persons", "n_vehicles"], ascending=False)
        .head(20)[["image_path", "n_persons", "n_faces", "n_vehicles", "flagged_reason"]]
    )

    detector_str = (
        "YOLOv8n (general) + YOLOv8n-face"
        if face_model_used
        else "YOLOv8n (general) — no face model"
    )
    lines = [
        "# PII screening summary",
        "",
        f"Local screen of {n_total} images using {detector_str}. "
        f"Confidence threshold ≥ {conf:.2f}. No images left the machine.",
        "",
        "## Counts",
        "",
        f"- **Total images**: {n_total}",
        f"- **Flagged** (any face / person / vehicle): {n_flagged} "
        f"({n_flagged / n_total:.1%})",
        f"- Contain a face: {n_with_face} ({n_with_face / n_total:.1%})",
        f"- Contain a person body (no face needed): {n_with_person} ({n_with_person / n_total:.1%})",
        f"- Contain a vehicle: {n_with_vehicle} ({n_with_vehicle / n_total:.1%})",
        "",
        "## Recommended action before cloud upload",
        "",
        f"- **Manually review** the {n_flagged} flagged images.",
        "- For images with faces: blur / redact faces, OR exclude from upload.",
        "- For images with vehicles: consider blurring license plates "
        "(YOLO finds the vehicle bounding box; a license-plate model can narrow it).",
        "- Re-screen after redaction to confirm faces are gone.",
        "",
        "## First 20 flagged images (by face count → person → vehicle)",
        "",
        flagged_sample.to_markdown(index=False) if not flagged_sample.empty else "_None flagged._",
        "",
        "## Notes",
        "",
        f"- Detector models: {detector_str}.",
        "- This screen is **recall-oriented**: borderline detections are flagged. "
        "Manual review is still needed.",
        "- Re-run with `--conf 0.3` for an even more cautious screen.",
        "- For a closer face-specific pass, download `yolov8n-face.pt` from "
        "https://github.com/akanametov/yolov8-face and pass `--face-model yolov8n-face.pt`.",
        "- Source CSV: `data/predictions/pii_screen.csv` — full per-image rows.",
        "- Regenerate with `python scripts/screen_images_for_pii.py`.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/raw/citywide/images"),
        help="Directory to recurse for images.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/predictions/pii_screen.csv"),
    )
    p.add_argument(
        "--summary-md",
        type=Path,
        default=Path("reports/pii_screen_summary.md"),
    )
    p.add_argument("--general-model", default=DEFAULT_GENERAL_MODEL)
    p.add_argument(
        "--face-model",
        default=None,
        help="Optional path to a YOLO face model (e.g., yolov8n-face.pt). Not auto-downloaded.",
    )
    p.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-images", type=int, default=None, help="Smoke-test cap.")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto if omitted).")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    paths = list(iter_image_paths(args.image_dir))
    if args.max_images is not None:
        paths = paths[: args.max_images]
    if not paths:
        raise SystemExit(f"No images found under {args.image_dir}")
    logger.info("Found %d images under %s", len(paths), args.image_dir)

    device = args.device or _select_device()
    df = screen_images(
        paths,
        general_model_id=args.general_model,
        face_model_id=args.face_model,
        conf=args.conf,
        device=device,
        batch_size=args.batch_size,
        repo_root=REPO_ROOT,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info(
        "Wrote %d screening rows to %s (%d flagged)",
        len(df),
        args.output_csv,
        int(df["flagged"].sum()),
    )

    render_summary(
        df, args.summary_md, conf=args.conf, face_model_used=bool(args.face_model)
    )
    logger.info("Wrote %s", args.summary_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
