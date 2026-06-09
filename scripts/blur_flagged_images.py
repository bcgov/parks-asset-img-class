"""For each PII-flagged image, write three side-by-side copies under data/pii_review/.

Reads ``data/predictions/pii_screen.csv``, re-runs YOLOv8 on the flagged
rows to recover bounding boxes (the screening CSV only stored counts),
then writes:

- ``data/pii_review/originals/<rel_path>``  — unmodified copy of the source image
- ``data/pii_review/boxed/<rel_path>``      — original with red rectangles drawn
                                              over every person / vehicle / face
- ``data/pii_review/blurred/<rel_path>``    — Gaussian-blurred inside the bboxes
                                              (rest of the image untouched)

This lets you eyeball the algorithm: did YOLO catch the actual person?
Is the blur enough to hide them? Are there false positives?

Usage:
    python scripts/blur_flagged_images.py
    python scripts/blur_flagged_images.py --max-images 20         # subset
    python scripts/blur_flagged_images.py --blur-radius 40        # heavier blur
    python scripts/blur_flagged_images.py --pad 10                # expand boxes
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("YOLO_VERBOSE", "False")

import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageOps  # noqa: E402

logger = logging.getLogger(__name__)

PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

# Per-category drawing colors (RGB).
COLORS = {
    "person": (255, 64, 64),    # red
    "vehicle": (255, 165, 0),   # orange
    "face": (255, 0, 255),      # magenta
}


def _select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _boxes_above(results, class_names: set[str], conf: float) -> list[tuple[float, float, float, float, str, float]]:
    """Return [(x1, y1, x2, y2, class_name, conf), ...] for detections above conf."""
    if results.boxes is None or len(results.boxes) == 0:
        return []
    names = results.names
    confs = results.boxes.conf.cpu().tolist()
    cls_ids = results.boxes.cls.cpu().tolist()
    xyxy = results.boxes.xyxy.cpu().tolist()
    out: list[tuple[float, float, float, float, str, float]] = []
    for (x1, y1, x2, y2), cid, c in zip(xyxy, cls_ids, confs, strict=False):
        if c < conf:
            continue
        cname = names[int(cid)]
        if class_names and cname not in class_names:
            continue
        out.append((float(x1), float(y1), float(x2), float(y2), cname, float(c)))
    return out


def _pad_box(box: tuple[float, float, float, float], pad: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, int(x1) - pad),
        max(0, int(y1) - pad),
        min(w, int(x2) + pad),
        min(h, int(y2) + pad),
    )


def draw_boxes(img: Image.Image, detections: list[tuple[int, int, int, int, str, float]]) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    line_w = max(2, min(img.width, img.height) // 200)
    for x1, y1, x2, y2, cname, c in detections:
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        category = "vehicle" if cname in VEHICLE_CLASSES else ("face" if cname == "face" else "person")
        color = COLORS[category]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        draw.text((x1 + 4, y1 + 4), f"{cname} {c:.2f}", fill=color)
    return out


def blur_regions(
    img: Image.Image,
    detections: list[tuple[int, int, int, int, str, float]],
    *,
    blur_radius: int,
) -> Image.Image:
    """Apply Gaussian blur only inside the detection bounding boxes."""
    if not detections:
        return img.copy()
    blurred_full = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    out = img.copy()
    for x1, y1, x2, y2, _cname, _c in detections:
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        region = blurred_full.crop((x1, y1, x2, y2))
        out.paste(region, (x1, y1))
    return out


def process_one_image(
    *,
    src: Path,
    flagged_rel: str,
    out_orig: Path,
    out_boxed: Path,
    out_blurred: Path,
    general,
    face,
    conf: float,
    pad: int,
    blur_radius: int,
    device: str,
) -> dict | None:
    """Run detection on a single flagged image and write the 3 output copies."""
    if not src.exists():
        logger.warning("missing source: %s", src)
        return None
    # Apply EXIF orientation up front so the pixels we detect on, draw on, and
    # save are all in the same (upright) coordinate frame. Without this, YOLO
    # detects on the EXIF-corrected image while we would blur the raw rotated
    # pixels, leaving the blur misaligned with the person.
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    w, h = img.size

    detections: list[tuple[int, int, int, int, str, float]] = []
    gen_results = general(img, conf=conf, device=device, verbose=False)[0]
    for box in _boxes_above(gen_results, PERSON_CLASSES | VEHICLE_CLASSES, conf):
        x1, y1, x2, y2, cname, c = box
        px1, py1, px2, py2 = _pad_box((x1, y1, x2, y2), pad, w, h)
        detections.append((px1, py1, px2, py2, cname, c))
    if face is not None:
        face_results = face(img, conf=conf, device=device, verbose=False)[0]
        for box in _boxes_above(face_results, set(), conf):
            x1, y1, x2, y2, _cname, c = box
            px1, py1, px2, py2 = _pad_box((x1, y1, x2, y2), pad, w, h)
            detections.append((px1, py1, px2, py2, "face", c))

    for out_path in (out_orig, out_boxed, out_blurred):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, out_orig)
    draw_boxes(img, detections).save(out_boxed, quality=90)
    blur_regions(img, detections, blur_radius=blur_radius).save(out_blurred, quality=90)

    return {
        "image_path": flagged_rel,
        "n_boxes": len(detections),
        "categories": ",".join(sorted({d[4] for d in detections})),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen-csv", type=Path, default=Path("data/predictions/pii_screen.csv"))
    p.add_argument("--output-root", type=Path, default=Path("data/pii_review"))
    p.add_argument("--general-model", default="yolov8n.pt")
    p.add_argument("--face-model", default=None, help="Optional path to a YOLO face model.")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument(
        "--pad",
        type=int,
        default=8,
        help="Pixels to expand each bbox before blur (so edges blend in).",
    )
    p.add_argument("--blur-radius", type=int, default=25, help="Gaussian blur radius in pixels.")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if not args.screen_csv.exists():
        raise SystemExit(
            f"{args.screen_csv} not found. Run scripts/screen_images_for_pii.py first."
        )
    screen = pd.read_csv(args.screen_csv)
    flagged = screen[screen["flagged"]].copy()
    if args.max_images is not None:
        flagged = flagged.head(args.max_images)
    if flagged.empty:
        logger.info("No flagged rows in %s; nothing to process.", args.screen_csv)
        return 0
    logger.info("Processing %d flagged images", len(flagged))

    from ultralytics import YOLO

    device = args.device or _select_device()
    general = YOLO(args.general_model)
    face = YOLO(args.face_model) if args.face_model else None

    out_orig_root = args.output_root / "originals"
    out_boxed_root = args.output_root / "boxed"
    out_blurred_root = args.output_root / "blurred"

    log_rows: list[dict] = []
    for i, rel in enumerate(flagged["image_path"].tolist(), start=1):
        src = REPO_ROOT / rel
        # Strip "data/raw/" so we don't double-nest data/ in the output tree.
        rel_path = Path(rel)
        if rel_path.parts[:2] == ("data", "raw"):
            rel_path = Path(*rel_path.parts[2:])
        record = process_one_image(
            src=src,
            flagged_rel=rel,
            out_orig=out_orig_root / rel_path,
            out_boxed=out_boxed_root / rel_path,
            out_blurred=out_blurred_root / rel_path,
            general=general,
            face=face,
            conf=args.conf,
            pad=args.pad,
            blur_radius=args.blur_radius,
            device=device,
        )
        if record is not None:
            log_rows.append(record)
        if i % 50 == 0:
            logger.info("Processed %d / %d", i, len(flagged))

    log_csv = args.output_root / "blur_log.csv"
    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(log_csv, index=False)
    print(
        f"\nWrote {len(log_rows)} flagged images × 3 copies under {args.output_root}:\n"
        f"  - {out_orig_root}  (untouched copies)\n"
        f"  - {out_boxed_root}  (with bbox overlays)\n"
        f"  - {out_blurred_root}  (with bbox regions blurred)\n"
        f"  - {log_csv}  (per-image detection log)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
