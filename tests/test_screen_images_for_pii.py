"""Tests for the PII screening script.

We don't run YOLO here (no model load, no GPU). Instead we mock the
ultralytics output and exercise the row-building / flagging logic
directly. End-to-end is smoke-tested via ``python scripts/screen_images_for_pii.py
--max-images 6`` in CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_screener():
    spec = importlib.util.spec_from_file_location(
        "screen_images_for_pii",
        REPO_ROOT / "scripts" / "screen_images_for_pii.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


screener = _load_screener()


class _FakeTensor:
    """Mimics torch.Tensor.cpu().tolist() — ultralytics' Boxes returns these."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def cpu(self) -> "_FakeTensor":
        return self

    def tolist(self) -> list[float]:
        return list(self._values)


def _fake_result(class_names: dict[int, str], detections: list[tuple[str, float]]):
    cls_to_id = {n: i for i, n in class_names.items()}
    boxes = SimpleNamespace(
        conf=_FakeTensor([c for _, c in detections]),
        cls=_FakeTensor([float(cls_to_id[n]) for n, _ in detections]),
    )
    # ultralytics also lets us len(boxes)
    boxes.__len__ = lambda: len(detections)  # type: ignore[attr-defined]

    class _Boxes(SimpleNamespace):
        def __len__(self_inner) -> int:
            return len(detections)

    return SimpleNamespace(
        names=class_names,
        boxes=_Boxes(conf=boxes.conf, cls=boxes.cls),
    )


COCO_NAMES = {0: "person", 2: "car", 5: "bus", 7: "truck", 3: "motorcycle"}


def test_classes_above_filters_by_confidence() -> None:
    res = _fake_result(COCO_NAMES, [("person", 0.9), ("person", 0.3)])
    n, mc = screener._classes_above(res, {"person"}, conf=0.5)
    assert n == 1
    assert mc == 0.9


def test_classes_above_picks_only_named_classes() -> None:
    res = _fake_result(COCO_NAMES, [("person", 0.9), ("car", 0.8), ("truck", 0.7)])
    n_p, _ = screener._classes_above(res, {"person"}, conf=0.4)
    n_v, _ = screener._classes_above(res, {"car", "truck"}, conf=0.4)
    assert n_p == 1
    assert n_v == 2


def test_classes_above_empty_boxes_is_safe() -> None:
    res = SimpleNamespace(names=COCO_NAMES, boxes=None)
    n, mc = screener._classes_above(res, {"person"}, conf=0.5)
    assert (n, mc) == (0, 0.0)


def test_face_count_returns_count_above_threshold() -> None:
    res = _fake_result({0: "face"}, [("face", 0.7), ("face", 0.2), ("face", 0.6)])
    n, mc = screener._face_count(res, conf=0.5)
    assert n == 2
    assert mc == 0.7


def test_flag_reason_combines_categories() -> None:
    row = pd.Series({"n_faces": 0, "n_persons": 2, "n_vehicles": 1})
    assert screener._flag_reason(row) == "person×2, vehicle×1"

    row2 = pd.Series({"n_faces": 3, "n_persons": 0, "n_vehicles": 0})
    assert screener._flag_reason(row2) == "face×3"

    row3 = pd.Series({"n_faces": 0, "n_persons": 0, "n_vehicles": 0})
    assert screener._flag_reason(row3) == ""


def test_iter_image_paths_finds_jpegs(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.png").touch()
    (tmp_path / "c.txt").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "d.jpeg").touch()
    found = sorted(p.name for p in screener.iter_image_paths(tmp_path))
    assert found == ["a.jpg", "b.png", "d.jpeg"]


def test_render_summary_writes_expected_sections(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {"image_path": "a.jpg", "n_persons": 0, "n_faces": 0, "n_vehicles": 0,
             "persons_max_conf": 0, "faces_max_conf": 0, "vehicles_max_conf": 0,
             "flagged_reason": "", "flagged": False},
            {"image_path": "b.jpg", "n_persons": 1, "n_faces": 1, "n_vehicles": 0,
             "persons_max_conf": 0.9, "faces_max_conf": 0.8, "vehicles_max_conf": 0,
             "flagged_reason": "face×1, person×1", "flagged": True},
        ]
    )
    out = tmp_path / "summary.md"
    screener.render_summary(df, out, conf=0.4, face_model_used=False)
    text = out.read_text()
    assert "PII screening summary" in text
    assert "Total images**: 2" in text
    assert "Flagged" in text
    assert "b.jpg" in text  # flagged image listed
    assert "no face model" in text


def test_render_summary_notes_face_model_when_used(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [{"image_path": "x.jpg", "n_persons": 0, "n_faces": 0, "n_vehicles": 0,
          "persons_max_conf": 0, "faces_max_conf": 0, "vehicles_max_conf": 0,
          "flagged_reason": "", "flagged": False}]
    )
    out = tmp_path / "summary.md"
    screener.render_summary(df, out, conf=0.4, face_model_used=True)
    text = out.read_text()
    assert "YOLOv8n-face" in text
