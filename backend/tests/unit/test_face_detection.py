"""Face detection wiring for the embedded face-detail pass.

Covers two paths:

  * ``_detect_face_bbox`` — opencv YuNet (``cv2.FaceDetectorYN``).
    Returns the largest detected face as ``(left, top, right,
    bottom)`` or ``None`` when no face was found, and raises
    ``FaceDetectorUnavailableError`` when the detector itself can't
    run (opencv missing, no ``FaceDetectorYN``, model missing) — a
    broken detector must never look like "no faces" (#74). One test
    drives the real detector against the checked-in Dream Guide
    placeholder render, so an opencv upgrade that loads the model
    but stops finding faces fails here too.
  * ``_estimate_face_dimensions`` — geometric fallback used when
    detection finds no face on a particular render. Pins that full-body framings get a small face
    estimate, head-and-shoulders framings get a large one, and
    the fallback never returns zero / negative dimensions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from lucidium.providers import embedded_image_client
from lucidium.providers.embedded_image_client import (
    FaceDetectorUnavailableError,
    _detect_face_bbox,
    _detect_face_bboxes,
    _estimate_face_dimensions,
    _load_face_detector,
)

_DREAM_GUIDE = (
    Path(__file__).resolve().parents[2] / "workflows" / "placeholders" / "dream_guide.png"
)


@pytest.fixture
def fresh_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the cached detector so the test exercises the load path."""
    monkeypatch.setattr(embedded_image_client, "_FACE_DETECTOR", None)


def test_installed_opencv_ships_face_detector() -> None:
    """The installed opencv must provide ``FaceDetectorYN`` and the
    packaged YuNet model must load. OpenCV 5 dropped the Haar cascade
    API detection used to rely on; detection caught the
    AttributeError and returned no faces, silently disabling the
    face-detail pass (#74). Fail loudly here instead."""
    import cv2

    assert hasattr(cv2, "FaceDetectorYN"), f"opencv {cv2.__version__} has no FaceDetectorYN"
    model = embedded_image_client._YUNET_MODEL_PATH
    assert model.is_file(), f"YuNet face-detector model missing at {model}"
    assert _load_face_detector() is not None


def test_face_detection_finds_face_in_real_render() -> None:
    """End-to-end check against a real render: the Dream Guide
    placeholder (a full-figure portrait with alpha) has one face at
    the top-centre. Catches an opencv build that loads the model but
    no longer produces detections."""
    image = Image.open(_DREAM_GUIDE).convert("RGBA")
    alpha = image.getchannel("A")
    faces = _detect_face_bboxes(image, figure_bbox=alpha.getbbox(), alpha=alpha)
    assert len(faces) == 1, f"expected exactly one face, got {faces}"
    left, top, right, bottom = faces[0]
    cx, cy = (left + right) // 2, (top + bottom) // 2
    assert abs(cx - 411) <= 40 and abs(cy - 171) <= 40, (
        f"face centre {(cx, cy)} is not on the Dream Guide's head (~(411, 171))"
    )
    assert 50 <= right - left <= 150 and 70 <= bottom - top <= 200, faces[0]


def test_face_detection_returns_none_for_blank_image() -> None:
    """A solid-colour blank PNG has no face. The detector should
    return None rather than e.g. flagging a corner."""
    blank = Image.new("RGB", (512, 512), (128, 128, 128))
    assert _detect_face_bbox(blank) is None


def test_face_detection_raises_when_opencv_missing(
    monkeypatch: pytest.MonkeyPatch,
    fresh_detector: None,
) -> None:
    """When opencv can't be imported, detection raises instead of
    reporting "no faces". The face-detail and expression callers log
    the failure and carry on with the body render, so it shows up in
    the log rather than silently disabling the pass (#74)."""
    # ``None`` in sys.modules makes ``import cv2`` raise ImportError.
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(FaceDetectorUnavailableError, match="not installed"):
        _detect_face_bbox(Image.new("RGB", (128, 128), (0, 0, 0)))


def test_face_detection_raises_when_opencv_lacks_face_detector(
    monkeypatch: pytest.MonkeyPatch,
    fresh_detector: None,
) -> None:
    """An opencv build without ``FaceDetectorYN`` is the #74 failure
    mode (an API removed under us) — it must raise, not return []."""
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(__version__="9.9.9"))
    with pytest.raises(FaceDetectorUnavailableError, match="FaceDetectorYN"):
        _detect_face_bboxes(Image.new("RGB", (128, 128), (0, 0, 0)))


def test_face_detection_raises_when_model_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fresh_detector: None,
) -> None:
    """A missing YuNet ONNX (e.g. a bundle built without it) must
    raise rather than look like a faceless render."""
    monkeypatch.setattr(embedded_image_client, "_YUNET_MODEL_PATH", tmp_path / "missing.onnx")
    with pytest.raises(FaceDetectorUnavailableError, match="model missing"):
        _detect_face_bboxes(Image.new("RGB", (128, 128), (0, 0, 0)))


def test_estimate_face_dimensions_full_body_gives_small_face() -> None:
    """A full-body figure (filling 100% of canvas height) should
    produce a face estimate of roughly 1/7 of figure height —
    classic 7-and-a-half-heads-tall human proportions."""
    canvas_h = 1216
    fig_bbox = (33, 0, 778, canvas_h)
    face_h, face_w = _estimate_face_dimensions(
        fig_bbox,
        canvas_width=832,
        canvas_height=canvas_h,
    )
    # Full-body face fraction is 1/7 → ~173 px on a 1216-tall figure
    assert 140 <= face_h <= 200, (
        f"full-body face_h_est should be around canvas/7 (~173); got {face_h}"
    )
    assert 90 <= face_w <= 150, f"face_w should be ~0.7 × face_h (~120); got {face_w}"


def test_estimate_face_dimensions_portrait_gives_large_face() -> None:
    """A head-and-shoulders portrait (figure fills 50% of canvas)
    should produce a face estimate that's a larger fraction of the
    figure — the face fills most of the visible body."""
    canvas_h = 1216
    fig_top = canvas_h // 4
    fig_bottom = fig_top + canvas_h // 2  # figure fills 50% of canvas
    fig_bbox = (200, fig_top, 632, fig_bottom)
    face_h, _face_w = _estimate_face_dimensions(
        fig_bbox,
        canvas_width=832,
        canvas_height=canvas_h,
    )
    fig_h = fig_bottom - fig_top
    # Tight portrait should give face ~50% of figure height.
    assert face_h >= fig_h // 3, (
        f"portrait face_h_est should be at least figure_height/3 "
        f"({fig_h // 3}); got {face_h} for fig_height={fig_h}"
    )


def test_estimate_face_dimensions_no_bbox_uses_canvas_fallback() -> None:
    """When the figure bbox is None (no alpha channel, e.g. RGB-only
    test stub render), the helper falls back to a canvas-relative
    estimate so callers don't crash."""
    face_h, face_w = _estimate_face_dimensions(
        None,
        canvas_width=512,
        canvas_height=512,
    )
    assert face_h > 0
    assert face_w > 0


def test_estimate_face_dimensions_zero_height_figure_returns_zeros() -> None:
    """Defensive: a degenerate bbox where top == bottom should
    return (0, 0) so the caller skips the inpaint pass instead of
    dividing by zero somewhere downstream."""
    fig_bbox = (10, 50, 100, 50)  # zero height
    face_h, face_w = _estimate_face_dimensions(
        fig_bbox,
        canvas_width=832,
        canvas_height=1216,
    )
    assert face_h == 0 and face_w == 0
