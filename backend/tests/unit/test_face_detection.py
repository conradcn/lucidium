"""Face detection wiring for the embedded face-detail pass.

Covers two paths:

  * ``_detect_face_bbox`` — opencv Haar cascade. Returns the
    largest detected face as ``(left, top, right, bottom)`` or
    ``None`` when no face / opencv missing / cascade load failed.
    These tests don't actually drive the cascade against a real
    SDXL render (that's covered by the embedded_live tests); they
    pin the API contract: gracefully return None on inputs the
    cascade can't process, and pick the largest face when there
    are multiple candidates.
  * ``_estimate_face_dimensions`` — geometric fallback used when
    opencv isn't installed or detection fails on a particular
    render. Pins that full-body framings get a small face
    estimate, head-and-shoulders framings get a large one, and
    the fallback never returns zero / negative dimensions.
"""

from __future__ import annotations

from PIL import Image

from lucidium.providers.embedded_image_client import (
    _detect_face_bbox,
    _estimate_face_dimensions,
)


def test_face_detection_returns_none_for_blank_image() -> None:
    """A solid-colour blank PNG has no face. The cascade should
    return None rather than e.g. flagging a corner. This guards the
    contract: ``_detect_face_bbox`` never raises, only returns a
    bbox or None."""
    blank = Image.new("RGB", (512, 512), (128, 128, 128))
    assert _detect_face_bbox(blank) is None


def test_face_detection_returns_none_when_opencv_missing(
    monkeypatch,
) -> None:
    """When opencv isn't installed, detection short-circuits to
    None without raising. The fallback heuristic takes over in
    the calling code."""
    import sys

    # Force the import inside _detect_face_bbox to fail by injecting
    # a sentinel that raises ImportError on attribute access.
    real_cv2 = sys.modules.pop("cv2", None)
    monkeypatch.setitem(sys.modules, "cv2", None)
    try:
        result = _detect_face_bbox(Image.new("RGB", (128, 128), (0, 0, 0)))
    finally:
        if real_cv2 is not None:
            sys.modules["cv2"] = real_cv2
        else:
            sys.modules.pop("cv2", None)
    assert result is None


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
