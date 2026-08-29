"""Tests for portrait post-processing.

The primary purpose of this module — beyond the per-function
behaviour checks — is to PROVE that ``orchestration.portrait_post``
imports cleanly in the test environment. The module depends on
Pillow; if ``Pillow`` is missing from the active venv the import
fails immediately. Catching that here means the production sidecar
never crashes mid-render with ``ModuleNotFoundError: PIL``.
"""

from __future__ import annotations

import io

import pytest


def test_import_does_not_raise() -> None:
    """If Pillow isn't installed in the active env, this import
    raises and the whole test module fails to collect — surfacing
    the missing-dep at ``pytest`` time instead of at runtime."""
    from lucidium.orchestration import portrait_post  # noqa: F401


def test_crop_to_figure_tightens_alpha_bbox() -> None:
    """Crop should reduce a mostly-transparent canvas to its
    opaque-pixel bounding box (with a small padding margin)."""
    from PIL import Image

    from lucidium.orchestration.portrait_post import crop_to_figure

    canvas = Image.new("RGBA", (200, 400), (0, 0, 0, 0))
    # Paint an opaque rectangle from (60, 80) to (140, 320).
    for x in range(60, 140):
        for y in range(80, 320):
            canvas.putpixel((x, y), (200, 200, 200, 255))
    buf = io.BytesIO()
    canvas.save(buf, "PNG")
    cropped_bytes = crop_to_figure(buf.getvalue())

    cropped = Image.open(io.BytesIO(cropped_bytes))
    # Original is 200x400; opaque region is 80x240 plus 6px padding
    # on each side -> 92x252.
    assert cropped.size == (92, 252)
    assert cropped.size[0] < canvas.size[0]
    assert cropped.size[1] < canvas.size[1]


def test_crop_returns_original_for_fully_transparent_input() -> None:
    """Best-effort by contract: if there's no opaque pixel, return
    the input unchanged rather than crashing."""
    from PIL import Image

    from lucidium.orchestration.portrait_post import crop_to_figure

    canvas = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    buf = io.BytesIO()
    canvas.save(buf, "PNG")
    raw = buf.getvalue()

    assert crop_to_figure(raw) == raw


def test_crop_returns_original_for_invalid_bytes() -> None:
    """Failure-tolerant: malformed input returns unchanged bytes."""
    from lucidium.orchestration.portrait_post import crop_to_figure

    junk = b"not a png"
    assert crop_to_figure(junk) == junk


@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_crop_handles_non_rgba_input(mode: str) -> None:
    """RMBG always produces RGBA, but the helper coerces other
    modes too — keeps the contract simple if a caller ever feeds
    in a flattened image."""
    from PIL import Image

    from lucidium.orchestration.portrait_post import crop_to_figure

    canvas = Image.new(mode, (100, 100), 200)
    buf = io.BytesIO()
    canvas.save(buf, "PNG")
    # Should not raise; cropping a fully-opaque image just returns
    # (close to) the same canvas.
    out = crop_to_figure(buf.getvalue())
    assert isinstance(out, bytes) and len(out) > 0
