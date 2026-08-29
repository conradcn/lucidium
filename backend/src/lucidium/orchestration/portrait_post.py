"""Post-render processing for character portraits.

The ComfyUI character workflow outputs an 832x1216 PNG with the
figure occupying only the central region — RMBG strips the
background, but the figure typically sits in roughly the upper 60%
of the canvas with transparent margin everywhere else. The renderer
then has to fit a 832x1216 alpha-bordered PNG into a fixed CSS box,
which forces the visible figure to be a fraction of the available
space.

The fix is to tightly crop the saved PNG to the figure's alpha
bounding box (with a small padding pixel margin so antialiasing
isn't sliced off). The cropped PNG ends up "all character" — every
pixel of every saved portrait is part of the figure, and the
renderer can scale it to fill its container with no empty
margins.

This module is pure-Python — uses only Pillow (already a project
dep). Failure-tolerant: if the input isn't a valid PNG with an
alpha channel, the function returns the original bytes unchanged
so the calling pipeline never breaks on a malformed render.
"""

from __future__ import annotations

import io
import logging

from PIL import Image

_log = logging.getLogger(__name__)

# Padding in pixels around the alpha bounding box. Enough to keep
# antialiased pixels on every edge without re-introducing visible
# margin in the cropped output.
_PADDING_PX = 6


def crop_to_figure(png_bytes: bytes) -> bytes:
    """Return ``png_bytes`` cropped to the alpha bounding box of its
    foreground figure. If the input has no alpha channel, no opaque
    pixels, or fails to decode, the original bytes are returned.
    """
    try:
        with Image.open(io.BytesIO(png_bytes)) as opened:
            image: Image.Image = opened
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            width, height = image.size
            alpha = image.getchannel("A")
            bbox = alpha.getbbox()
            if bbox is None:
                # Fully transparent — bail rather than crash.
                _log.warning("portrait crop: no opaque pixels, returning original bytes")
                return png_bytes
            left, top, right, bottom = bbox
            left = max(0, left - _PADDING_PX)
            top = max(0, top - _PADDING_PX)
            right = min(width, right + _PADDING_PX)
            bottom = min(height, bottom + _PADDING_PX)
            cropped = image.crop((left, top, right, bottom))
            buf = io.BytesIO()
            cropped.save(buf, "PNG", optimize=True)
            return buf.getvalue()
    except Exception:
        _log.warning("portrait crop failed; returning original bytes", exc_info=True)
        return png_bytes


__all__ = ["crop_to_figure"]
