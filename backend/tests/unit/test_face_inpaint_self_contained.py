"""The face-detail inpaint pass MUST be self-contained:

  * Pipeline weights — reused from the already-loaded text2img
    pipeline (same vae / unet / encoders / tokenizers). No
    second checkpoint, no download.
  * Face-detector model — opencv haar cascade XML, bundled via
    the spec's cv2/data datas entry.

If either drifts, a packaged user offline render either fails
with a network error or silently falls back to body-only with no
face-detail pass at all. Pin both via spec-source inspection +
embedded-image-client structural checks.
"""

from __future__ import annotations

from pathlib import Path

from lucidium.providers.embedded_image_client import (
    _build_inpaint_pipeline,
)


def _read_spec() -> str:
    spec = Path(__file__).resolve().parents[2] / "lucidium.spec"
    return spec.read_text(encoding="utf-8")


def test_spec_bundles_haar_cascade_data() -> None:
    """The spec's datas list must include the cv2/data
    directory (or at least the explicit XML lookup path) so the
    haar cascade XML rides into ``_internal/cv2/data/`` in the
    bundle."""
    src = _read_spec()
    # The spec resolves the cv2 data dir via ``_cv2.data.haarcascades``
    # at spec evaluation time — pin that bit of source.
    assert "haarcascades" in src, (
        "spec must reference cv2's haarcascades data path; "
        "without it the face-detail bbox detector falls back "
        "to a geometric guess that often misses the face."
    )


def test_inpaint_pipeline_reuses_text2img_components() -> None:
    """Helper that builds the inpaint pipeline must NOT call
    ``from_pretrained`` or fetch new weights — it reads vae /
    text_encoder / unet / scheduler off the already-loaded
    text2img pipeline. Verified structurally: the helper
    returns ``None`` for a pipeline missing any of those
    attributes (a Mock-style stub), proving the codepath
    requires the existing components rather than reaching for
    a remote checkpoint."""

    class _StubMissingUnet:
        # Has every attr except unet — ``_ensure_inpaint_pipeline``
        # checks ``hasattr(pipeline, "unet")`` and returns None
        # rather than trying to load one fresh.
        vae = object()
        text_encoder = object()
        text_encoder_2 = object()
        tokenizer = object()
        tokenizer_2 = object()
        scheduler = object()

    result = _build_inpaint_pipeline(_StubMissingUnet())
    assert result is None, (
        "inpaint helper must refuse to build a pipeline when "
        "the text2img pipeline is incomplete — falling back to "
        "from_pretrained would reach the network."
    )


def test_spec_disables_external_weight_fetching_for_inpaint() -> None:
    """Smoke-check: nothing in the inpaint codepath references
    ``from_pretrained`` or huggingface_hub. The face-detail
    pass must build entirely from in-memory components."""
    src = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "lucidium"
        / "providers"
        / "embedded_image_client.py"
    ).read_text(encoding="utf-8")
    # Pull the explicit face-inpaint helpers — exactly the
    # functions whose body runs at face-detail time, NOTHING
    # else. Each is captured by anchoring on its ``def`` line
    # and walking until the next module-level ``def`` / ``class``.
    targets = (
        "_run_face_inpaint",
        "_inpaint_one_face",
        "_face_inpaint_mask",
        "_build_inpaint_pipeline",
    )
    bodies: list[str] = []
    for name in targets:
        idx = src.index(f"def {name}")
        # Walk forward to the next top-level def/class.
        end = idx + len(name) + 4
        while True:
            nxt = src.find("\ndef ", end)
            cls = src.find("\nclass ", end)
            candidates = [c for c in (nxt, cls) if c >= 0]
            if not candidates:
                end = len(src)
                break
            end = min(candidates) + 1
            break
        bodies.append(src[idx:end])
    inpaint_body = "\n".join(bodies)
    forbidden = ("from_pretrained", "huggingface_hub", "hf_hub_download")
    for token in forbidden:
        assert token not in inpaint_body, (
            f"face-inpaint codepath references {token!r} — that "
            f"would reach the network at render time. Inpaint "
            f"must reuse the loaded text2img pipeline only."
        )
