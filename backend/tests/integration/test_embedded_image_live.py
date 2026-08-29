"""Live smoke test for the embedded image-generation backend.

Unlike ``tests/unit/test_embedded_image_client.py`` (which mocks the
diffusers pipeline), this module loads the user's actual SDXL-family
checkpoint and renders one real PNG per workflow. The point is to
prove the embedded client's contract holds when wired to production
code paths — not just when the pipeline is faked.

Gated behind the ``embedded_live`` pytest marker so it doesn't run
in default CI invocations. To run::

    pytest -m embedded_live tests/integration/test_embedded_image_live.py

Requirements:
  * the ``embedded`` extras installed: ``pip install -e .[embedded]``
    (plus ``rembg[cpu]`` or ``rembg[gpu]`` for the alpha-cut pass);
  * a checkpoint .safetensors file present in either
    ``ImageSettings.embedded_models_dir`` or the bundled default
    (``<app-data>/models/image``).

The test reads the live ``Settings`` so it picks up whatever the
player has configured through the Settings UI. CPU-only
checkpoints at SDXL resolutions take many minutes per pass; this
file is intentionally slim (one background, one character) to keep
opt-in runtime under ~30 minutes on a mid-range CPU.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from PIL import Image

# huggingface_hub on Windows raises a UserWarning when symlinks
# aren't available (the cache then uses copies). The default
# ``filterwarnings = "error"`` in pyproject.toml escalates that
# to a hard fail; setting the env var BEFORE importing
# huggingface_hub silences it. Symlink-vs-copy doesn't change
# behaviour, just disk usage.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from lucidium.persistence import settings_store
from lucidium.providers.embedded_image_client import (
    WORKFLOW_DIMENSIONS,
    EmbeddedImageClient,
)
from lucidium.providers.embedded_models import (
    list_models,
    pick_default_model,
    resolve_models_dir,
)

# diffusers + NumPy 2.x: EulerDiscreteScheduler triggers a
# DeprecationWarning ("__array__ doesn't accept a copy keyword")
# during set_timesteps. The library still runs correctly; the warning
# is upstream's problem, not ours. Skipping symlink + the numpy
# deprecation here so the strict ``error`` filter elsewhere doesn't
# fail this opt-in live test.
_LIVE_FILTERS = [
    "ignore::DeprecationWarning",
    "ignore::UserWarning",
    "ignore::FutureWarning",
]


def _resolved_checkpoint_or_skip() -> Path:
    """Resolve the checkpoint the live test will load. Skips the test
    cleanly when nothing is configured + nothing is on disk —
    downloading SDXL Turbo (~6 GB) is a side-effect we only want to
    trigger on explicit player action, not in a smoke test."""
    settings = settings_store.load_settings()
    models_dir = resolve_models_dir(settings.image.embedded_models_dir)
    if not models_dir.exists() or not list_models(models_dir):
        pytest.skip(
            f"no checkpoint found at {models_dir}; configure "
            "ImageSettings.embedded_models_dir or drop a .safetensors "
            "file in the default app-data location"
        )
    target = pick_default_model(models_dir, settings.image.embedded_model_name)
    assert target is not None
    return target


def _decode_png(buf: bytes) -> Image.Image:
    return Image.open(io.BytesIO(buf))


@pytest.mark.embedded_live
@pytest.mark.asyncio
@pytest.mark.filterwarnings(*_LIVE_FILTERS)
async def test_embedded_background_renders_real_png() -> None:
    """Real diffusers pipeline → background workflow → PNG.

    Asserts the embedded client's output contract:
      * the bytes decode as PNG;
      * dimensions match ``WORKFLOW_DIMENSIONS["background.json"]``
        (the same shape the ComfyUI client returns for that
        workflow, so callers see one consistent contract).
    """
    target = _resolved_checkpoint_or_skip()
    settings = settings_store.load_settings()

    client = EmbeddedImageClient(
        models_dir=settings.image.embedded_models_dir,
        model_name=settings.image.embedded_model_name,
    )
    raw = await client.generate(
        "background.json",
        {"positive_prompt": "a stone harbor at dawn, dramatic light"},
        seed=12345,
    )

    image = _decode_png(raw)
    assert image.format == "PNG"
    assert (image.width, image.height) == WORKFLOW_DIMENSIONS["background.json"], (
        f"embedded background rendered at {(image.width, image.height)} but "
        f"the workflow contract requires "
        f"{WORKFLOW_DIMENSIONS['background.json']} (matches the "
        f"ComfyUI EmptyLatentImage node in workflows/background.json). "
        f"Loaded checkpoint: {target}"
    )


@pytest.mark.embedded_live
@pytest.mark.asyncio
@pytest.mark.filterwarnings(*_LIVE_FILTERS)
async def test_embedded_character_renders_with_alpha_cut() -> None:
    """Real diffusers pipeline + rembg → character workflow → PNG.

    Asserts:
      * bytes decode as PNG with an RGBA mode (rembg returns
        transparency; the renderer composites the figure over the
        scene background so an opaque PNG would visibly clash).
      * dimensions match ``WORKFLOW_DIMENSIONS["character.json"]``.
    """
    target = _resolved_checkpoint_or_skip()
    settings = settings_store.load_settings()

    client = EmbeddedImageClient(
        models_dir=settings.image.embedded_models_dir,
        model_name=settings.image.embedded_model_name,
    )
    raw = await client.generate(
        "character.json",
        {
            "positive_prompt": (
                "a young archivist in a long wool coat, full body, standing centered, neutral pose"
            ),
            "face_prompt": "calm watchful expression",
            "negative_extras": "extra people, crowd",
        },
        seed=4242,
    )

    image = _decode_png(raw)
    assert image.format == "PNG"
    assert (image.width, image.height) == WORKFLOW_DIMENSIONS["character.json"], (
        f"embedded character rendered at {(image.width, image.height)}; checkpoint: {target}"
    )
    assert image.mode == "RGBA", (
        f"character workflow must produce an alpha-cut PNG (got mode "
        f"{image.mode!r}); rembg is required for parity with the "
        f"ComfyUI character.json RMBG node"
    )


@pytest.mark.embedded_live
@pytest.mark.asyncio
@pytest.mark.filterwarnings(*_LIVE_FILTERS)
async def test_embedded_same_seed_produces_identical_renders() -> None:
    """Render the same scene twice with the same seed; assert the
    rendered images are byte-for-byte identical.

    This is the regression test for the generator-device bug: the
    earlier ``torch.Generator()`` (CPU) was being silently ignored
    when the SDXL pipeline lived on CUDA, and diffusers fell back
    to a fresh per-call random generator — so identical inputs
    produced different outputs every time. With the generator
    pinned to the pipeline's device, identical inputs reproduce
    identical outputs (which is the whole point of storing
    ``Character.seed`` on the character object).

    Skipping rembg (``bg_remover=None`` is the default; rembg
    introduces a small amount of nondeterminism via ONNX runtime).
    Background workflow exercises just the pipeline, no rembg.
    """
    _resolved_checkpoint_or_skip()
    settings = settings_store.load_settings()

    client = EmbeddedImageClient(
        models_dir=settings.image.embedded_models_dir,
        model_name=settings.image.embedded_model_name,
    )
    params = {"positive_prompt": "a stone harbor at dawn, dramatic light"}
    seed = 8675309

    first = await client.generate("background.json", params, seed=seed)
    second = await client.generate("background.json", params, seed=seed)

    assert first == second, (
        "Same prompt + same seed must reproduce byte-identical "
        "PNG output. If this fails, the diffusers generator "
        "device probably regressed (or a non-deterministic op "
        "snuck into the pipeline) — see "
        "_resolve_pipeline_device in embedded_image_client.py."
    )

    # And then with a DIFFERENT seed the output must differ — to
    # prove the seed actually has an effect (not just that the
    # pipeline is deterministically broken in some way that
    # produces the same thing regardless of seed).
    other = await client.generate("background.json", params, seed=seed + 1)
    assert other != first, (
        "Different seeds produced identical output — the seed "
        "is being ignored entirely. Check that the generator is "
        "constructed on the same device as the pipeline and that "
        "manual_seed is being applied."
    )


@pytest.mark.embedded_live
@pytest.mark.asyncio
@pytest.mark.filterwarnings(*_LIVE_FILTERS)
async def test_embedded_repeated_generations_no_scheduler_off_by_one() -> None:
    """Regression: ``EulerAncestralDiscreteScheduler.step()`` accesses
    ``self.sigmas[self.step_index + 1]``. If ``step_index`` ever
    runs past the end of the timesteps array (off-by-one in the
    pipeline's denoising loop, OR scheduler state leaking between
    calls) the access raises::

        IndexError: index 26 is out of bounds for dimension 0
        with size 26

    The user hits this on the embedded backend mid-session, and
    replicating it requires multiple consecutive renders against
    the SAME loaded pipeline (so any state that leaks between
    calls accumulates). This test fires four character renders
    back-to-back through the same client instance — each render
    uses different prompts and seeds so no caching path can
    short-circuit the pipeline. If any of them throws, the
    scheduler regression is confirmed.

    Skipping rembg keeps the variable-time tail (background
    removal) out of the picture; we want to catch ONLY the
    scheduler / pipeline interaction.
    """
    _resolved_checkpoint_or_skip()
    settings = settings_store.load_settings()

    client = EmbeddedImageClient(
        models_dir=settings.image.embedded_models_dir,
        model_name=settings.image.embedded_model_name,
        bg_remover=lambda b: b,  # no-op so we don't pay rembg per call
    )

    # Mix short prompts (single CLIP chunk) with long ones (force
    # compel into multi-chunk encoding). The off-by-one bug
    # surfaced under production conditions where consecutive
    # renders alternate between short and long prompts; reproducing
    # it needs that variation.
    long_prompt = (
        "(full body shot from head to toe:1.5), (standing centered, "
        "head and feet fully visible:1.4), wide framing, full-length "
        "figure, single subject, thirty-year-old female, ashkenazi "
        "jewish, dark brown loose curls hair, hazel eyes, "
        "(pose: standing rigid, socked feet flat:1.4), "
        "(wearing white bra and gray sweatpants:1.3), fair lightly "
        "freckled skin, slender build, small bust, masterpiece, "
        "cinematic photorealistic painting, soft window light, "
        "dramatic god rays, rich gold and crimson palette, painterly "
        "brushwork, fine detail, expressive lighting, calm dawn"
    )
    prompt_variants = [
        ("a tall paladin in plate armour, dawn light", 1001),
        (long_prompt, 2002),
        ("a wiry rogue in dark leathers, lamp glow", 3003),
        (long_prompt + ", autumn leaves drifting past window", 4004),
        ("a robed mage casting a calm spell, lantern shine", 5005),
    ]

    failures: list[str] = []
    for idx, (positive, seed) in enumerate(prompt_variants):
        try:
            raw = await client.generate(
                "character.json",
                {
                    "positive_prompt": positive,
                    "face_prompt": "calm watchful expression",
                    "negative_extras": "extra people",
                },
                seed=seed,
            )
            assert raw, f"render {idx} returned empty bytes"
        except IndexError as exc:
            failures.append(
                f"render {idx} (positive={positive!r}, seed={seed}) "
                f"hit a scheduler index error: {exc}"
            )
        except Exception as exc:
            failures.append(
                f"render {idx} (positive={positive!r}, seed={seed}) "
                f"raised {type(exc).__name__}: {exc}"
            )

    assert not failures, (
        "Repeated generations against the same pipeline tripped a "
        "failure — likely a scheduler step_index off-by-one. "
        "Failures:\n  - " + "\n  - ".join(failures)
    )


@pytest.mark.embedded_live
@pytest.mark.asyncio
@pytest.mark.filterwarnings(*_LIVE_FILTERS)
async def test_face_detail_does_not_leak_scheduler_state() -> None:
    """Regression: when ``face_detail=True``, every character render
    runs an SDXL inpaint pass through the same shared
    UNet/VAE/encoders. If the inpaint pipeline is built with the
    text2img pipeline's scheduler INSTANCE (rather than a fresh
    instance from the same config), the inpaint pipeline's
    ``get_timesteps`` truncates ``self.scheduler.timesteps`` and
    calls ``set_begin_index`` for the strength<1 path. The next
    text2img render then sees a truncated sigmas array but the
    full step_index counter and trips ``IndexError: index N is out
    of bounds for dimension 0 with size N`` on its final denoise
    step.

    This test fires several consecutive character renders WITH
    face_detail enabled, alternating workflows so any leaked
    scheduler state from the inpaint pass surfaces on the next
    text2img call.
    """
    _resolved_checkpoint_or_skip()
    settings = settings_store.load_settings()

    client = EmbeddedImageClient(
        models_dir=settings.image.embedded_models_dir,
        model_name=settings.image.embedded_model_name,
        bg_remover=lambda b: b,
        face_detail=True,
    )

    body_prompt = (
        "a forty-year-old woman in a long charcoal coat, full body, "
        "standing centered, dramatic side lighting"
    )
    face_prompts = [
        "calm composed expression, eyes forward",
        "fierce snarl, teeth bared, brows knit tight",
        "sad downcast eyes, lips trembling",
        "wide grin, beaming, eyes crinkled",
    ]

    failures: list[str] = []
    for idx, face in enumerate(face_prompts):
        try:
            raw = await client.generate(
                "character.json",
                {
                    "positive_prompt": body_prompt,
                    "face_prompt": face,
                    "negative_extras": "",
                },
                seed=10000 + idx,
            )
            assert raw, f"render {idx} returned empty bytes"
        except IndexError as exc:
            failures.append(
                f"render {idx} (face={face!r}) hit scheduler "
                f"IndexError: {exc} — inpaint pipeline is sharing a "
                f"scheduler instance with text2img and leaking state"
            )
        except Exception as exc:
            failures.append(f"render {idx} (face={face!r}) raised {type(exc).__name__}: {exc}")

    assert not failures, (
        "Face-detail pass leaked scheduler state into the next "
        "text2img call. Failures:\n  - " + "\n  - ".join(failures)
    )


@pytest.mark.embedded_live
@pytest.mark.asyncio
@pytest.mark.filterwarnings(*_LIVE_FILTERS)
async def test_face_detail_uses_guide_size_upscaling() -> None:
    """Regression for the ComfyUI-parity rewrite. The face-detail
    pass MUST match ComfyUI Impact-Pack's ``enhance_detail`` flow:
    detect → crop with crop_factor padding → resize UP to guide_size
    → inpaint at upscaled resolution → resize DOWN → composite.
    Without the guide_size upscale, a small detected face on a
    full-body render runs SDXL at ~12 latent rows of face — which
    produces the "nightmarish" output the user reported.

    This test renders a full-body figure (where the face is small
    relative to the canvas, the worst-case scenario for naive
    inpaint) with face_detail enabled and asserts the output
    differs MEANINGFULLY from a face_detail=False baseline. The
    naive (pre-fix) impl would land near-identical face pixels
    because SDXL had nothing to work with at 12-latent-row
    resolution.
    """
    from lucidium.providers.embedded_image_client import _detect_face_bboxes

    _resolved_checkpoint_or_skip()
    settings = settings_store.load_settings()

    body_prompt = (
        "a forty-year-old woman in a long charcoal coat, full body "
        "shot from head to toe, standing centered, head and feet "
        "fully visible, dramatic side lighting"
    )
    face_prompt = "fierce snarl, teeth bared, brows knit tight"
    seed = 12345

    client_off = EmbeddedImageClient(
        models_dir=settings.image.embedded_models_dir,
        model_name=settings.image.embedded_model_name,
        face_detail=False,
    )
    raw_off = await client_off.generate(
        "character.json",
        {"positive_prompt": body_prompt, "face_prompt": face_prompt},
        seed=seed,
    )

    client_on = EmbeddedImageClient(
        models_dir=settings.image.embedded_models_dir,
        model_name=settings.image.embedded_model_name,
        face_detail=True,
    )
    raw_on = await client_on.generate(
        "character.json",
        {"positive_prompt": body_prompt, "face_prompt": face_prompt},
        seed=seed,
    )

    img_off = _decode_png(raw_off)
    img_on = _decode_png(raw_on)
    assert img_off.size == img_on.size, (
        f"OFF / ON image dimensions differ: {img_off.size} vs {img_on.size}"
    )

    # Detection must find at least one face on the rendered image.
    detections = _detect_face_bboxes(img_on)
    assert len(detections) >= 1, (
        "opencv detector found no faces on the rendered character — "
        "the inpaint pass would have fallen back to the geometric "
        "heuristic, defeating the point of the new detection path"
    )

    # The face bbox region must change meaningfully between OFF and
    # ON (>10% of pixels differ). A naive non-upscaling inpaint
    # produces face changes around 1-3% on a full-body render
    # because SDXL has too few latent rows to work with — the new
    # impl's guide_size upscaling lifts the face latent grid to
    # ~64 rows, where SDXL renders crisp detail.
    fl, ft, fr, fb = detections[0]
    face_box = (fl, ft, fr, fb)
    a = img_off.crop(face_box).convert("RGB").tobytes()
    b = img_on.crop(face_box).convert("RGB").tobytes()
    diff = sum(1 for x, y in zip(a, b, strict=True) if x != y) / len(a)
    assert diff > 0.10, (
        f"face region differs by only {diff:.1%} between OFF and ON — "
        f"the inpaint pass barely changed pixels, which means SDXL "
        f"didn't have enough resolution to produce detail. Verify "
        f"the guide_size upscaling step in _inpaint_one_face is "
        f"actually firing."
    )


def _face_region_pixels(image: Image.Image) -> bytes:
    """Extract the top-third of the rendered character image — the
    region the face occupies. Comparing this slice (rather than the
    whole image) isolates face changes from variation in the body
    composition that any prompt change incidentally produces.

    Top-third on a 832x1216 character render is the 832x405 box at
    y=[0, 405). For a head-and-shoulders SDXL portrait that's a
    forgiving but tight bound — broad enough to survive small face
    repositioning, narrow enough that a body-only diff can't trip it.
    """
    width, height = image.size
    face_box = (0, 0, width, height // 3)
    cropped = image.crop(face_box)
    if cropped.mode != "RGB":
        cropped = cropped.convert("RGB")
    return cropped.tobytes()


def _pixel_diff_ratio(a: bytes, b: bytes) -> float:
    """Fraction of byte positions where ``a`` and ``b`` differ.
    Returns 0.0 when identical, 1.0 when fully different. Cheap
    proxy for "did the model produce a noticeably different image
    in this region" — a real perceptual metric would need scikit-
    image, which is too heavy a dep for an opt-in live test."""
    if len(a) != len(b):
        return 1.0
    if not a:
        return 0.0
    differing = sum(1 for x, y in zip(a, b, strict=True) if x != y)
    return differing / len(a)


@pytest.mark.embedded_live
@pytest.mark.asyncio
@pytest.mark.filterwarnings(*_LIVE_FILTERS)
async def test_face_prompt_actually_modifies_face() -> None:
    """Face-detailing regression test: rendering the SAME character
    body prompt with the SAME seed but DIFFERENT face_prompt MUST
    produce a different face region.

    The embedded backend appends ``face_prompt`` to the END of the
    positive prompt for character renders. If the appended tokens
    are silently dropped (e.g. a prompt-shape regression that
    swallows trailing text, or an encoding change that ignores the
    second SDXL CLIP encoder's input past 75 tokens), the face will
    look identical regardless of the face_prompt — the user's
    "face detailer doesn't seem to be working" report. This
    asserts the face region differs by a non-trivial fraction
    (>5%) when the face_prompt swings from "happy grin" to
    "angry scowl" while seed and body prompt are pinned.

    Skipping rembg because background removal can introduce edge
    noise around the face that masks the underlying face
    difference; raw RGB output isolates the SDXL contribution.
    """
    _resolved_checkpoint_or_skip()
    settings = settings_store.load_settings()

    client = EmbeddedImageClient(
        models_dir=settings.image.embedded_models_dir,
        model_name=settings.image.embedded_model_name,
        bg_remover=lambda b: b,  # skip rembg for clean pixel comparison
    )

    body_prompt = (
        "a thirty-year-old woman in a long charcoal trench coat, "
        "full body, standing centered, head and feet visible, "
        "neutral pose, dramatic studio lighting"
    )
    seed = 7777

    happy = await client.generate(
        "character.json",
        {
            "positive_prompt": body_prompt,
            "face_prompt": "wide grin, beaming, eyes crinkled with joy",
            "negative_extras": "",
        },
        seed=seed,
    )
    angry = await client.generate(
        "character.json",
        {
            "positive_prompt": body_prompt,
            "face_prompt": "fierce scowl, brows drawn together, lips pressed thin in anger",
            "negative_extras": "",
        },
        seed=seed,
    )

    # First-line check: the bytes themselves can't be identical —
    # if they are, face_prompt has zero effect on the encoder
    # output, which means the appending step is broken upstream.
    assert happy != angry, (
        "Same seed + same body prompt + DIFFERENT face_prompt "
        "produced byte-identical PNGs. face_prompt is being "
        "ignored entirely — check that the embedded client still "
        "appends face_prompt to positive_prompt for character "
        "workflows (see EmbeddedImageClient.generate)."
    )

    happy_face = _face_region_pixels(_decode_png(happy))
    angry_face = _face_region_pixels(_decode_png(angry))
    diff_ratio = _pixel_diff_ratio(happy_face, angry_face)
    # 5% is generous — empirically a swing this large in face
    # expression produces 30-60% pixel diff on the top-third of
    # an SDXL portrait. A diff under 5% means the face_prompt is
    # only nudging texture noise, not actually shaping the face.
    assert diff_ratio > 0.05, (
        f"face region pixel diff between happy/angry face_prompt is "
        f"only {diff_ratio:.1%} — face_prompt is barely influencing "
        f"the rendered face. Expect >5% on a real expression swing; "
        f"this looks like the face_prompt is being heavily diluted "
        f"by the body prompt or dropped past CLIP's 75-token cutoff."
    )


@pytest.mark.embedded_live
@pytest.mark.asyncio
@pytest.mark.filterwarnings(*_LIVE_FILTERS)
async def test_face_prompt_actually_changes_expression_axis() -> None:
    """Stronger check than the previous test: a smile face_prompt
    vs a neutral face_prompt (with the same seed/body) should
    produce a face that's MORE different from the neutral baseline
    than two neutral renders are from each other.

    Why this matters: the previous test only checks "different →
    different pixels". A noisy pipeline could fail that check by
    pure stochasticity (small sampler differences, CUDA reduction
    nondeterminism, etc.). This test calibrates the noise floor by
    measuring TWO neutral renders against each other (same seed
    AND same prompts → should be 0% diff if the engine is
    deterministic) and then asserts the angry render diverges from
    the neutral baseline by SUBSTANTIALLY more than that floor.
    """
    _resolved_checkpoint_or_skip()
    settings = settings_store.load_settings()

    client = EmbeddedImageClient(
        models_dir=settings.image.embedded_models_dir,
        model_name=settings.image.embedded_model_name,
        bg_remover=lambda b: b,
    )

    body_prompt = (
        "a forty-year-old man in a worn leather jacket, full body, "
        "standing centered, head and feet visible, dramatic side lighting"
    )
    seed = 99999
    neutral_prompt = "neutral expression, mouth closed, eyes forward"
    angry_prompt = "fierce snarl, teeth bared, brows knit tight in fury"

    neutral_a = await client.generate(
        "character.json",
        {"positive_prompt": body_prompt, "face_prompt": neutral_prompt},
        seed=seed,
    )
    neutral_b = await client.generate(
        "character.json",
        {"positive_prompt": body_prompt, "face_prompt": neutral_prompt},
        seed=seed,
    )
    angry = await client.generate(
        "character.json",
        {"positive_prompt": body_prompt, "face_prompt": angry_prompt},
        seed=seed,
    )

    neutral_a_face = _face_region_pixels(_decode_png(neutral_a))
    neutral_b_face = _face_region_pixels(_decode_png(neutral_b))
    angry_face = _face_region_pixels(_decode_png(angry))

    noise_floor = _pixel_diff_ratio(neutral_a_face, neutral_b_face)
    expression_signal = _pixel_diff_ratio(neutral_a_face, angry_face)

    # The angry expression must produce a face diff at LEAST 5x
    # the noise floor. If the floor is non-zero (some sampler
    # nondeterminism we can't avoid) we still want the expression
    # change to be visible against it. The 5x multiplier survives
    # routine pipeline variance while flagging a face_prompt that
    # only contributes background-noise-level variation.
    threshold = max(0.05, noise_floor * 5)
    assert expression_signal > threshold, (
        f"face_prompt swing 'neutral' → 'angry snarl' produced only "
        f"{expression_signal:.1%} face-region diff, while the "
        f"deterministic noise floor between two identical neutral "
        f"renders is {noise_floor:.1%}. Threshold for a real "
        f"expression effect: >{threshold:.1%}. face_prompt is being "
        f"swamped by the body prompt or dropped past CLIP's window."
    )
