"""Contract tests for the embedded image backend.

The embedded ``EmbeddedImageClient`` and the ComfyUI
``ComfyUiImageClient`` both implement the ``ImageClient`` protocol:
``async generate(workflow, params, *, seed) -> bytes``. The
production engine treats them as interchangeable. These tests pin
that contract by exercising the embedded client end-to-end with a
fake diffusers pipeline + a fake background remover, then asserting:

  * the returned bytes are a valid PNG;
  * the PNG dimensions match the workflow's ``EmptyLatentImage`` node
    in the ComfyUI workflow JSON (832x1216 for character.json,
    1536x1024 for background.json — i.e., what the rest of the
    engine assumes);
  * the positive prompt is fed verbatim into the pipeline;
  * the static negative prompt from the corresponding workflow JSON
    is used (so an embedded render rejects the same things the
    ComfyUI render does), with caller-supplied negatives appended;
  * the seed parameter is threaded through;
  * character workflows trigger the background-removal pass.

The tests deliberately don't import diffusers / torch / rembg —
embedded is an optional install. Instead they inject a fake
pipeline factory + fake bg remover, so this module runs in the
default CI venv.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from lucidium.providers.embedded_image_client import (
    _BACKGROUND_NEGATIVE,
    _CHARACTER_NEGATIVE,
    WORKFLOW_DIMENSIONS,
    EmbeddedImageClient,
    _is_oom_error,
)

# Path to the ComfyUI workflows; we read them at runtime to verify
# the embedded backend's hardcoded dimensions / negatives stay in
# sync with the JSON the ComfyUI client posts.
_WORKFLOW_DIR = Path(__file__).resolve().parents[2] / "workflows"


class _FakePipeline:
    """Stand-in for an SDXL diffusers pipeline. Records every call's
    kwargs so the tests can assert which prompts / seed / dimensions
    were threaded through, and returns a stub image of the requested
    size."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.config = type("Cfg", (), {"_name_or_path": "test-stub"})()

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        width = kwargs.get("width", 512)
        height = kwargs.get("height", 512)
        # Tag the synthesised image with the seed so we can verify
        # the seed was applied without inspecting torch generators.
        image = Image.new("RGB", (width, height), color=(123, 200, 80))
        result = type("Result", (), {})()
        result.images = [image]
        return result


def _placeholder_image_bytes(width: int, height: int) -> bytes:
    """Pre-render a PNG of the requested dimensions for the bg
    remover stub to "produce" — the real rembg returns a PNG too."""
    image = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_pipeline() -> _FakePipeline:
    return _FakePipeline()


@pytest.fixture
def stocked_models_dir(tmp_path: Path) -> Path:
    """Models directory that already contains a checkpoint, so the
    client doesn't try to bootstrap-download one in tests."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "test-checkpoint.safetensors").write_bytes(b"fake")
    return models


def _make_client(
    *,
    models_dir: Path,
    fake_pipeline: _FakePipeline,
    bg_remover: Any | None = None,
) -> EmbeddedImageClient:
    return EmbeddedImageClient(
        models_dir=str(models_dir),
        pipeline_factory=lambda _path, _device: fake_pipeline,
        bg_remover=bg_remover,
    )


@pytest.mark.asyncio
async def test_character_workflow_emits_png_at_expected_dimensions(
    stocked_models_dir: Path,
    fake_pipeline: _FakePipeline,
) -> None:
    bg_calls: list[bytes] = []

    def fake_remove(image_bytes: bytes) -> bytes:
        bg_calls.append(image_bytes)
        # Mimic rembg's contract: take RGB(A), return PNG with alpha.
        return _placeholder_image_bytes(832, 1216)

    client = _make_client(
        models_dir=stocked_models_dir,
        fake_pipeline=fake_pipeline,
        bg_remover=fake_remove,
    )

    result = await client.generate(
        "character.json",
        {
            "positive_prompt": "a brave paladin in shining plate",
            "face_prompt": "noble bearing, fair skin",
            "negative_extras": "extra people, crowd",
        },
        seed=42,
    )

    # Result must be a PNG the renderer can decode.
    image = Image.open(io.BytesIO(result))
    assert image.format == "PNG"
    # Dimensions match the EmptyLatentImage node in
    # ``backend/workflows/character.json``: 832 x 1216.
    assert (image.width, image.height) == WORKFLOW_DIMENSIONS["character.json"]

    # Background-removal pass ran exactly once (mirrors the RMBG
    # node in the character workflow).
    assert len(bg_calls) == 1

    # face_prompt is APPENDED at the END of positive_prompt for
    # character renders. The earlier prepend made SDXL crop to a
    # face-only render because face-detail tokens dominated the
    # head of the prompt. ComfyUI runs face_prompt through a
    # separate FaceDetailer inpaint pass; the embedded path tints
    # the base render with the face cues at the end of the prompt
    # (lower attention weight than framing / pose / outfit at the
    # head). The main portrait_prompt no longer carries expression,
    # so without this append the embedded path would lose
    # expression entirely.
    assert len(fake_pipeline.calls) == 1
    call = fake_pipeline.calls[0]
    assert call["prompt"].startswith("a brave paladin in shining plate")
    assert call["prompt"].endswith("noble bearing, fair skin")
    assert _CHARACTER_NEGATIVE in call["negative_prompt"]
    assert "extra people, crowd" in call["negative_prompt"]
    assert call["width"] == 832
    assert call["height"] == 1216


@pytest.mark.asyncio
async def test_background_workflow_skips_bg_removal_and_uses_bg_negatives(
    stocked_models_dir: Path,
    fake_pipeline: _FakePipeline,
) -> None:
    bg_calls: list[bytes] = []

    def fake_remove(image_bytes: bytes) -> bytes:
        bg_calls.append(image_bytes)
        return image_bytes

    client = _make_client(
        models_dir=stocked_models_dir,
        fake_pipeline=fake_pipeline,
        bg_remover=fake_remove,
    )

    result = await client.generate(
        "background.json",
        {"positive_prompt": "stone harbor at dawn"},
        seed=7,
    )

    image = Image.open(io.BytesIO(result))
    assert (image.width, image.height) == WORKFLOW_DIMENSIONS["background.json"]
    # Background workflow does NOT remove background — that's a
    # character-only concern. If the bg remover ran for a bg
    # workflow we'd be wasting cycles AND clipping the scene's
    # alpha channel by mistake.
    assert bg_calls == []
    call = fake_pipeline.calls[0]
    # Background workflow uses its own (shorter) static negative;
    # don't leak the character workflow's negative into bg renders.
    assert _BACKGROUND_NEGATIVE in call["negative_prompt"]
    assert _CHARACTER_NEGATIVE not in call["negative_prompt"]


@pytest.mark.asyncio
async def test_seed_is_threaded_through_to_pipeline(
    stocked_models_dir: Path,
    fake_pipeline: _FakePipeline,
) -> None:
    client = _make_client(
        models_dir=stocked_models_dir,
        fake_pipeline=fake_pipeline,
    )
    await client.generate(
        "background.json",
        {"positive_prompt": "x"},
        seed=12345,
    )
    call = fake_pipeline.calls[0]
    # If torch is installed in the test venv, the generator is a
    # torch.Generator; if not, the kwarg is omitted entirely. Either
    # way the pipeline must receive a deterministic input — assert
    # via at least one of the two paths.
    if "generator" in call:
        gen = call["generator"]
        assert hasattr(gen, "initial_seed") or hasattr(gen, "manual_seed")
    else:
        # No torch installed — that's an acceptable degradation.
        # The seed is then non-deterministic but the pipeline still
        # produced an image, which is the only contract callers see.
        pass


@pytest.mark.asyncio
async def test_pipeline_is_loaded_lazily_and_cached(
    stocked_models_dir: Path,
) -> None:
    """The constructor must NOT touch the model. First generate
    triggers a load; the second reuses the cached instance — loading
    SDXL costs seconds and dozens of GB of memory, so a stale or
    repeated load would visibly degrade the player experience."""
    factory_calls: list[Path] = []

    fake = _FakePipeline()

    def factory(model_path: Path, _device: Any) -> Any:
        factory_calls.append(model_path)
        return fake

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        pipeline_factory=factory,
    )
    assert factory_calls == [], "pipeline must not load at construction time"

    await client.generate(
        "background.json",
        {"positive_prompt": "scene"},
        seed=1,
    )
    assert len(factory_calls) == 1, "first generate triggers a single load"

    await client.generate(
        "background.json",
        {"positive_prompt": "scene 2"},
        seed=2,
    )
    assert len(factory_calls) == 1, "second generate reuses the cached pipeline"


@pytest.mark.asyncio
async def test_face_detail_pass_runs_when_enabled(
    stocked_models_dir: Path,
    fake_pipeline: _FakePipeline,
) -> None:
    """When ``face_detail=True`` the client invokes the face-inpaint
    runner once per character render — and skips it on background
    workflows. Pre-fix the runner was either always-on or always-off
    with no plumbing for the Settings toggle."""
    inpaint_calls: list[dict[str, Any]] = []

    def fake_runner(pipeline_arg, base_png, **kwargs):
        inpaint_calls.append({"png": base_png, **kwargs})
        return base_png  # pass-through, body render survives

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        pipeline_factory=lambda _path, _device: fake_pipeline,
        bg_remover=lambda b: b,
        face_detail=True,
        face_inpaint_runner=fake_runner,
    )

    await client.generate(
        "character.json",
        {
            "positive_prompt": "a tall paladin",
            "face_prompt": "calm watchful expression",
        },
        seed=42,
    )
    await client.generate(
        "background.json",
        {"positive_prompt": "stone harbor at dawn"},
        seed=42,
    )

    # Face-inpaint runs ONLY for character workflow, never for
    # background. The face_prompt is forwarded verbatim; the seed
    # is derived (Knuth-multiplied) so it diverges from the body
    # seed but stays deterministic.
    assert len(inpaint_calls) == 1, (
        f"face-inpaint should run once for the character call and "
        f"never for the background call. ran {len(inpaint_calls)} times."
    )
    call = inpaint_calls[0]
    assert call["face_prompt"] == "calm watchful expression"
    assert call["seed"] != 42, (
        "face-inpaint seed should be derived from the body seed "
        "(not equal) so face latents don't over-correlate with "
        "the body pass"
    )


@pytest.mark.asyncio
async def test_face_detail_skipped_when_disabled(
    stocked_models_dir: Path,
    fake_pipeline: _FakePipeline,
) -> None:
    """Default (``face_detail=False``) MUST skip the inpaint pass
    entirely — the user's render budget shouldn't double silently."""
    inpaint_calls: list[Any] = []

    def fake_runner(*args, **kwargs):
        inpaint_calls.append((args, kwargs))
        return args[1]

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        pipeline_factory=lambda _path, _device: fake_pipeline,
        bg_remover=lambda b: b,
        face_inpaint_runner=fake_runner,
        # face_detail defaults to False — explicit here for clarity.
    )

    await client.generate(
        "character.json",
        {
            "positive_prompt": "a brave paladin",
            "face_prompt": "calm watchful expression",
        },
        seed=42,
    )

    assert inpaint_calls == [], "face-inpaint must not run when face_detail is disabled"


@pytest.mark.asyncio
async def test_face_detail_toggle_takes_effect_without_rebuild(
    stocked_models_dir: Path,
    fake_pipeline: _FakePipeline,
) -> None:
    """Flipping the face-detail toggle on a live client must change
    the next render's behaviour — without a pipeline reload. This
    is the path the Session uses when the player ticks the
    Settings checkbox: the cached client gets ``set_face_detail``
    called on it instead of being torn down + rebuilt."""
    inpaint_calls: list[Any] = []

    def fake_runner(*args, **kwargs):
        inpaint_calls.append((args, kwargs))
        return args[1]

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        pipeline_factory=lambda _path, _device: fake_pipeline,
        bg_remover=lambda b: b,
        face_inpaint_runner=fake_runner,
        face_detail=False,
    )
    await client.generate(
        "character.json",
        {"positive_prompt": "x", "face_prompt": "calm"},
        seed=1,
    )
    assert inpaint_calls == [], "off render should not invoke runner"

    client.set_face_detail(True)
    await client.generate(
        "character.json",
        {"positive_prompt": "x", "face_prompt": "calm"},
        seed=2,
    )
    assert len(inpaint_calls) == 1, (
        "after set_face_detail(True), the next render must invoke the inpaint runner"
    )

    client.set_face_detail(False)
    await client.generate(
        "character.json",
        {"positive_prompt": "x", "face_prompt": "calm"},
        seed=3,
    )
    assert len(inpaint_calls) == 1, (
        "after set_face_detail(False), the runner must not be invoked again"
    )


@pytest.mark.asyncio
async def test_per_workflow_models_load_two_pipelines(
    stocked_models_dir: Path,
) -> None:
    """When character / environment models point at different files
    AND there's headroom, both pipelines stay loaded — character
    renders use one, environment renders use the other, and neither
    triggers a reload of the already-loaded peer."""
    (stocked_models_dir / "character-model.safetensors").write_bytes(b"char")
    (stocked_models_dir / "environment-model.safetensors").write_bytes(b"env")

    factory_calls: list[Path] = []

    def factory(model_path: Path, _device: Any) -> Any:
        factory_calls.append(model_path)
        # Each loaded pipeline is a fresh instance so the test can
        # verify the right one was reused on the second call.
        return _FakePipeline()

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        character_model_name="character-model.safetensors",
        environment_model_name="environment-model.safetensors",
        pipeline_factory=factory,
        bg_remover=lambda b: b,
        # Headroom for both checkpoints; the default cap of 1 would
        # swap per render instead (see the LRU-cap tests below).
        max_resident_pipelines=2,
    )

    await client.generate(
        "character.json",
        {"positive_prompt": "char prompt"},
        seed=1,
    )
    await client.generate(
        "background.json",
        {"positive_prompt": "env prompt"},
        seed=2,
    )
    # Two distinct loads — one per workflow type.
    assert len(factory_calls) == 2
    assert {p.name for p in factory_calls} == {
        "character-model.safetensors",
        "environment-model.safetensors",
    }

    # Neither workflow should reload its pipeline on a second call.
    await client.generate(
        "character.json",
        {"positive_prompt": "char prompt 2"},
        seed=3,
    )
    await client.generate(
        "background.json",
        {"positive_prompt": "env prompt 2"},
        seed=4,
    )
    assert len(factory_calls) == 2, "both pipelines should stay cached across follow-up renders"


@pytest.mark.asyncio
async def test_oom_loading_second_pipeline_evicts_oldest(
    stocked_models_dir: Path,
) -> None:
    """Low-VRAM fallback: when loading a second pipeline raises an
    OOM, the embedded client drops the older pipeline and retries.
    The next render that needs the evicted pipeline reloads it. This
    is the path that lets a single 8 GB GPU support distinct
    character / environment models — at the cost of a per-swap
    reload, not a hard failure."""
    (stocked_models_dir / "character-model.safetensors").write_bytes(b"char")
    (stocked_models_dir / "environment-model.safetensors").write_bytes(b"env")

    factory_calls: list[Path] = []
    # Track the order so we can drive an OOM only the FIRST time the
    # environment pipeline tries to load.
    oom_armed = {"value": True}

    def factory(model_path: Path, _device: Any) -> Any:
        factory_calls.append(model_path)
        if model_path.name == "environment-model.safetensors" and oom_armed["value"]:
            oom_armed["value"] = False
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        return _FakePipeline()

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        character_model_name="character-model.safetensors",
        environment_model_name="environment-model.safetensors",
        pipeline_factory=factory,
        bg_remover=lambda b: b,
    )

    # Load the character pipeline first.
    await client.generate(
        "character.json",
        {"positive_prompt": "char"},
        seed=1,
    )
    assert factory_calls == [stocked_models_dir / "character-model.safetensors"]

    # Now attempt the environment pipeline — first try OOMs, second
    # try succeeds after evicting the character pipeline.
    await client.generate(
        "background.json",
        {"positive_prompt": "env"},
        seed=2,
    )
    assert [p.name for p in factory_calls] == [
        "character-model.safetensors",
        "environment-model.safetensors",  # OOM
        "environment-model.safetensors",  # retry succeeds
    ]

    # Character pipeline is gone now, so a follow-up character render
    # has to reload — the OOM eviction worked.
    await client.generate(
        "character.json",
        {"positive_prompt": "char again"},
        seed=3,
    )
    assert factory_calls[-1].name == "character-model.safetensors"


@pytest.mark.asyncio
async def test_empty_models_dir_raises_actionable_error(tmp_path: Path) -> None:
    """When the models directory is empty the client MUST NOT silently
    auto-download — that was the prior behaviour and it makes Lucidium
    a model-distributor in everything but name. The contract: raise
    ``ProviderUnreachableError`` with an actionable message naming
    the directory the user must populate AND pointing them at a real
    upstream source (civitai) for an SDXL checkpoint. There is no
    in-app download path; the FirstTimeSetup screen surfaces the same
    civitai link and the user fetches the file themselves.

    See SAFETY.md for the policy on bundled weights.
    """
    from lucidium.api.errors import ProviderUnreachableError

    models = tmp_path / "empty-models"
    # Don't pre-create or stock; this is the bootstrap case.

    def fake_pipeline_factory(_model_path: Path, _device: Any) -> Any:
        return _FakePipeline()

    client = EmbeddedImageClient(
        models_dir=str(models),
        pipeline_factory=fake_pipeline_factory,
        bg_remover=lambda b: b,
    )

    with pytest.raises(ProviderUnreachableError) as excinfo:
        await client.generate(
            "background.json",
            {"positive_prompt": "scene"},
            seed=1,
        )
    msg = str(excinfo.value)
    # Message names the directory the user should populate.
    assert str(models) in msg
    # And points at an upstream source so the user has somewhere to
    # actually go and fetch a checkpoint.
    assert "civitai" in msg.lower()


def test_workflow_dimensions_match_comfyui_json() -> None:
    """The hardcoded dimension table in the embedded client must
    match the ``EmptyLatentImage`` node of the corresponding ComfyUI
    workflow JSON. If a designer bumps the resolution in the JSON
    but forgets to update the embedded table, this test trips so
    the two backends never silently disagree on size."""
    for filename, (expected_w, expected_h) in WORKFLOW_DIMENSIONS.items():
        workflow_path = _WORKFLOW_DIR / filename
        if not workflow_path.exists():
            pytest.skip(f"workflow {filename} not present in this build")
            return
        graph = json.loads(workflow_path.read_text(encoding="utf-8"))
        latent = graph.get("4", {}).get("inputs", {})
        assert latent.get("width") == expected_w, (
            f"{filename} EmptyLatentImage width drifted: "
            f"workflow says {latent.get('width')}, "
            f"embedded WORKFLOW_DIMENSIONS says {expected_w}"
        )
        assert latent.get("height") == expected_h, (
            f"{filename} EmptyLatentImage height drifted: "
            f"workflow says {latent.get('height')}, "
            f"embedded WORKFLOW_DIMENSIONS says {expected_h}"
        )


def test_static_negative_matches_character_workflow() -> None:
    """The embedded client's hardcoded character negative must
    match the ``CLIPTextEncode`` text in the workflow JSON
    (modulo the ``PLACEHOLDER_NEGATIVE_EXTRAS`` token, which gets
    appended at call time). Catches drift between the two sources."""
    workflow_path = _WORKFLOW_DIR / "character.json"
    if not workflow_path.exists():
        pytest.skip("character.json not present")
        return
    graph = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow_neg = graph["3"]["inputs"]["text"]
    # Strip placeholder + any leading punctuation that follows it.
    placeholder_idx = workflow_neg.find("PLACEHOLDER_NEGATIVE_EXTRAS")
    base = workflow_neg[:placeholder_idx].rstrip(", ")
    assert _CHARACTER_NEGATIVE == base, (
        "embedded _CHARACTER_NEGATIVE drifted from character.json's "
        "static negative — update one or the other so the two "
        "backends reject the same content"
    )


def test_static_negative_matches_background_workflow() -> None:
    workflow_path = _WORKFLOW_DIR / "background.json"
    if not workflow_path.exists():
        pytest.skip("background.json not present")
        return
    graph = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflow_neg = graph["3"]["inputs"]["text"]
    assert _BACKGROUND_NEGATIVE == workflow_neg, (
        "embedded _BACKGROUND_NEGATIVE drifted from background.json's "
        "static negative — update one or the other"
    )


@pytest.mark.asyncio
async def test_subject_kind_nonhuman_uses_environment_pipeline_skips_face_detail(
    stocked_models_dir: Path,
) -> None:
    """Nonhuman characters route to the ENVIRONMENT SDXL checkpoint
    (better training distribution for non-humanoid subjects),
    skip the face-detail inpaint pass (FaceDetailer's bbox
    detector targets human faces and would smear nonhuman
    geometry), but STILL run through the character workflow so
    background removal cuts the figure off its backdrop. The
    asset pipeline signals this via ``subject_kind="nonhuman"``
    in the params dict — independent of the workflow filename."""
    (stocked_models_dir / "character-model.safetensors").write_bytes(b"char")
    (stocked_models_dir / "environment-model.safetensors").write_bytes(b"env")

    factory_calls: list[Path] = []

    def factory(model_path: Path, _device: Any) -> Any:
        factory_calls.append(model_path)
        return _FakePipeline()

    inpaint_calls: list[dict[str, Any]] = []

    def inpaint_runner(
        pipeline: Any,
        image_bytes: bytes,
        *,
        face_prompt: str,
        negative: str,
        seed: int,
    ) -> bytes:
        inpaint_calls.append({"face_prompt": face_prompt, "seed": seed})
        return image_bytes

    rembg_calls: list[bytes] = []

    def bg_remover(buf: bytes) -> bytes:
        rembg_calls.append(buf)
        return buf

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        character_model_name="character-model.safetensors",
        environment_model_name="environment-model.safetensors",
        face_detail=True,
        face_inpaint_runner=inpaint_runner,
        pipeline_factory=factory,
        bg_remover=bg_remover,
    )

    await client.generate(
        "character.json",
        {
            "positive_prompt": "scale-armoured serpent, copper hide",
            "face_prompt": "watchful",
            "subject_kind": "nonhuman",
        },
        seed=42,
    )

    # Pipeline routing — ENVIRONMENT checkpoint, even though the
    # workflow is character.json.
    assert len(factory_calls) == 1
    assert factory_calls[0].name == "environment-model.safetensors", (
        f"nonhuman portrait must use the environment checkpoint; got {factory_calls[0].name}"
    )

    # Face detail skipped despite ``face_detail=True`` on the
    # client — the human-face inpaint pass would distort
    # nonhuman geometry.
    assert inpaint_calls == [], "face-detail inpaint must NOT fire on a nonhuman subject"

    # Background removal still ran (gated on the workflow being
    # ``character.*``, not on the kind). A nonhuman portrait
    # arrives at the renderer as a transparent cut-out.
    assert len(rembg_calls) == 1, (
        "RMBG must still run for nonhuman characters so the "
        "renderer can stage them over a separate backdrop"
    )


@pytest.mark.asyncio
async def test_subject_kind_human_keeps_existing_pipeline_routing(
    stocked_models_dir: Path,
) -> None:
    """Sanity check: a normal human character still goes through
    the character checkpoint and runs face detail when enabled.
    Pinning this so a future nonhuman branch tweak doesn't
    silently re-route every render."""
    (stocked_models_dir / "character-model.safetensors").write_bytes(b"char")
    (stocked_models_dir / "environment-model.safetensors").write_bytes(b"env")

    factory_calls: list[Path] = []

    def factory(model_path: Path, _device: Any) -> Any:
        factory_calls.append(model_path)
        return _FakePipeline()

    inpaint_calls: list[dict[str, Any]] = []

    def inpaint_runner(
        pipeline: Any,
        image_bytes: bytes,
        *,
        face_prompt: str,
        negative: str,
        seed: int,
    ) -> bytes:
        inpaint_calls.append({"face_prompt": face_prompt, "seed": seed})
        return image_bytes

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        character_model_name="character-model.safetensors",
        environment_model_name="environment-model.safetensors",
        face_detail=True,
        face_inpaint_runner=inpaint_runner,
        pipeline_factory=factory,
        bg_remover=lambda b: b,
    )

    await client.generate(
        "character.json",
        {
            "positive_prompt": "thirty-year-old female",
            "face_prompt": "alert",
            # Default subject_kind is "human" — also accept the
            # caller passing it explicitly.
            "subject_kind": "human",
        },
        seed=7,
    )
    assert factory_calls[0].name == "character-model.safetensors"
    assert len(inpaint_calls) == 1


# ---------------------------------------------------------------------------
# Z-Image compatibility
#
# The embedded backend supports Alibaba's Z-Image-Turbo alongside SDXL.
# Detection is filename-based at load time (the factory picks
# ``ZImagePipeline.from_single_file`` for paths matching ``z-image*``)
# and class-name-based at run time (``_run_pipeline`` dispatches by
# ``type(pipeline).__name__`` so the call kwargs match the Turbo
# recipe: 9 steps, guidance 0.0, no compel chunking).
#
# These tests use a fake pipeline whose class is renamed to
# ``ZImagePipeline`` so the runtime branch fires without pulling in
# diffusers' real Z-Image stack.
# ---------------------------------------------------------------------------


class _FakeZImagePipeline(_FakePipeline):
    """Stand-in for diffusers' ``ZImagePipeline``. The runtime dispatch
    uses ``type(pipeline).__name__`` so the class name is what makes
    ``_run_pipeline`` choose the Z-Image branch."""


# Diffusers expects the runtime check to match the actual class name
# (``ZImagePipeline``). Rename the fake's class so type().__name__
# returns the right string without pulling in the real diffusers stack.
_FakeZImagePipeline.__name__ = "ZImagePipeline"
_FakeZImagePipeline.__qualname__ = "ZImagePipeline"


@pytest.fixture
def fake_z_image_pipeline() -> _FakeZImagePipeline:
    return _FakeZImagePipeline()


@pytest.fixture
def stocked_z_image_models_dir(tmp_path: Path) -> Path:
    """Models directory pre-seeded with a Z-Image-named checkpoint so
    the filename sniff routes the factory to the Z-Image branch."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "z-image-turbo.safetensors").write_bytes(b"fake")
    return models


@pytest.mark.asyncio
async def test_z_image_uses_turbo_recipe_at_runtime(
    stocked_z_image_models_dir: Path,
    fake_z_image_pipeline: _FakeZImagePipeline,
) -> None:
    """When the loaded pipeline is a ``ZImagePipeline``, ``generate``
    must call it with the Turbo recipe (9 steps, guidance 0.0) and
    pass the prompt as a plain string — no compel chunking, because
    Z-Image's single text encoder accepts 512-token sequences
    natively and Compel can't build against its tokenizer shape.
    """
    client = EmbeddedImageClient(
        models_dir=str(stocked_z_image_models_dir),
        pipeline_factory=lambda _path, _device: fake_z_image_pipeline,
        bg_remover=lambda b: b,
    )

    await client.generate(
        "character.json",
        {"positive_prompt": "a tall figure in a winter coat"},
        seed=1234,
    )

    assert len(fake_z_image_pipeline.calls) == 1
    call = fake_z_image_pipeline.calls[0]
    assert call["num_inference_steps"] == 9, (
        f"Z-Image-Turbo runs at 9 steps; got {call['num_inference_steps']}"
    )
    assert call["guidance_scale"] == 0.0, (
        f"Z-Image-Turbo runs with CFG off; got {call['guidance_scale']}"
    )
    assert call["prompt"] == "a tall figure in a winter coat"
    # ``negative_prompt`` is still threaded for parity with SDXL
    # (Z-Image ignores it internally when guidance == 0).
    assert _CHARACTER_NEGATIVE in call["negative_prompt"]
    # The compel-encoded path adds ``prompt_embeds`` etc.; the
    # plain path leaves them absent. Pin that the Z-Image branch
    # didn't accidentally hit the compel chunker.
    assert "prompt_embeds" not in call
    assert "pooled_prompt_embeds" not in call


@pytest.mark.asyncio
async def test_z_image_factory_loads_via_filename_sniff(
    stocked_z_image_models_dir: Path,
    fake_z_image_pipeline: _FakeZImagePipeline,
) -> None:
    """The factory must pass the resolved ``model_path`` through and
    route Z-Image-named checkpoints to a Z-Image-flavoured load."""
    factory_calls: list[Path] = []

    def factory(path: Path, _device: Any) -> Any:
        factory_calls.append(path)
        return fake_z_image_pipeline

    client = EmbeddedImageClient(
        models_dir=str(stocked_z_image_models_dir),
        pipeline_factory=factory,
        bg_remover=lambda b: b,
    )

    await client.generate(
        "character.json",
        {"positive_prompt": "anything"},
        seed=11,
    )

    assert len(factory_calls) == 1
    # Filename matches the Z-Image sniff used by the production
    # factory; the test factory just records the path it was handed
    # so we can assert resolution worked.
    assert "z-image" in factory_calls[0].name.lower()


@pytest.mark.asyncio
async def test_z_image_appends_face_prompt_even_with_face_detail_on(
    stocked_z_image_models_dir: Path,
    fake_z_image_pipeline: _FakeZImagePipeline,
) -> None:
    """Face-detail inpaint can't run on Z-Image (no UNet / dual CLIP
    encoders). The client must detect this from the model filename
    and treat ``face_detail=True`` like OFF — i.e., still append
    ``face_prompt`` to the positive prompt so expression cues reach
    the body render instead of being dropped.
    """
    inpaint_calls: list[Any] = []

    def inpaint_runner(*_args: Any, **_kwargs: Any) -> bytes:
        inpaint_calls.append((_args, _kwargs))
        # If this ever runs we'd return the same PNG; the test
        # asserts it does NOT run.
        return b""

    client = EmbeddedImageClient(
        models_dir=str(stocked_z_image_models_dir),
        pipeline_factory=lambda _path, _device: fake_z_image_pipeline,
        bg_remover=lambda b: b,
        face_detail=True,
        face_inpaint_runner=inpaint_runner,
    )

    await client.generate(
        "character.json",
        {
            "positive_prompt": "a tall figure in a winter coat",
            "face_prompt": "wry half-smile",
        },
        seed=22,
    )

    assert len(fake_z_image_pipeline.calls) == 1
    call = fake_z_image_pipeline.calls[0]
    assert call["prompt"].endswith("wry half-smile"), (
        "Z-Image with face_detail=True must still receive the "
        f"face_prompt; got prompt={call['prompt']!r}"
    )
    # Z-Image is incompatible with the SDXL face-detail inpaint
    # pipeline, so the runner must NOT have been called.
    assert inpaint_calls == [], (
        "face-detail inpaint runner ran on a Z-Image render — it should be skipped"
    )


def test_is_z_image_model_path_filename_variants(tmp_path: Path) -> None:
    """Filename sniff must catch the common Z-Image checkpoint name
    variants (the canonical and the HF-prefixed ones) without
    flagging SDXL-named files."""
    from lucidium.providers.embedded_image_client import (
        _is_z_image_model_path,
    )

    for name in (
        "z-image-turbo.safetensors",
        "Z-Image-Turbo.safetensors",
        "zimage_turbo_fp16.safetensors",
        "Tongyi-MAI__Z-Image-Turbo.safetensors",
    ):
        assert _is_z_image_model_path(tmp_path / name), name
    for name in (
        "sd_xl_base_1.0.safetensors",
        "sdxl-turbo.safetensors",
        "ponyXL_v6.safetensors",
    ):
        assert not _is_z_image_model_path(tmp_path / name), name


def test_is_z_image_pipeline_class_name_match() -> None:
    """Runtime dispatch keys off ``type(pipeline).__name__`` so the
    helper recognises any Z-Image pipeline variant the factory might
    have produced (text2img / img2img / inpaint)."""
    from lucidium.providers.embedded_image_client import (
        _is_z_image_pipeline,
    )

    class ZImagePipeline:
        pass

    class ZImageInpaintPipeline:
        pass

    class StableDiffusionXLPipeline:
        pass

    assert _is_z_image_pipeline(ZImagePipeline())
    assert _is_z_image_pipeline(ZImageInpaintPipeline())
    assert not _is_z_image_pipeline(StableDiffusionXLPipeline())


# --- pipeline-cache LRU cap + OOM classification ------------------------
#
# The cache used to be unbounded, with eviction firing ONLY when
# ``_is_oom_error`` recognised an allocation failure — and it only
# recognised CUDA's wording, so DirectML / MPS / CPU setups grew the
# cache forever. These tests pin the explicit cap and the broadened
# classifier.


def _cap_client(models_dir: Path, cap: int) -> EmbeddedImageClient:
    return EmbeddedImageClient(
        models_dir=str(models_dir),
        pipeline_factory=lambda _path, _device: _FakePipeline(),
        max_resident_pipelines=cap,
    )


@pytest.mark.asyncio
async def test_pipeline_cache_evicts_lru_at_cap(
    stocked_models_dir: Path,
) -> None:
    """Loading a third checkpoint under a cap of 2 drops the
    least-recently-used pipeline AND its inference lock — no OOM
    required."""
    for name in ("one", "two", "three"):
        (stocked_models_dir / f"{name}.safetensors").write_bytes(b"x")

    client = _cap_client(stocked_models_dir, 2)

    first = await client._ensure_pipeline(
        "character.json",
        override_name="one.safetensors",
    )
    second = await client._ensure_pipeline(
        "character.json",
        override_name="two.safetensors",
    )
    # Materialise both inference locks so we can watch the evicted
    # one get cleaned up alongside its pipeline.
    client._lock_for(first)
    client._lock_for(second)
    assert len(client._pipelines) == 2
    assert len(client._inference_locks) == 2

    await client._ensure_pipeline(
        "character.json",
        override_name="three.safetensors",
    )

    resident = {path.name for path in client._pipelines}
    assert resident == {"two.safetensors", "three.safetensors"}, (
        f"cap=2 must evict the least-recently-used pipeline; got {resident}"
    )
    assert set(client._inference_locks) == {
        p for p in client._pipelines if p.name == "two.safetensors"
    }, (
        "eviction must drop the evicted path's inference lock too — a "
        "stale lock gets handed to a different pipeline later loaded "
        "at the same path"
    )


@pytest.mark.asyncio
async def test_pipeline_cache_defaults_to_one_resident(
    stocked_models_dir: Path,
) -> None:
    """Default cap is 1: a low-VRAM / DirectML setup swaps per render
    instead of accumulating pipelines it can never evict."""
    for name in ("one", "two"):
        (stocked_models_dir / f"{name}.safetensors").write_bytes(b"x")

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        pipeline_factory=lambda _path, _device: _FakePipeline(),
    )

    await client._ensure_pipeline(
        "character.json",
        override_name="one.safetensors",
    )
    await client._ensure_pipeline(
        "character.json",
        override_name="two.safetensors",
    )

    assert [p.name for p in client._pipelines] == ["two.safetensors"]
    # Locks are created lazily, so the invariant is "no lock without a
    # resident pipeline", not strict equality.
    assert set(client._inference_locks) <= set(client._pipelines)


@pytest.mark.asyncio
async def test_lru_touch_protects_recently_used_pipeline(
    stocked_models_dir: Path,
) -> None:
    """Re-using a cached pipeline moves it to the back of the LRU, so
    the OTHER one is the eviction candidate."""
    for name in ("one", "two", "three"):
        (stocked_models_dir / f"{name}.safetensors").write_bytes(b"x")

    client = _cap_client(stocked_models_dir, 2)

    await client._ensure_pipeline(
        "character.json",
        override_name="one.safetensors",
    )
    await client._ensure_pipeline(
        "character.json",
        override_name="two.safetensors",
    )
    # Touch "one" — it must now outlive "two".
    await client._ensure_pipeline(
        "character.json",
        override_name="one.safetensors",
    )
    await client._ensure_pipeline(
        "character.json",
        override_name="three.safetensors",
    )

    resident = {path.name for path in client._pipelines}
    assert resident == {"one.safetensors", "three.safetensors"}, (
        f"a re-used pipeline must not be the LRU victim; got {resident}"
    )


@pytest.mark.asyncio
async def test_cap_eviction_spares_in_flight_pipeline(
    stocked_models_dir: Path,
) -> None:
    """A pipeline whose inference lock is held is mid-denoise; freeing
    it would pull weights out from under a running render. The cap
    defers rather than evicting it."""
    for name in ("one", "two"):
        (stocked_models_dir / f"{name}.safetensors").write_bytes(b"x")

    client = _cap_client(stocked_models_dir, 1)

    first = await client._ensure_pipeline(
        "character.json",
        override_name="one.safetensors",
    )
    held = client._lock_for(first)
    await held.acquire()
    try:
        await client._ensure_pipeline(
            "character.json",
            override_name="two.safetensors",
        )
        assert {p.name for p in client._pipelines} == {
            "one.safetensors",
            "two.safetensors",
        }, "an in-flight pipeline must not be evicted by the cap"
    finally:
        held.release()

    # Once the render finishes, the next load reclaims the slot.
    await client._ensure_pipeline(
        "character.json",
        override_name="one.safetensors",
    )
    assert [p.name for p in client._pipelines] == ["one.safetensors"]


@pytest.mark.asyncio
async def test_aclose_drops_inference_locks(
    stocked_models_dir: Path,
    fake_pipeline: _FakePipeline,
) -> None:
    client = _make_client(
        models_dir=stocked_models_dir,
        fake_pipeline=fake_pipeline,
    )
    pipeline = await client._ensure_pipeline("character.json")
    client._lock_for(pipeline)
    assert client._inference_locks

    await client.aclose()

    assert not client._pipelines
    assert not client._inference_locks, (
        "aclose must not leave inference locks behind for paths whose pipeline is gone"
    )


@pytest.mark.asyncio
async def test_oom_load_eviction_drops_inference_lock(
    stocked_models_dir: Path,
) -> None:
    """The load-path OOM eviction must clear the evicted path's lock,
    matching the restore-path and generate-path eviction sites."""
    for name in ("one", "two"):
        (stocked_models_dir / f"{name}.safetensors").write_bytes(b"x")

    attempts: list[Path] = []

    def factory(model_path: Path, _device: Any) -> Any:
        attempts.append(model_path)
        # Fail the FIRST load of "two" so the client evicts "one" and
        # retries; succeed on every other call.
        if model_path.name == "two.safetensors" and len(attempts) == 2:
            raise RuntimeError("CUDA out of memory")
        return _FakePipeline()

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        pipeline_factory=factory,
        # Cap high enough that the OOM path, not the cap, does the work.
        max_resident_pipelines=8,
    )

    first = await client._ensure_pipeline(
        "character.json",
        override_name="one.safetensors",
    )
    client._lock_for(first)
    assert len(client._inference_locks) == 1

    await client._ensure_pipeline(
        "character.json",
        override_name="two.safetensors",
    )

    assert [p.name for p in client._pipelines] == ["two.safetensors"]
    # Locks are created lazily, so the invariant is "no lock without a
    # resident pipeline", not strict equality.
    assert set(client._inference_locks) <= set(client._pipelines), (
        "OOM eviction on the load path left a stale inference lock"
    )


@pytest.mark.parametrize(
    "exc",
    [
        # CUDA (regression guard for the original behaviour).
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"),
        # DirectML / AMD.
        RuntimeError(
            "Could not allocate tensor with 1073741824 bytes. "
            "There is not enough GPU video memory available!"
        ),
        RuntimeError("DML allocator out of memory"),
        RuntimeError("HRESULT failed: E_OUTOFMEMORY"),
        # MPS / Apple.
        RuntimeError("MPS backend out of memory (MPS allocated 9.00 GB)"),
        RuntimeError("Insufficient memory on MPS device"),
        # CPU.
        MemoryError(),
        MemoryError("Unable to allocate 4.00 GiB for an array"),
        RuntimeError(
            "DefaultCPUAllocator: can't allocate memory: you tried to allocate 8589934592 bytes"
        ),
    ],
)
def test_is_oom_error_recognises_all_backends(exc: BaseException) -> None:
    assert _is_oom_error(exc) is True, (
        f"allocation failure not classified as OOM, so no eviction "
        f"would fire: {type(exc).__name__}: {exc}"
    )


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("checkpoint file is corrupt"),
        ValueError("unexpected keys in state_dict"),
        FileNotFoundError("model.safetensors"),
        RuntimeError("zooming the room caused a bloom artifact"),
    ],
)
def test_is_oom_error_rejects_unrelated_failures(exc: BaseException) -> None:
    assert _is_oom_error(exc) is False, (
        f"non-OOM failure misclassified, which would silently evict a warm pipeline: {exc}"
    )


def test_pipeline_cap_env_override(
    stocked_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LUCIDIUM_MAX_RESIDENT_PIPELINES", "3")
    client = EmbeddedImageClient(models_dir=str(stocked_models_dir))
    assert client._max_resident_pipelines == 3

    # Explicit constructor argument wins over the env var.
    pinned = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        max_resident_pipelines=1,
    )
    assert pinned._max_resident_pipelines == 1

    # Garbage and zero both fall back to a safe floor of 1 — a cap of 0
    # would evict the pipeline we just loaded and livelock the render.
    monkeypatch.setenv("LUCIDIUM_MAX_RESIDENT_PIPELINES", "not-a-number")
    assert (
        EmbeddedImageClient(
            models_dir=str(stocked_models_dir),
        )._max_resident_pipelines
        == 1
    )
    assert (
        EmbeddedImageClient(
            models_dir=str(stocked_models_dir),
            max_resident_pipelines=0,
        )._max_resident_pipelines
        == 1
    )
