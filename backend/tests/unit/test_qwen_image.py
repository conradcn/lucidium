"""Tests for the embedded backend's Qwen-Image support.

Covers three layers, all with fakes so the suite never imports the real
diffusers Qwen stack (torch is present in the venv but the heavy Qwen
pipelines are not exercised):

  * **Detection** — ``_is_qwen_model_path`` (filename sniff used by the
    factory) and ``_is_qwen_pipeline`` (class-name / config dispatch used
    at run time).
  * **Runtime recipe** — ``generate`` on a fake ``QwenImagePipeline``
    uses the Lightning few-step recipe by default (8 steps, CFG off via
    ``true_cfg_scale`` 1.0), plain-string prompts, no compel chunking.
  * **Character-change img2img** — ``regenerate_from_image`` declines
    (returns ``None``) on non-Qwen checkpoints and runs the injected
    img2img runner on Qwen ones; and ``assets._build_portrait_render``
    routes a change through that path when a prior portrait exists,
    falling back to a full render when the backend declines.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from lucidium.providers.embedded_image_client import (
    _CHARACTER_NEGATIVE,
    _QWEN_IMG2IMG_STRENGTH,
    EmbeddedImageClient,
    _is_qwen_model_path,
    _is_qwen_pipeline,
)

# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _FakePipeline:
    """Records call kwargs and returns a stub image of the requested size."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.config = type("Cfg", (), {"_name_or_path": "test-stub"})()

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        width = kwargs.get("width", 512)
        height = kwargs.get("height", 512)
        result = type("Result", (), {})()
        result.images = [Image.new("RGB", (width, height), color=(80, 120, 200))]
        return result


class _FakeQwenPipeline(_FakePipeline):
    """Stand-in for diffusers' ``QwenImagePipeline``. Runtime dispatch
    keys off ``type(pipeline).__name__`` so renaming the class is what
    makes ``_run_pipeline`` choose the Qwen branch."""


_FakeQwenPipeline.__name__ = "QwenImagePipeline"
_FakeQwenPipeline.__qualname__ = "QwenImagePipeline"


def _png_bytes(width: int, height: int, *, alpha: bool = False) -> bytes:
    mode = "RGBA" if alpha else "RGB"
    color = (0, 0, 0, 0) if alpha else (10, 20, 30)
    buf = io.BytesIO()
    Image.new(mode, (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def stocked_qwen_models_dir(tmp_path: Path) -> Path:
    """Models dir pre-seeded with a Qwen-named checkpoint so the filename
    sniff routes to the Qwen branch."""
    models = tmp_path / "models"
    models.mkdir()
    (models / "Qwen-Image.safetensors").write_bytes(b"fake")
    return models


@pytest.fixture
def stocked_sdxl_models_dir(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    (models / "some-sdxl-checkpoint.safetensors").write_bytes(b"fake")
    return models


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Qwen-Image.safetensors", True),
        ("qwen_image_fp8_e4m3fn.safetensors", True),
        ("MyQwenImageFinetune.safetensors", True),
        ("z-image-turbo.safetensors", False),
        ("sd_xl_base_1.0.safetensors", False),
        ("dreamshaper.safetensors", False),
    ],
)
def test_is_qwen_model_path(name: str, expected: bool) -> None:
    assert _is_qwen_model_path(Path(name)) is expected


def test_is_qwen_pipeline_class_name_match() -> None:
    assert _is_qwen_pipeline(_FakeQwenPipeline())
    # A non-Qwen fake with a neutral config must NOT match.
    assert not _is_qwen_pipeline(_FakePipeline())


def test_is_qwen_pipeline_config_substring_match() -> None:
    """A stub whose class isn't named QwenImage* still opts in via the
    ``config._name_or_path`` substring (mirrors the other detectors)."""
    stub = _FakePipeline()
    stub.config = type("Cfg", (), {"_name_or_path": "Qwen/Qwen-Image"})()
    assert _is_qwen_pipeline(stub)


# --------------------------------------------------------------------------
# Runtime recipe (text2img)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_qwen_uses_lightning_recipe_by_default(
    stocked_qwen_models_dir: Path,
) -> None:
    """A loaded ``QwenImagePipeline`` with no recipe flag samples with the
    Lightning few-step recipe (8 steps, CFG off via true_cfg_scale 1.0)
    and plain-string prompts (no compel chunking — Qwen's single long-
    context encoder doesn't take the SDXL dual-CLIP embeds path)."""
    fake = _FakeQwenPipeline()
    client = EmbeddedImageClient(
        models_dir=str(stocked_qwen_models_dir),
        pipeline_factory=lambda _path, _device: fake,
        bg_remover=lambda b: b,
    )

    await client.generate(
        "character.json",
        {"positive_prompt": "a knight in silver armor"},
        seed=99,
    )

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["num_inference_steps"] == 8
    assert call["true_cfg_scale"] == 1.0
    assert call["guidance_scale"] == 1.0
    assert call["prompt"] == "a knight in silver armor"
    assert _CHARACTER_NEGATIVE in call["negative_prompt"]
    # Plain-string path — must NOT have hit the compel chunker.
    assert "prompt_embeds" not in call
    assert "pooled_prompt_embeds" not in call


@pytest.mark.asyncio
async def test_qwen_uses_base_recipe_when_flagged(
    stocked_qwen_models_dir: Path,
) -> None:
    """When the Lightning LoRA couldn't be fused the pipeline is flagged
    for the base recipe (more steps + real CFG)."""
    import lucidium.providers.embedded_image_client as eic

    fake = _FakeQwenPipeline()
    setattr(fake, eic._QWEN_RECIPE_ATTR, eic._QWEN_BASE_RECIPE)
    client = EmbeddedImageClient(
        models_dir=str(stocked_qwen_models_dir),
        pipeline_factory=lambda _path, _device: fake,
        bg_remover=lambda b: b,
    )
    await client.generate(
        "character.json",
        {"positive_prompt": "a knight"},
        seed=1,
    )
    call = fake.calls[0]
    assert call["num_inference_steps"] == eic._QWEN_BASE_RECIPE[0]
    assert call["true_cfg_scale"] == eic._QWEN_BASE_RECIPE[1]


def test_qwen_recipe_default_is_lightning() -> None:
    import lucidium.providers.embedded_image_client as eic

    assert eic._qwen_recipe(_FakeQwenPipeline()) == eic._QWEN_LIGHTNING_RECIPE


@pytest.mark.parametrize(
    "name",
    [
        "Qwen-Image-Lightning-8steps.safetensors",
        "Qwen-Image-Distill.safetensors",  # the one-click download
        "qwen_image_distill_full_fp8.safetensors",
        "Qwen-Image-Turbo.safetensors",
    ],
)
def test_apply_qwen_lightning_lora_few_step_filename_skips_fetch(name: str) -> None:
    """A checkpoint already marked few-step (Lightning / distill / turbo /
    N-step) is used as-is: no LoRA fetch, few-step recipe."""
    import lucidium.providers.embedded_image_client as eic

    calls = []

    class _Pipe:
        def load_lora_weights(self, *a, **k):
            calls.append((a, k))

    assert eic._apply_qwen_lightning_lora(_Pipe(), Path(name)) is True
    assert calls == []  # never tried to fetch a LoRA


# --------------------------------------------------------------------------
# Character-change img2img (regenerate_from_image)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_regenerate_from_image_declines_on_non_qwen(
    stocked_sdxl_models_dir: Path,
) -> None:
    """On a non-Qwen checkpoint the img2img path is a no-op: it returns
    ``None`` WITHOUT loading a pipeline, so SDXL / Z-Image keep their
    existing full-render + face-inpaint behaviour."""
    loaded: list[Any] = []

    def factory(_path, _device):
        loaded.append(_path)
        return _FakePipeline()

    client = EmbeddedImageClient(
        models_dir=str(stocked_sdxl_models_dir),
        pipeline_factory=factory,
    )
    result = await client.regenerate_from_image(
        "character.json",
        _png_bytes(832, 1216, alpha=True),
        {"positive_prompt": "new outfit", "subject_kind": "human"},
        seed=3,
    )
    assert result is None
    # Declined before loading anything.
    assert loaded == []


@pytest.mark.asyncio
async def test_regenerate_from_image_requires_positive(
    stocked_qwen_models_dir: Path,
) -> None:
    """No positive prompt → decline (nothing to steer the edit toward)."""
    client = EmbeddedImageClient(
        models_dir=str(stocked_qwen_models_dir),
        pipeline_factory=lambda _p, _d: _FakeQwenPipeline(),
    )
    result = await client.regenerate_from_image(
        "character.json",
        _png_bytes(832, 1216),
        {"positive_prompt": "  "},
        seed=1,
    )
    assert result is None


@pytest.mark.asyncio
async def test_regenerate_from_image_runs_qwen_img2img(
    stocked_qwen_models_dir: Path,
) -> None:
    """On a Qwen checkpoint with a positive prompt, the injected img2img
    runner is invoked with the full prompt (face folded in), the tunable
    strength, and the workflow's portrait dimensions; the character
    workflow then runs background removal on the result."""
    runner_calls: list[dict[str, Any]] = []
    bg_calls: list[bytes] = []

    def fake_runner(pipeline, base_png, **kwargs):
        runner_calls.append({"pipeline": pipeline, "base_png": base_png, **kwargs})
        return _png_bytes(832, 1216)

    def fake_bg(image_bytes: bytes) -> bytes:
        bg_calls.append(image_bytes)
        return _png_bytes(832, 1216, alpha=True)

    client = EmbeddedImageClient(
        models_dir=str(stocked_qwen_models_dir),
        pipeline_factory=lambda _p, _d: _FakeQwenPipeline(),
        bg_remover=fake_bg,
        qwen_img2img_runner=fake_runner,
    )

    base = _png_bytes(832, 1216, alpha=True)
    result = await client.regenerate_from_image(
        "character.json",
        base,
        {
            "positive_prompt": "a knight, now in a red cloak",
            "face_prompt": "smiling",
            "subject_kind": "human",
        },
        seed=42,
    )

    assert result is not None
    assert len(runner_calls) == 1
    call = runner_calls[0]
    assert call["base_png"] == base
    # Face prompt folded into the body prompt (Qwen has no inpaint pass).
    assert call["positive"] == "a knight, now in a red cloak, smiling"
    assert _CHARACTER_NEGATIVE in call["negative"]
    assert (call["width"], call["height"]) == (832, 1216)
    assert call["seed"] == 42
    assert call["strength"] == _QWEN_IMG2IMG_STRENGTH
    # Character workflow re-cuts the background off the img2img output.
    assert len(bg_calls) == 1


@pytest.mark.asyncio
async def test_regenerate_from_image_declines_when_runner_returns_none(
    stocked_qwen_models_dir: Path,
) -> None:
    """When the img2img pipeline can't be built (runner returns None),
    the method declines so the caller full-renders — and never calls
    background removal on a missing image."""
    bg_calls: list[bytes] = []

    client = EmbeddedImageClient(
        models_dir=str(stocked_qwen_models_dir),
        pipeline_factory=lambda _p, _d: _FakeQwenPipeline(),
        bg_remover=lambda b: bg_calls.append(b) or b,
        qwen_img2img_runner=lambda *a, **k: None,
    )
    result = await client.regenerate_from_image(
        "character.json",
        _png_bytes(832, 1216),
        {"positive_prompt": "new outfit", "subject_kind": "human"},
        seed=5,
    )
    assert result is None
    assert bg_calls == []


# --------------------------------------------------------------------------
# Residency / offload strategy (_apply_qwen_offload)
# --------------------------------------------------------------------------
#
# Qwen-Image's dense transformer is ~40 GiB in bf16. It does NOT need to be
# held resident at that size: the loader stores it in fp8 (ComfyUI's trick,
# ~20 GiB → fits a 24 GiB card) and only streams at block level when even
# that won't fit. These tests pin which strategy _apply_qwen_offload picks
# per VRAM / capability, using fakes (no real diffusers offload machinery).


class _GroupOffloadModule:
    """Fake transformer / text-encoder exposing ``enable_group_offload``
    (and, when ``fp8`` is set, ``enable_layerwise_casting``)."""

    def __init__(self, *, fp8: bool = False) -> None:
        self.group_kwargs: dict[str, Any] | None = None
        self.fp8_kwargs: dict[str, Any] | None = None
        self.moved_to: Any = None
        if fp8:

            def _cast(**kwargs: Any) -> None:
                self.fp8_kwargs = kwargs

            self.enable_layerwise_casting = _cast  # type: ignore[attr-defined]

    def enable_group_offload(self, **kwargs: Any) -> None:
        self.group_kwargs = kwargs

    def to(self, device: Any) -> _GroupOffloadModule:
        self.moved_to = device
        return self


class _PlainModule:
    """Fake module WITHOUT group-offload or fp8-casting support (old
    diffusers build)."""

    def __init__(self) -> None:
        self.moved_to: Any = None

    def to(self, device: Any) -> _PlainModule:
        self.moved_to = device
        return self


class _OffloadPipe:
    def __init__(self, *, group_capable: bool = True, fp8_capable: bool = False) -> None:
        def mk():
            if not group_capable:
                return _PlainModule()
            return _GroupOffloadModule(fp8=fp8_capable)

        self.transformer = mk()
        self.text_encoder = mk()
        self.vae = _GroupOffloadModule()
        self.model_offload = False
        self.seq_offload = False

    def enable_model_cpu_offload(self, device: Any = None) -> None:
        self.model_offload = True

    def enable_sequential_cpu_offload(self, device: Any = None) -> None:
        self.seq_offload = True


def test_apply_qwen_offload_cpu_encode_on_24gb(monkeypatch) -> None:
    """The 4090 path: fp8 transformer (here via layerwise casting, since
    torchao can't quantize the fake), then CPU-encode-resident placement —
    transformer resident on the GPU, text encoder parked on the CPU — NOT
    model offload or block streaming."""
    import torch

    import lucidium.providers.embedded_image_client as eic

    monkeypatch.setattr(eic, "detect_total_vram_gb", lambda: 24.0)
    pipe = _OffloadPipe(group_capable=True, fp8_capable=True)
    eic._apply_qwen_offload(pipe, "cuda", "cuda")

    # fp8 applied (torchao declines on the fake → layerwise casting).
    assert pipe.transformer.fp8_kwargs is not None
    assert pipe.transformer.fp8_kwargs["storage_dtype"] == torch.float8_e4m3fn
    # CPU-encode-resident: flagged, transformer resident, te on CPU.
    assert getattr(pipe, eic._QWEN_CPU_ENCODE_ATTR, False) is True
    assert pipe.transformer.moved_to == "cuda"
    assert pipe.text_encoder.moved_to == "cpu"
    assert pipe.model_offload is False
    assert pipe.transformer.group_kwargs is None


def test_apply_qwen_offload_streams_when_no_fp8_on_tight_vram(monkeypatch) -> None:
    """Without fp8 support the bf16 transformer is ~40 GiB, so a 24 GiB
    card can't hold it resident → block-level streaming (VAE resident)."""
    import lucidium.providers.embedded_image_client as eic

    monkeypatch.setattr(eic, "detect_total_vram_gb", lambda: 24.0)
    pipe = _OffloadPipe(group_capable=True, fp8_capable=False)
    eic._apply_qwen_offload(pipe, "cuda", "cuda")

    assert pipe.transformer.group_kwargs is not None
    assert pipe.transformer.group_kwargs["offload_type"] == "block_level"
    assert pipe.text_encoder.group_kwargs is not None
    assert pipe.vae.moved_to == "cuda"  # small VAE kept resident
    assert pipe.model_offload is False
    assert pipe.seq_offload is False


def test_apply_qwen_offload_non_cuda_fp8_uses_model_offload(monkeypatch) -> None:
    """The CPU-encode-resident path is CUDA-only. On another accelerator
    (e.g. XPU) with fp8 + resident-class VRAM, fall back to component-level
    model_cpu_offload, not CPU-encode."""
    import lucidium.providers.embedded_image_client as eic

    monkeypatch.setattr(eic, "detect_total_vram_gb", lambda: 80.0)
    pipe = _OffloadPipe(group_capable=True, fp8_capable=True)
    eic._apply_qwen_offload(pipe, "xpu", "xpu")

    assert getattr(pipe, eic._QWEN_CPU_ENCODE_ATTR, False) is False
    assert pipe.model_offload is True
    assert pipe.transformer.group_kwargs is None


def test_apply_qwen_offload_falls_back_to_sequential(monkeypatch) -> None:
    """When the diffusers build lacks group offload, a tight card falls
    back to sequential CPU offload (fits the smallest cards)."""
    import lucidium.providers.embedded_image_client as eic

    monkeypatch.setattr(eic, "detect_total_vram_gb", lambda: 12.0)
    pipe = _OffloadPipe(group_capable=False)
    eic._apply_qwen_offload(pipe, "cuda", "cuda")

    assert pipe.seq_offload is True
    assert pipe.model_offload is False


# --------------------------------------------------------------------------
# img2img from_pipe dtype restoration (_restore_qwen_transformer_param_dtype)
# --------------------------------------------------------------------------
#
# diffusers' ``from_pipe`` upcasts every non-fp8 param of the SHARED Qwen
# transformer to float32; torchao's fp8 ``addmm`` then rejects the float32
# bias ("Bias must be BFloat16 or Half, but got Float") and the float32
# latents fall off the fp8 fast path. The restore walk casts the plain
# float32 params back to bf16 but must NEVER ``.to`` the fp8 weight tensors
# (that strands the weight scale → a ``_scaled_mm`` error).


def test_restore_qwen_transformer_param_dtype_casts_f32_skips_fp8() -> None:
    import torch

    import lucidium.providers.embedded_image_client as eic

    class _Param:
        def __init__(self, data) -> None:
            self.data = data

    class Float8Tensor:
        """Stand-in for a torchao quantized weight: its class name matches
        the skip-list exactly, and it even *reports* float32 to prove the
        name guard (not just the dtype filter) protects it from a re-cast."""

        def __init__(self) -> None:
            self.dtype = torch.float32
            self.to_called = False

        def to(self, *_a, **_k):
            self.to_called = True
            return self

    f32_bias = _Param(torch.zeros(4, dtype=torch.float32))
    bf16_already = _Param(torch.zeros(4, dtype=torch.bfloat16))
    fp8_weight = _Param(Float8Tensor())

    class _Transformer:
        def parameters(self, recurse: bool = True):
            return [f32_bias, bf16_already, fp8_weight]

    eic._restore_qwen_transformer_param_dtype(_Transformer(), torch.bfloat16)

    # float32 bias restored to bf16; already-bf16 left alone.
    assert f32_bias.data.dtype == torch.bfloat16
    assert bf16_already.data.dtype == torch.bfloat16
    # The fp8 weight is skipped entirely — never re-cast, never replaced.
    assert isinstance(fp8_weight.data, Float8Tensor)
    assert fp8_weight.data.to_called is False


def test_restore_qwen_transformer_param_dtype_tolerates_missing() -> None:
    """No transformer / a stub without ``parameters`` is a quiet no-op
    (test stubs and the CPU-only path land here)."""
    import torch

    import lucidium.providers.embedded_image_client as eic

    eic._restore_qwen_transformer_param_dtype(None, torch.bfloat16)
    eic._restore_qwen_transformer_param_dtype(object(), torch.bfloat16)


# --------------------------------------------------------------------------
# assets._build_portrait_render routing
# --------------------------------------------------------------------------


def _fake_session(workflow: str = "character.json") -> Any:
    return SimpleNamespace(
        settings=SimpleNamespace(image=SimpleNamespace(portrait_workflow=workflow))
    )


def _fake_character(images: list[Any], *, seed: int = 7) -> Any:
    return SimpleNamespace(
        id="c1",
        seed=seed,
        kind=SimpleNamespace(value="human"),
        images=images,
    )


class _RoutingClient:
    """Minimal ImageClient stand-in recording which path the portrait
    render closure took."""

    def __init__(self, *, img2img_result: bytes | None) -> None:
        self._img2img_result = img2img_result
        self.generate_calls: list[dict[str, Any]] = []
        self.img2img_calls: list[dict[str, Any]] = []

    async def generate(self, workflow, params, *, seed):
        self.generate_calls.append({"workflow": workflow, "params": params, "seed": seed})
        return b"FULL-RENDER"

    async def regenerate_from_image(self, workflow, base_png, params, *, seed):
        self.img2img_calls.append(
            {"workflow": workflow, "base_png": base_png, "params": params, "seed": seed}
        )
        return self._img2img_result


@pytest.mark.asyncio
async def test_build_portrait_render_routes_change_through_img2img(tmp_path: Path) -> None:
    """With a prior portrait on disk and a backend exposing
    ``regenerate_from_image``, a change renders via img2img (full new
    prompt), NOT a fresh generate."""
    from lucidium.orchestration.assets import _build_portrait_render

    prior = tmp_path / "prior.png"
    prior.write_bytes(_png_bytes(832, 1216, alpha=True))
    character = _fake_character([SimpleNamespace(path=str(prior), identity_hash="")])
    client = _RoutingClient(img2img_result=b"IMG2IMG")

    render = _build_portrait_render(
        session=_fake_session(),
        image_client=client,
        character=character,
        positive="a knight in a red cloak",
        face="smiling",
        negative_extras="",
        identity_hash="ID",
    )
    result = await render()

    assert result == b"IMG2IMG"
    assert len(client.img2img_calls) == 1
    img2img = client.img2img_calls[0]
    assert img2img["base_png"] == prior.read_bytes()
    assert img2img["params"]["positive_prompt"] == "a knight in a red cloak"
    assert img2img["seed"] == 7
    # No full render when img2img succeeds.
    assert client.generate_calls == []


@pytest.mark.asyncio
async def test_build_portrait_render_falls_back_when_img2img_declines(
    tmp_path: Path,
) -> None:
    """When ``regenerate_from_image`` declines (returns None — e.g. a
    non-Qwen backend), the closure falls back to a full generate."""
    from lucidium.orchestration.assets import _build_portrait_render

    prior = tmp_path / "prior.png"
    prior.write_bytes(_png_bytes(832, 1216, alpha=True))
    character = _fake_character([SimpleNamespace(path=str(prior), identity_hash="")])
    client = _RoutingClient(img2img_result=None)

    render = _build_portrait_render(
        session=_fake_session(),
        image_client=client,
        character=character,
        positive="a knight in a red cloak",
        face="smiling",
        negative_extras="",
        identity_hash="ID",
    )
    result = await render()

    assert result == b"FULL-RENDER"
    assert len(client.img2img_calls) == 1
    assert len(client.generate_calls) == 1


@pytest.mark.asyncio
async def test_build_portrait_render_first_render_is_full_generate(
    tmp_path: Path,
) -> None:
    """A character with no prior portrait must full-render (text2img):
    there's no base image to img2img from."""
    from lucidium.orchestration.assets import _build_portrait_render

    character = _fake_character([])
    client = _RoutingClient(img2img_result=b"IMG2IMG")

    render = _build_portrait_render(
        session=_fake_session(),
        image_client=client,
        character=character,
        positive="a knight",
        face="neutral",
        negative_extras="",
        identity_hash="ID",
    )
    result = await render()

    assert result == b"FULL-RENDER"
    assert client.img2img_calls == []
    assert len(client.generate_calls) == 1
