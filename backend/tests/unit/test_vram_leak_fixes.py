"""Pin the VRAM-leak mitigations in ``embedded_image_client``.

Three fixes covered here:

  1. Long-prompt encoding registers NO forward hooks. This used
     to go through ``compel``, which captured SDXL's pooled output
     via a forward hook on ``text_encoder_2`` that it never
     removed — so a per-render Compel meant a per-render hook, each
     pinning a captured tensor in VRAM, and the mitigation was an
     awkward per-pipeline instance cache. ``providers.clip_long_
     prompt`` reads the pooled output off the encoder's return value
     instead, so there is no hook to leak and nothing to cache.

  2. The face-inpaint pipeline is cached per text2img pipeline.
     The wrapper registers config metadata on construction; once
     per pipeline is enough.

  3. ``restore_to_gpu`` drops a pipeline from cache when the
     ``.to(device)`` move fails. Without this, a stranded pipeline
     stays in ``_pipelines`` with its components on CPU and the
     next render trips on "tensors on different devices."
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

import pytest

from lucidium.providers.embedded_image_client import (
    _INPAINT_CACHE_ATTR,
    _VAE_MEMORY_OPT_ATTR,
    EmbeddedImageClient,
    _apply_vae_memory_optimizations,
    _encode_long_prompt,
    _get_or_build_inpaint_pipeline,
)

# ---------- Stub objects -----------------------------------------------------


class _StubEncoder:
    """Stand-in for an SDXL text encoder. Tracks every forward
    hook registered against it so the tests can assert that
    long-prompt encoding registers none at all."""

    def __init__(self) -> None:
        self.hooks: list[Any] = []

    def register_forward_hook(self, hook: Any) -> Any:
        self.hooks.append(hook)
        return _RemovableHandle(self, hook)


class _RemovableHandle:
    def __init__(self, owner: _StubEncoder, hook: Any) -> None:
        self._owner = owner
        self._hook = hook

    def remove(self) -> None:
        try:
            self._owner.hooks.remove(self._hook)
        except ValueError:
            pass


class _StubPipeline:
    """Minimal SDXL-shaped object with the attributes the
    long-prompt encoder and the inpaint wrapper inspect. Tests
    don't run real forward passes; they probe wiring only."""

    def __init__(self) -> None:
        self.tokenizer = object()
        self.tokenizer_2 = object()
        self.text_encoder = _StubEncoder()
        self.text_encoder_2 = _StubEncoder()
        self.unet = object()
        self.vae = object()
        self.scheduler = _StubScheduler()


class _StubScheduler:
    config: ClassVar[dict[str, int]] = {"steps": 25}

    @classmethod
    def from_config(cls, _cfg: Any) -> _StubScheduler:
        return cls()


# ---------- Long-prompt encoding registers no hooks --------------------------


def test_long_prompt_encoding_registers_no_forward_hooks() -> None:
    """The leak this replaces: under the old compel-backed shape,
    every ``_encode_long_prompt`` call constructed a Compel, which
    registered a forward hook on ``text_encoder_2`` that was never
    removed — 50 renders meant 50 hooks, each pinning a captured
    pooled-output tensor in VRAM.

    ``clip_long_prompt`` reads the pooled output off the encoder's
    return value, so no call registers a hook, no matter how many
    times it runs. The encoders here raise on any real forward, which
    is fine: the encoder swallows failures and returns ``None``, and
    what is being pinned is the hook count, not the tensors.
    """
    pipeline = _StubPipeline()
    for _ in range(50):
        _encode_long_prompt(pipeline, "a portrait", "blurry")

    assert pipeline.text_encoder.hooks == []
    assert pipeline.text_encoder_2.hooks == []


def test_long_prompt_returns_none_for_non_sdxl_pipeline() -> None:
    """SD 1.5-shaped pipelines (no tokenizer_2 / text_encoder_2)
    fall through to plain prompts, which is the existing
    contract."""

    class _Sd15:
        tokenizer = object()
        text_encoder = object()
        # No tokenizer_2 / text_encoder_2.

    assert _encode_long_prompt(_Sd15(), "a portrait", "blurry") is None


# ---------- Inpaint cache pin ------------------------------------------------


def test_inpaint_pipeline_is_cached_per_text2img(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin: ``_get_or_build_inpaint_pipeline`` returns the same
    instance on repeat calls."""

    constructions: list[None] = []

    class _FakeInpaint:
        def __init__(self, **_kwargs: Any) -> None:
            constructions.append(None)

    fake_diffusers = type(
        "fake",
        (),
        {"StableDiffusionXLInpaintPipeline": _FakeInpaint},
    )
    import sys

    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    pipeline = _StubPipeline()
    a = _get_or_build_inpaint_pipeline(pipeline)
    b = _get_or_build_inpaint_pipeline(pipeline)
    c = _get_or_build_inpaint_pipeline(pipeline)

    assert a is b is c
    assert len(constructions) == 1
    assert getattr(pipeline, _INPAINT_CACHE_ATTR) is a


def test_inpaint_cache_isolated_per_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two separate text2img pipelines get two separate inpaint
    wrappers — no cross-contamination."""

    class _FakeInpaint:
        def __init__(self, **kwargs: Any) -> None:
            self.unet_id = id(kwargs["unet"])

    fake_diffusers = type(
        "fake",
        (),
        {"StableDiffusionXLInpaintPipeline": _FakeInpaint},
    )
    import sys

    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    pipe_a = _StubPipeline()
    pipe_b = _StubPipeline()
    inpaint_a = _get_or_build_inpaint_pipeline(pipe_a)
    inpaint_b = _get_or_build_inpaint_pipeline(pipe_b)

    assert inpaint_a is not inpaint_b
    assert inpaint_a.unet_id == id(pipe_a.unet)
    assert inpaint_b.unet_id == id(pipe_b.unet)


# ---------- VAE memory optimization pin -------------------------------------


class _PipelineWithVaeOpts:
    """Stub that records every VAE optimization toggle call so
    tests can assert idempotency + shape."""

    def __init__(self) -> None:
        self.vae = object()
        self.tiling_calls = 0
        self.slicing_calls = 0

    def enable_vae_tiling(self) -> None:
        self.tiling_calls += 1

    def enable_vae_slicing(self) -> None:
        self.slicing_calls += 1


def test_apply_vae_memory_optimizations_enables_both_toggles() -> None:
    pipeline = _PipelineWithVaeOpts()
    _apply_vae_memory_optimizations(pipeline)
    assert pipeline.tiling_calls == 1
    assert pipeline.slicing_calls == 1
    assert getattr(pipeline, _VAE_MEMORY_OPT_ATTR) is True


def test_apply_vae_memory_optimizations_is_idempotent() -> None:
    """Repeated calls (e.g. the per-render generate path runs
    this every time) are no-ops once the flag is set. Without
    idempotency, toggling tiling N times across a session would
    pile up diffusers internal hooks."""
    pipeline = _PipelineWithVaeOpts()
    for _ in range(10):
        _apply_vae_memory_optimizations(pipeline)
    assert pipeline.tiling_calls == 1
    assert pipeline.slicing_calls == 1


def test_apply_vae_memory_optimizations_handles_missing_methods() -> None:
    """An older diffusers release without ``enable_vae_tiling``
    must NOT crash the pipeline factory — the optimization is a
    nice-to-have, not a hard requirement."""

    class _OldPipeline:
        vae = object()
        # Neither enable_vae_tiling nor enable_vae_slicing.

    # No exception expected.
    _apply_vae_memory_optimizations(_OldPipeline())


def test_apply_vae_memory_optimizations_skips_when_no_vae() -> None:
    """Test stubs and recorded-fixture clients without a VAE
    attribute fall through cleanly."""

    class _NoVae:
        pass

    _apply_vae_memory_optimizations(_NoVae())


def test_inpaint_pipeline_shares_weight_modules_with_text2img(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the no-double-load invariant: the inpaint pipeline
    holds Python ``is`` identity to text2img's UNet, VAE, and
    text encoders. Diffusers' ``register_modules`` is a plain
    ``setattr`` — no cloning, no .to() copy — so all six heavy
    modules live in VRAM exactly once.

    Without this pin a future refactor (e.g. switching to a
    factory that loads inpaint weights from disk) could silently
    double VRAM usage and the user would only notice from a
    mysterious render-time spike."""

    class _FakeInpaint:
        def __init__(self, **kwargs: Any) -> None:
            for name, mod in kwargs.items():
                setattr(self, name, mod)

    fake_diffusers = type(
        "fake",
        (),
        {"StableDiffusionXLInpaintPipeline": _FakeInpaint},
    )
    import sys

    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    text2img = _StubPipeline()
    inpaint = _get_or_build_inpaint_pipeline(text2img)

    # All five weight-bearing modules + both tokenizers share
    # Python identity with the text2img pipeline. Identity is the
    # right check here because nn.Module __eq__ falls back to
    # ``is``, and diffusers' register_modules doesn't clone.
    assert inpaint.vae is text2img.vae
    assert inpaint.unet is text2img.unet
    assert inpaint.text_encoder is text2img.text_encoder
    assert inpaint.text_encoder_2 is text2img.text_encoder_2
    assert inpaint.tokenizer is text2img.tokenizer
    assert inpaint.tokenizer_2 is text2img.tokenizer_2
    # Scheduler is the ONLY component intentionally cloned —
    # sharing it leaks step_index state between text2img and
    # inpaint and produces an off-by-one IndexError on the
    # final denoise step. See _build_inpaint_pipeline for the
    # full incident note.
    assert inpaint.scheduler is not text2img.scheduler


# ---------- restore_to_gpu fallback pin --------------------------------------


@pytest.mark.asyncio
async def test_restore_to_gpu_drops_pipeline_when_to_device_fails() -> None:
    """When ``.to(device)`` raises (e.g. OOM during restore from
    CPU), the pipeline must be dropped from cache instead of
    left half-on-GPU. Otherwise the next ``generate()`` would
    invoke a partially-placed pipeline and trip a confusing
    "tensors on different devices" error."""

    class _Pipe:
        def __init__(self) -> None:
            self.to_calls: list[str] = []

        def to(self, dev: str) -> _Pipe:
            self.to_calls.append(dev)
            if dev == "cuda":
                raise RuntimeError("CUDA out of memory")
            return self

    pipe = _Pipe()
    path = Path("a.safetensors")
    client = EmbeddedImageClient(
        models_dir=".",
        pipeline_factory=lambda *_a, **_kw: pipe,
        bg_remover=lambda b: b,
        device="cuda",
    )
    # Wire the pipeline as if it had been loaded then evicted.
    client._pipelines[path] = pipe
    client._evicted.add(path)
    client._inference_locks[path] = asyncio.Lock()

    client.restore_to_gpu()

    # Pipeline dropped from cache so the next generate cold-loads.
    assert path not in client._pipelines
    # Bookkeeping cleaned up so we don't leak the eviction marker.
    assert path not in client._evicted
    assert path not in client._inference_locks


@pytest.mark.asyncio
async def test_restore_to_gpu_keeps_pipeline_on_success() -> None:
    """Counterpart: the happy path doesn't drop the pipeline."""

    class _Pipe:
        def to(self, _dev: str) -> _Pipe:
            return self

    pipe = _Pipe()
    path = Path("a.safetensors")
    client = EmbeddedImageClient(
        models_dir=".",
        pipeline_factory=lambda *_a, **_kw: pipe,
        bg_remover=lambda b: b,
        device="cuda",
    )
    client._pipelines[path] = pipe
    client._evicted.add(path)

    client.restore_to_gpu()

    assert path in client._pipelines
    assert path not in client._evicted  # eviction cleared
