"""Production-path tests for the embedded image factory.

The existing ``test_embedded_image_client.py`` tests stub out the
pipeline factory entirely (``pipeline_factory=lambda _path, _device:
fake``) so the routing decision in ``_default_pipeline_factory``
itself never runs. That left a gap the user found: a regression in
the real factory — e.g. forgetting to pass ``tokenizer`` to
``ZImagePipeline.from_single_file``, or forgetting to swap the SDXL
scheduler — wouldn't break a single test, but would break every
real render in the app.

These tests drive ``_default_pipeline_factory`` directly with the
heavy diffusers/transformers APIs monkeypatched. They prove:

  * Z-Image filename sniff routes to ``_load_z_image_pipeline``;
  * the helper pre-loads the Qwen3 text encoder + tokenizer + VAE
    from the upstream HF repo;
  * those three components are handed to
    ``ZImagePipeline.from_single_file`` so the loader doesn't fall
    back to its broken "infer text encoder from checkpoint" branch
    (which raised ``SingleFileComponentError: Failed to load
    Qwen3Model`` in the app);
  * the SDXL fallback still works for non-Z-Image filenames and
    still swaps the scheduler to ``EulerAncestralDiscreteScheduler``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# NOTE: ``transformers`` ships as a ``_LazyModule`` that replaces
# itself in ``sys.modules`` after ``__init__`` runs. The reference
# bound by ``import transformers`` at the top of a test file is the
# ORIGINAL package object, not the lazy proxy that
# ``from transformers import X`` will hit at call time. Patching the
# local reference has no effect on production code. Always go through
# ``sys.modules['transformers']`` / ``sys.modules['diffusers']`` so
# the monkeypatch lands on the same module the loader will look up.
diffusers = sys.modules["diffusers"] if "diffusers" in sys.modules else None
transformers = sys.modules["transformers"] if "transformers" in sys.modules else None
if diffusers is None:
    import diffusers as diffusers  # type: ignore[no-redef]
if transformers is None:
    import transformers as transformers  # type: ignore[no-redef]
# Re-bind to whatever is in sys.modules after the imports settle —
# transformers may have swapped itself for its lazy proxy on import.
diffusers = sys.modules["diffusers"]
transformers = sys.modules["transformers"]

from lucidium.providers.embedded_image_client import (  # noqa: E402
    _default_pipeline_factory,
    _load_z_image_pipeline,
)


class _RecordingPipeline:
    """Stand-in for a diffusers pipeline. Records every ``.to(device)``
    and every ``.enable_model_cpu_offload`` call so the device-
    placement contract stays under test for both pipeline families.
    """

    def __init__(self) -> None:
        self.moved_to: list[str] = []
        self.offload_calls: list[dict[str, Any]] = []
        self.scheduler = MagicMock(name="scheduler")
        # The scheduler swap path reads ``scheduler.config``; supply a
        # plain object so ``EulerAncestralDiscreteScheduler.from_config``
        # has something to receive.
        self.scheduler.config = object()

    def to(self, device: str) -> _RecordingPipeline:
        self.moved_to.append(device)
        return self

    def enable_model_cpu_offload(self, *args: Any, **kwargs: Any) -> None:
        self.offload_calls.append({"args": args, "kwargs": kwargs})


def _make_safetensors(tmp_path: Path, name: str) -> Path:
    """Create a stub safetensors file with the given name. The
    factory's filename sniff is the only thing that touches the
    contents (and it doesn't), so empty bytes are fine."""
    p = tmp_path / name
    p.write_bytes(b"stub")
    return p


# ---------------------------------------------------------------------------
# Z-Image branch
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_z_image_loaders(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Monkeypatch every ``from_pretrained`` / ``from_single_file``
    call ``_load_z_image_pipeline`` makes. Returns a dict so each
    test can assert on the specific call it cares about.
    """
    auto_model = MagicMock(name="AutoModel")
    text_encoder = MagicMock(name="text_encoder")
    auto_model.from_pretrained.return_value = text_encoder

    auto_tokenizer = MagicMock(name="AutoTokenizer")
    tokenizer = MagicMock(name="tokenizer")
    auto_tokenizer.from_pretrained.return_value = tokenizer

    autoencoder = MagicMock(name="AutoencoderKL")
    vae = MagicMock(name="vae")
    autoencoder.from_pretrained.return_value = vae

    z_image_pipeline_cls = MagicMock(name="ZImagePipeline")
    pipeline_instance = _RecordingPipeline()
    z_image_pipeline_cls.from_single_file.return_value = pipeline_instance

    # Pre-touch the lazy-module attributes so they're materialised
    # in ``sys.modules['transformers'].__dict__`` before monkeypatch
    # tries to swap them. Without this, setattr against a not-yet-
    # realised lazy attribute lands on the wrong slot and the real
    # class stays reachable through ``from transformers import X``.
    _ = sys.modules["transformers"].AutoModel
    _ = sys.modules["transformers"].AutoTokenizer
    _ = sys.modules["diffusers"].AutoencoderKL
    _ = sys.modules["diffusers"].ZImagePipeline

    monkeypatch.setattr(
        sys.modules["transformers"],
        "AutoModel",
        auto_model,
    )
    monkeypatch.setattr(
        sys.modules["transformers"],
        "AutoTokenizer",
        auto_tokenizer,
    )
    monkeypatch.setattr(
        sys.modules["diffusers"],
        "AutoencoderKL",
        autoencoder,
    )
    monkeypatch.setattr(
        sys.modules["diffusers"],
        "ZImagePipeline",
        z_image_pipeline_cls,
    )

    return {
        "auto_model": auto_model,
        "auto_tokenizer": auto_tokenizer,
        "autoencoder": autoencoder,
        "z_image_pipeline_cls": z_image_pipeline_cls,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "vae": vae,
        "pipeline_instance": pipeline_instance,
    }


def test_factory_routes_z_image_filename_to_z_image_loader(
    tmp_path: Path,
    patched_z_image_loaders: dict[str, MagicMock],
) -> None:
    """The filename sniff that the app relies on: any safetensors
    whose name contains the Z-Image marker MUST land in the
    Z-Image-shaped load path. A regression that loses the sniff
    would silently fall back to SDXL and the app would explode
    later inside diffusers.

    Pin: the factory does NOT eagerly ``.to(device)`` Z-Image
    pipelines — ``enable_model_cpu_offload`` manages device placement
    component-by-component, and a stray ``.to('cuda')`` after the
    offload hook is installed pins every component back on GPU and
    silently undoes the VRAM-savings the offload is FOR.
    """
    # ``device="cuda"`` puts us on the offload-enabled branch (the
    # CPU device suppresses offload activation by design — no GPU
    # to rotate components through).
    p = _make_safetensors(tmp_path, "zImageTurbo_turbo.safetensors")
    result = _default_pipeline_factory(p, device="cuda")

    z = patched_z_image_loaders
    assert z["z_image_pipeline_cls"].from_single_file.called, (
        "ZImagePipeline.from_single_file was never invoked — the "
        "filename sniff or the loader branch is broken"
    )
    assert result is z["pipeline_instance"]
    assert z["pipeline_instance"].moved_to == [], (
        "factory called pipeline.to(...) on the Z-Image branch — "
        "this overrides enable_model_cpu_offload and puts peak VRAM "
        "back over 24 GiB"
    )
    # enable_model_cpu_offload MUST have fired exactly once for the
    # offload to take effect.
    assert len(z["pipeline_instance"].offload_calls) == 1, (
        "expected enable_model_cpu_offload to be called once on the "
        f"Z-Image pipeline; got {len(z['pipeline_instance'].offload_calls)}"
    )


def test_z_image_skips_offload_on_cpu_only_install(
    tmp_path: Path,
    patched_z_image_loaders: dict[str, MagicMock],
) -> None:
    """CPU-only installs (no CUDA) have nothing to offload TO.
    ``enable_model_cpu_offload`` would either no-op or raise without
    a CUDA device, so the loader skips it entirely on the CPU path.
    """
    p = _make_safetensors(tmp_path, "zImageTurbo_turbo.safetensors")
    _default_pipeline_factory(p, device="cpu")

    z = patched_z_image_loaders
    assert z["pipeline_instance"].offload_calls == [], (
        "enable_model_cpu_offload should not fire on a CPU-only "
        "install — there's no accelerator to rotate components through"
    )


def test_z_image_loader_preloads_text_encoder_tokenizer_and_vae(
    tmp_path: Path,
    patched_z_image_loaders: dict[str, MagicMock],
) -> None:
    """Pin every component the Z-Image loader has to pre-load. The
    app's failure log lived right here: forgetting one of these
    arguments triggers ``SingleFileComponentError`` deep inside
    diffusers, which the factory wraps as
    ``ProviderUnreachableError`` for the renderer.

    Each from_pretrained call must target the canonical HF repo
    (``Tongyi-MAI/Z-Image-Turbo``) and the right subfolder, otherwise
    diffusers would try to dispatch by config-class name and fail
    to find the right weights.
    """
    p = _make_safetensors(tmp_path, "z-image-turbo.safetensors")
    _load_z_image_pipeline(p, device="cpu")

    z = patched_z_image_loaders

    # text_encoder — Qwen3Model lives under the ``text_encoder``
    # subfolder of the HF repo.
    z["auto_model"].from_pretrained.assert_called_once()
    args, kwargs = z["auto_model"].from_pretrained.call_args
    assert args[0] == "Tongyi-MAI/Z-Image-Turbo"
    assert kwargs.get("subfolder") == "text_encoder"

    # tokenizer
    z["auto_tokenizer"].from_pretrained.assert_called_once()
    args, kwargs = z["auto_tokenizer"].from_pretrained.call_args
    assert args[0] == "Tongyi-MAI/Z-Image-Turbo"
    assert kwargs.get("subfolder") == "tokenizer"

    # VAE
    z["autoencoder"].from_pretrained.assert_called_once()
    args, kwargs = z["autoencoder"].from_pretrained.call_args
    assert args[0] == "Tongyi-MAI/Z-Image-Turbo"
    assert kwargs.get("subfolder") == "vae"

    # from_single_file must receive ALL three pre-loaded components.
    # This is the kwarg set that fixed the app's CLIPTextModel /
    # Qwen3Model load failures.
    z["z_image_pipeline_cls"].from_single_file.assert_called_once()
    args, kwargs = z["z_image_pipeline_cls"].from_single_file.call_args
    assert args[0] == str(p)
    assert kwargs.get("text_encoder") is z["text_encoder"], (
        "ZImagePipeline.from_single_file must receive the preloaded "
        "text_encoder; without it the loader falls back to the broken "
        "in-checkpoint branch and raises SingleFileComponentError"
    )
    assert kwargs.get("tokenizer") is z["tokenizer"]
    assert kwargs.get("vae") is z["vae"]


def test_z_image_loader_uses_bfloat16_when_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patched_z_image_loaders: dict[str, MagicMock],
) -> None:
    """Z-Image's transformer is bf16-trained. Loading at fp16
    silently underflows. Pin that the loader asks for bf16 when the
    GPU supports it."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    p = _make_safetensors(tmp_path, "zImageTurbo_turbo.safetensors")
    _load_z_image_pipeline(p, device="cuda")

    z = patched_z_image_loaders
    expected_dtype = torch.bfloat16
    for mock_name in ("auto_model", "autoencoder"):
        _, kwargs = z[mock_name].from_pretrained.call_args
        assert kwargs.get("torch_dtype") == expected_dtype, (
            f"{mock_name} loaded with dtype={kwargs.get('torch_dtype')}; "
            "expected bfloat16 for Z-Image"
        )
    _, kwargs = z["z_image_pipeline_cls"].from_single_file.call_args
    assert kwargs.get("torch_dtype") == expected_dtype


# ---------------------------------------------------------------------------
# SDXL branch — make sure the fallback still works AND still applies
# the EulerAncestral scheduler swap the embedded backend depends on.
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_sdxl_loaders(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    sdxl_cls = MagicMock(name="StableDiffusionXLPipeline")
    pipeline_instance = _RecordingPipeline()
    sdxl_cls.from_single_file.return_value = pipeline_instance

    scheduler_cls = MagicMock(name="EulerAncestralDiscreteScheduler")
    swapped_scheduler = MagicMock(name="swapped_scheduler")
    scheduler_cls.from_config.return_value = swapped_scheduler

    # Materialise the lazy attrs before patching (same lazy-module
    # gotcha as the Z-Image fixture).
    _ = sys.modules["diffusers"].StableDiffusionXLPipeline
    _ = sys.modules["diffusers"].EulerAncestralDiscreteScheduler
    _ = sys.modules["diffusers"].ZImagePipeline

    monkeypatch.setattr(
        sys.modules["diffusers"],
        "StableDiffusionXLPipeline",
        sdxl_cls,
    )
    monkeypatch.setattr(
        sys.modules["diffusers"],
        "EulerAncestralDiscreteScheduler",
        scheduler_cls,
    )

    # Z-Image classes should NOT be touched on the SDXL path; make
    # them raise so a routing regression is loud.
    z_image_cls = MagicMock(name="ZImagePipeline")
    z_image_cls.from_single_file.side_effect = AssertionError(
        "SDXL path must not call ZImagePipeline.from_single_file"
    )
    monkeypatch.setattr(sys.modules["diffusers"], "ZImagePipeline", z_image_cls)

    return {
        "sdxl_cls": sdxl_cls,
        "scheduler_cls": scheduler_cls,
        "swapped_scheduler": swapped_scheduler,
        "pipeline_instance": pipeline_instance,
    }


def test_factory_falls_through_to_sdxl_for_non_z_image_filename(
    tmp_path: Path,
    patched_sdxl_loaders: dict[str, MagicMock],
) -> None:
    p = _make_safetensors(tmp_path, "dreamshaperXL10_alpha2Xl10.safetensors")
    result = _default_pipeline_factory(p, device="cpu")

    s = patched_sdxl_loaders
    s["sdxl_cls"].from_single_file.assert_called_once()
    args, _kwargs = s["sdxl_cls"].from_single_file.call_args
    assert args[0] == str(p)
    assert result is s["pipeline_instance"]
    assert s["pipeline_instance"].moved_to == ["cpu"]


def test_sdxl_branch_swaps_scheduler_to_euler_ancestral(
    tmp_path: Path,
    patched_sdxl_loaders: dict[str, MagicMock],
) -> None:
    """The SDXL workflow JSON pins ``sampler_name: euler_ancestral``
    and diffusers' default EulerDiscreteScheduler has an off-by-one
    that surfaces as IndexError on the final denoise step. The
    factory MUST swap the scheduler on every SDXL load."""
    p = _make_safetensors(tmp_path, "ponyDiffusionV6XL_v6.safetensors")
    result = _default_pipeline_factory(p, device="cpu")

    s = patched_sdxl_loaders
    s["scheduler_cls"].from_config.assert_called_once()
    assert result.scheduler is s["swapped_scheduler"], (
        "factory loaded an SDXL pipeline without swapping in "
        "EulerAncestralDiscreteScheduler — the embedded renders will "
        "trip the off-by-one IndexError on the final step"
    )


# ---------------------------------------------------------------------------
# Same routing on the path the app exercises end-to-end:
# EmbeddedImageClient.generate -> _ensure_pipeline -> factory.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_routes_z_image_filename_through_real_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patched_z_image_loaders: dict[str, MagicMock],
) -> None:
    """End-to-end-shaped: build a real ``EmbeddedImageClient`` with
    NO pipeline_factory override (the same shape the app uses) and
    pin Z-Image as the character model. The first ``generate`` call
    must drive the real factory + the real ``_load_z_image_pipeline``
    helper, which is the path that broke in production."""
    from PIL import Image  # type: ignore

    from lucidium.providers.embedded_image_client import EmbeddedImageClient

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    target = models_dir / "zImageTurbo_turbo.safetensors"
    target.write_bytes(b"stub")
    # A second checkpoint that would otherwise win the alphabetical
    # default — exactly the trap the production setup fell into.
    (models_dir / "Qwen-Rapid-AIO.safetensors").write_bytes(b"stub")

    # The pipeline is a callable that returns ``result.images = [PIL]``.
    # Stub the ZImagePipeline instance accordingly so generate() can
    # encode the result back to PNG bytes through the normal path.
    pipeline = patched_z_image_loaders["pipeline_instance"]
    pipeline.config = type("Cfg", (), {"_name_or_path": "Z-Image-Turbo"})()

    def _call_pipeline(**kwargs: Any) -> Any:
        img = Image.new("RGB", (kwargs["width"], kwargs["height"]), color=(20, 20, 20))
        return type("Result", (), {"images": [img]})()

    pipeline.__call__ = _call_pipeline  # type: ignore[method-assign]
    # MagicMock pipelines don't proxy __call__ via instance attr —
    # bind a callable wrapper so ``pipeline(**kwargs)`` actually
    # dispatches to ``_call_pipeline``.
    monkeypatch.setattr(
        type(pipeline),
        "__call__",
        lambda self, **kw: _call_pipeline(**kw),
    )

    client = EmbeddedImageClient(
        models_dir=str(models_dir),
        # Pin Z-Image so we exercise the production fix recommended
        # to the user (otherwise alphabetical-first wins).
        character_model_name="zImageTurbo_turbo.safetensors",
        environment_model_name="zImageTurbo_turbo.safetensors",
        bg_remover=lambda b: b,
    )

    result_bytes = await client.generate(
        "character.json",
        {"positive_prompt": "a quiet figure under a streetlight"},
        seed=42,
    )

    # Real factory drove the Z-Image branch (via the monkeypatched
    # diffusers/transformers).
    z = patched_z_image_loaders
    z["z_image_pipeline_cls"].from_single_file.assert_called_once()
    assert result_bytes.startswith(b"\x89PNG\r\n\x1a\n")
