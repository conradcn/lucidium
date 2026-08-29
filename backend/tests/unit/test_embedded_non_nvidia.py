"""Pin: the embedded image backend works on non-NVIDIA accelerators.

CUDA / ROCm (AMD-on-Linux) flow through the same ``torch.cuda``
APIs — already covered by the existing tests. The other accelerator
families need their own pins:

  * Intel Arc / oneAPI: ``torch.xpu`` — present on recent PyTorch
    builds but not always populated; the device-detection helper and
    the accelerator-module abstraction must surface it correctly.
  * Apple Silicon: ``torch.backends.mps`` — supported since PyTorch
    1.12 but ``enable_model_cpu_offload`` may not work, so the
    Z-Image loader must fall back to all-resident placement instead
    of raising.

These tests don't require any of the above hardware to be present:
they exercise the dispatch logic via ``sys.modules`` patches against
``torch`` so the production code paths execute exactly the way they
would on the corresponding real accelerator.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lucidium.providers.embedded_image_client import (
    _accelerator_module,
    _make_seeded_generator,
    _release_vram,
    _resolve_torch_dtype,
    _torch_device_arg,
    detect_compute_device,
)


@pytest.fixture
def fake_xpu_torch(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ``torch`` so the accelerator picker thinks an Intel Arc
    is the active device. Returns the fake torch so individual tests
    can drive the mock further."""
    fake = MagicMock(name="torch")
    fake.__version__ = "2.5.0+xpu"
    fake.cuda.is_available.return_value = False

    fake.xpu = MagicMock(name="xpu")
    fake.xpu.is_available.return_value = True
    fake.xpu.empty_cache = MagicMock()
    fake.xpu.memory_allocated.return_value = 2 * (1024**3)
    fake.xpu.max_memory_allocated.return_value = 4 * (1024**3)
    fake.xpu.mem_get_info.return_value = (12 * (1024**3), 16 * (1024**3))
    fake.xpu.reset_peak_memory_stats = MagicMock()
    fake.xpu.is_bf16_supported.return_value = True

    fake.backends.mps.is_available.return_value = False
    fake.float16 = "float16-sentinel"
    fake.bfloat16 = "bfloat16-sentinel"
    fake.float32 = "float32-sentinel"

    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


@pytest.fixture
def fake_mps_torch(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake = MagicMock(name="torch")
    fake.__version__ = "2.5.0"
    fake.cuda.is_available.return_value = False

    fake.xpu = MagicMock(name="xpu")
    fake.xpu.is_available.return_value = False

    fake.backends.mps.is_available.return_value = True
    fake.mps = MagicMock(name="mps")
    fake.mps.empty_cache = MagicMock()
    # MPS deliberately exposes a much thinner introspection surface;
    # the production code must tolerate the missing methods.
    del fake.mps.memory_allocated
    del fake.mps.mem_get_info

    fake.float16 = "float16-sentinel"
    fake.bfloat16 = "bfloat16-sentinel"
    fake.float32 = "float32-sentinel"

    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


@pytest.fixture
def fake_cpu_only_torch(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake = MagicMock(name="torch")
    fake.__version__ = "2.5.0+cpu"
    fake.cuda.is_available.return_value = False
    fake.xpu = MagicMock(name="xpu")
    fake.xpu.is_available.return_value = False
    fake.backends.mps.is_available.return_value = False
    fake.float32 = "float32-sentinel"
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


@pytest.fixture
def fake_dml_torch(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch ``torch`` so NO native accelerator is available, and
    inject a fake ``torch_directml`` module so the DirectML branch
    lights up. Mirrors a Windows-AMD machine on the ``torch-directml``
    wheel: ``torch.cuda/xpu/mps`` are all absent but a DX12 GPU is
    reachable through the separate ``torch_directml`` package.

    ``torch_directml`` is NOT installed in this venv, so the only way
    to exercise the code path is this ``sys.modules`` injection."""
    fake = MagicMock(name="torch")
    fake.__version__ = "2.5.0"
    fake.cuda.is_available.return_value = False
    fake.xpu = MagicMock(name="xpu")
    fake.xpu.is_available.return_value = False
    fake.backends.mps.is_available.return_value = False
    fake.float16 = "float16-sentinel"
    fake.bfloat16 = "bfloat16-sentinel"
    fake.float32 = "float32-sentinel"
    monkeypatch.setitem(sys.modules, "torch", fake)

    dml = MagicMock(name="torch_directml")
    dml.is_available.return_value = True
    dml.device_count.return_value = 1
    # The real handle a pipeline.to(...) call would receive.
    dml.device.return_value = "privateuseone:0-sentinel"
    monkeypatch.setitem(sys.modules, "torch_directml", dml)
    return dml


# ---------------------------------------------------------------------------
# detect_compute_device
# ---------------------------------------------------------------------------


def test_detect_picks_xpu_when_cuda_absent(fake_xpu_torch: MagicMock) -> None:
    device, diag = detect_compute_device()
    assert device == "xpu"
    assert diag == "", f"happy path must surface no diagnostic; got {diag!r}"


def test_detect_picks_mps_when_only_apple_silicon_present(
    fake_mps_torch: MagicMock,
) -> None:
    device, diag = detect_compute_device()
    assert device == "mps"
    assert diag == ""


def test_detect_cpu_diagnostic_routes_amd_to_rocm_wheel(
    fake_cpu_only_torch: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU-wheel + AMD-only present: diagnostic must name AMD AND
    surface the ROCm install path (earlier the message only suggested
    the CUDA wheel, which is wrong for AMD users)."""
    from lucidium.providers import embedded_image_client as eic

    monkeypatch.setattr(eic, "_has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(eic, "_has_amd_gpu", lambda: True)
    monkeypatch.setattr(eic, "_has_intel_arc_gpu", lambda: False)

    device, diag = detect_compute_device()
    assert device == "cpu"
    assert "AMD" in diag
    assert "rocm" in diag.lower()


def test_detect_cpu_diagnostic_routes_intel_arc_to_xpu_wheel(
    fake_cpu_only_torch: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU-wheel + Intel Arc present: diagnostic must name Intel Arc
    AND surface the XPU install path."""
    from lucidium.providers import embedded_image_client as eic

    monkeypatch.setattr(eic, "_has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(eic, "_has_amd_gpu", lambda: False)
    monkeypatch.setattr(eic, "_has_intel_arc_gpu", lambda: True)

    device, diag = detect_compute_device()
    assert device == "cpu"
    assert "Intel Arc" in diag
    assert "/xpu" in diag.lower() or "xpu wheel" in diag.lower()


def test_amd_vendor_probe_does_not_spawn_subprocess_on_linux(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """The AMD vendor probe MUST NOT shell out to ``rocminfo`` on
    Linux when the sysfs PCI vendor file already answers the
    question. On a healthy-driver but broken-HIP install,
    ``rocminfo`` has been observed to hang during runtime probe
    long enough that the backend's LUCIDIUM_WS_PORT announcement
    outruns Electron's connection handshake — surfacing to the
    user as 'AppImage hangs on Linux/AMD'. Pin that the sysfs path
    short-circuits the subprocess.
    """
    import subprocess

    from lucidium.providers import embedded_image_client as eic

    # Pretend we're on Linux.
    monkeypatch.setattr(eic.sys, "platform", "linux")

    # Build a fake /sys/class/drm tree with an AMD card.
    fake_root = tmp_path.mktemp("sys") if hasattr(tmp_path, "mktemp") else tmp_path  # type: ignore[union-attr]
    sysfs = fake_root / "class" / "drm" / "card0" / "device"  # type: ignore[operator]
    sysfs.mkdir(parents=True)
    (sysfs / "vendor").write_text("0x1002\n")

    # Redirect ``glob`` to our fake tree.
    from glob import glob as real_glob

    def fake_glob(pattern: str) -> list[str]:
        if pattern == "/sys/class/drm/card?/device/vendor":
            return [str(sysfs / "vendor")]
        return real_glob(pattern)

    monkeypatch.setattr("glob.glob", fake_glob)

    # Any subprocess.run is a hard failure: if the probe falls
    # through to the CLI path, it can hang.
    def hard_fail(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "vendor probe spawned a subprocess on Linux — the sysfs "
            "fast path must have short-circuited it"
        )

    monkeypatch.setattr(subprocess, "run", hard_fail)

    assert eic._has_amd_gpu() is True


# ---------------------------------------------------------------------------
# Accelerator-module abstraction
# ---------------------------------------------------------------------------


def test_accelerator_module_resolves_to_xpu(fake_xpu_torch: MagicMock) -> None:
    accel = _accelerator_module(fake_xpu_torch)
    assert accel is fake_xpu_torch.xpu


def test_accelerator_module_resolves_to_mps(fake_mps_torch: MagicMock) -> None:
    accel = _accelerator_module(fake_mps_torch)
    assert accel is fake_mps_torch.mps


def test_accelerator_module_returns_none_on_cpu_only(
    fake_cpu_only_torch: MagicMock,
) -> None:
    assert _accelerator_module(fake_cpu_only_torch) is None


# ---------------------------------------------------------------------------
# _release_vram routes to the active accelerator's empty_cache
# ---------------------------------------------------------------------------


def test_release_vram_calls_xpu_empty_cache(
    fake_xpu_torch: MagicMock,
) -> None:
    """On an XPU machine, _release_vram MUST call torch.xpu.empty_cache,
    not torch.cuda.empty_cache (which would be a no-op because CUDA
    isn't initialised) — that would leak fragments per render."""
    _release_vram()
    fake_xpu_torch.xpu.empty_cache.assert_called_once()
    fake_xpu_torch.cuda.empty_cache.assert_not_called()


def test_release_vram_calls_mps_empty_cache(
    fake_mps_torch: MagicMock,
) -> None:
    _release_vram()
    fake_mps_torch.mps.empty_cache.assert_called_once()


def test_release_vram_noop_on_cpu_only(
    fake_cpu_only_torch: MagicMock,
) -> None:
    # No exception, no empty_cache call to look for — the assertion
    # is "didn't crash". CPU-only installs are the path most likely
    # to silently get this wrong via a stray ``torch.cuda.empty_cache``.
    _release_vram()
    fake_cpu_only_torch.cuda.empty_cache.assert_not_called()


# ---------------------------------------------------------------------------
# Z-Image loader: cpu_offload fallback on accelerators that can't run it
# ---------------------------------------------------------------------------


def test_z_image_loader_falls_back_to_eager_when_offload_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some accelerators (older diffusers on MPS, some XPU builds)
    raise on ``enable_model_cpu_offload``. The loader must fall
    through to plain ``.to(device)`` rather than propagate the
    error — otherwise non-NVIDIA users would hit a hard failure
    instead of just losing the VRAM savings.
    """
    # Import + materialise the lazy attrs on the real modules so
    # monkeypatch can swap them (same dance as
    # test_embedded_pipeline_factory). The explicit imports also
    # populate ``sys.modules`` — without them, the dict lookup
    # below KeyErrors because the production code only does these
    # imports inside ``_load_z_image_pipeline``.
    import diffusers  # noqa: F401
    import transformers  # noqa: F401

    from lucidium.providers import embedded_image_client as eic

    _ = sys.modules["transformers"].AutoModel
    _ = sys.modules["transformers"].AutoTokenizer
    _ = sys.modules["diffusers"].AutoencoderKL
    _ = sys.modules["diffusers"].ZImagePipeline

    pipeline = MagicMock(name="pipeline")
    pipeline.enable_model_cpu_offload.side_effect = NotImplementedError(
        "offload not implemented on this backend",
    )
    moved_pipeline = MagicMock(name="moved_pipeline")
    pipeline.to.return_value = moved_pipeline

    z_image_cls = MagicMock(name="ZImagePipeline")
    z_image_cls.from_single_file.return_value = pipeline

    auto_model = MagicMock(name="AutoModel")
    auto_model.from_pretrained.return_value = MagicMock(name="text_encoder")
    auto_tokenizer = MagicMock(name="AutoTokenizer")
    auto_tokenizer.from_pretrained.return_value = MagicMock(name="tokenizer")
    autoencoder = MagicMock(name="AutoencoderKL")
    autoencoder.from_pretrained.return_value = MagicMock(name="vae")

    monkeypatch.setattr(sys.modules["transformers"], "AutoModel", auto_model)
    monkeypatch.setattr(sys.modules["transformers"], "AutoTokenizer", auto_tokenizer)
    monkeypatch.setattr(sys.modules["diffusers"], "AutoencoderKL", autoencoder)
    monkeypatch.setattr(sys.modules["diffusers"], "ZImagePipeline", z_image_cls)

    p = tmp_path / "zImageTurbo_turbo.safetensors"
    p.write_bytes(b"stub")

    # The fallback path must not raise — that's the whole point.
    result = eic._load_z_image_pipeline(p, device="mps")

    pipeline.enable_model_cpu_offload.assert_called()
    pipeline.to.assert_called_with("mps")
    assert result is moved_pipeline, (
        "loader must return the eager-resident pipeline after the cpu_offload fallback fires"
    )


# ---------------------------------------------------------------------------
# DirectML (Windows-AMD / any DX12 GPU via the torch-directml wheel)
# ---------------------------------------------------------------------------


def test_detect_picks_dml_after_mps_and_before_cpu(
    fake_dml_torch: MagicMock,
) -> None:
    """No native accelerator, but torch_directml reports a device:
    detect must return the ``"dml"`` sentinel with no diagnostic
    (it's a happy accelerator path, not a CPU-fallback warning)."""
    device, diag = detect_compute_device()
    assert device == "dml"
    assert diag == "", f"DML is an accelerator path; expected no diag, got {diag!r}"


def test_detect_dml_uses_device_count_when_is_available_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older torch_directml builds expose only ``device_count()`` and
    never grew ``is_available()``. The defensive-getattr fallback must
    treat ``device_count() > 0`` as available so those builds still
    light up the DML branch."""
    fake = MagicMock(name="torch")
    fake.__version__ = "2.5.0"
    fake.cuda.is_available.return_value = False
    fake.xpu = MagicMock(name="xpu")
    fake.xpu.is_available.return_value = False
    fake.backends.mps.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", fake)

    dml = MagicMock(name="torch_directml")
    # Simulate the old build: no is_available attribute at all.
    del dml.is_available
    dml.device_count.return_value = 1
    monkeypatch.setitem(sys.modules, "torch_directml", dml)

    device, diag = detect_compute_device()
    assert device == "dml"
    assert diag == ""


def test_detect_falls_to_cpu_when_directml_unavailable(
    fake_cpu_only_torch: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch_directml present but reporting no device must NOT pick
    dml — it falls through to the CPU diagnostics."""
    dml = MagicMock(name="torch_directml")
    dml.is_available.return_value = False
    dml.device_count.return_value = 0
    monkeypatch.setitem(sys.modules, "torch_directml", dml)
    # No GPU vendors present so the CPU-wheel branch produces a diag.
    from lucidium.providers import embedded_image_client as eic

    monkeypatch.setattr(eic, "_has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(eic, "_has_amd_gpu", lambda: False)
    monkeypatch.setattr(eic, "_has_intel_arc_gpu", lambda: False)

    device, _diag = detect_compute_device()
    assert device == "cpu"


def test_accelerator_module_is_none_on_dml(
    fake_dml_torch: MagicMock,
) -> None:
    """DirectML has no cuda-parity memory module, so the accelerator
    abstraction returns None — which makes _release_vram / evict /
    restore cleanly no-op rather than crash on a missing empty_cache."""
    import torch  # the fake injected by the fixture

    assert _accelerator_module(torch) is None


def test_release_vram_noop_on_dml(fake_dml_torch: MagicMock) -> None:
    """On DML the accel module is None, so _release_vram must not try
    to call empty_cache on anything — and must not raise."""
    import torch

    _release_vram()
    torch.cuda.empty_cache.assert_not_called()


def test_resolve_dtype_is_float16_on_dml(fake_dml_torch: MagicMock) -> None:
    """The accel module is None on DML (same as CPU), but DML needs
    fp16, NOT the CPU fp32 default — running SDXL fp32 on a 8-16 GB
    DML card doubles VRAM and is far slower. Pin fp16."""
    assert _resolve_torch_dtype() == "float16-sentinel"


def test_resolve_dtype_stays_fp32_on_genuine_cpu(
    fake_cpu_only_torch: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard the inverse: when torch_directml is NOT importable, a
    None accel means genuinely CPU-only and fp32 is correct."""
    # Ensure no stray torch_directml is importable.
    monkeypatch.setitem(sys.modules, "torch_directml", None)
    # ``import torch_directml`` with the value None in sys.modules
    # raises ImportError in CPython, which is exactly the "not
    # installed" signal the production code keys on.
    assert _resolve_torch_dtype() == "float32-sentinel"


def test_resolve_dtype_never_offers_bf16_on_dml(
    fake_dml_torch: MagicMock,
) -> None:
    """Even when a caller asks for bf16 (Z-Image / Flux), DML must
    still resolve fp16 — DirectML's bf16 support is unreliable across
    driver/runtime versions."""
    assert _resolve_torch_dtype(prefer_bfloat16=True) == "float16-sentinel"


# ---------------------------------------------------------------------------
# _torch_device_arg — the "dml" -> torch_directml.device() mapping
# ---------------------------------------------------------------------------


def test_torch_device_arg_maps_dml_to_directml_handle(
    fake_dml_torch: MagicMock,
) -> None:
    """The literal string "dml" is NOT a valid torch device; it must
    be mapped to ``torch_directml.device()`` for ``.to(...)``."""
    arg = _torch_device_arg("dml")
    assert arg == "privateuseone:0-sentinel"
    fake_dml_torch.device.assert_called_once()


@pytest.mark.parametrize("device", ["cuda", "xpu", "mps", "cpu"])
def test_torch_device_arg_passes_through_native_devices(device: str) -> None:
    """Every non-DML device string must pass through untouched so
    CUDA/XPU/MPS/CPU placement is byte-identical to before the helper
    existed."""
    assert _torch_device_arg(device) == device


# ---------------------------------------------------------------------------
# _make_seeded_generator — DML builds the generator on CPU
# ---------------------------------------------------------------------------


def test_make_seeded_generator_uses_cpu_for_dml() -> None:
    """DirectML often lacks a device-side generator, so for "dml" (and
    the registered ``privateuseone`` device) the generator must be
    built with NO device arg (CPU)."""
    fake_torch = MagicMock(name="torch")
    gen = MagicMock(name="generator")
    fake_torch.Generator.return_value = gen

    _make_seeded_generator(fake_torch, "dml", 42)

    # No-device-arg form, then manual_seed.
    fake_torch.Generator.assert_called_once_with()
    gen.manual_seed.assert_called_once_with(42)


def test_make_seeded_generator_uses_cpu_for_privateuseone() -> None:
    """The pipeline's reported device on DML is ``privateuseone:0``;
    that must take the CPU-generator branch too."""
    fake_torch = MagicMock(name="torch")
    gen = MagicMock(name="generator")
    fake_torch.Generator.return_value = gen

    _make_seeded_generator(fake_torch, "privateuseone:0", 7)

    fake_torch.Generator.assert_called_once_with()


@pytest.mark.parametrize("device", ["cuda", "mps", "xpu", "cpu"])
def test_make_seeded_generator_pins_native_device(device: str) -> None:
    """Non-DML devices keep the device-pinned generator so manual_seed
    actually takes effect (diffusers ignores a CPU generator on GPU)."""
    fake_torch = MagicMock(name="torch")
    gen = MagicMock(name="generator")
    fake_torch.Generator.return_value = gen

    _make_seeded_generator(fake_torch, device, 99)

    fake_torch.Generator.assert_called_once_with(device=device)
    gen.manual_seed.assert_called_once_with(99)
