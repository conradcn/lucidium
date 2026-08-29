"""Compute-device detection — the diagnostic that tells the player
why their SDXL renders are running on CPU instead of GPU.

Background: PyPI's default torch wheel on Windows is the CPU-only
build (``torch X.Y.Z+cpu``). A user who runs ``pip install -e .``
without the PyTorch CUDA index URL silently gets a CPU-only install
and has no idea their RTX 4090 isn't being used. SDXL still works
on CPU but renders take many minutes per image instead of seconds.

``detect_compute_device`` returns ``(device, diagnostic)``:

  * device — one of ``"cuda"``, ``"mps"``, ``"cpu"``.
  * diagnostic — empty on the happy paths (``cuda``/``mps``); on
    the CPU paths it carries a one-line, copy-pasteable summary of
    WHY we're on CPU and (when applicable) the exact uv command to
    install the CUDA wheel.

This file pins the dispatch table so a future refactor that drops a
case (e.g. forgets to flag the ``+cpu`` wheel + NVIDIA case) gets
caught.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from lucidium.providers import embedded_image_client


def _stub_torch(
    *,
    cuda_available: bool,
    mps_available: bool = False,
    version: str = "2.5.1",
) -> types.ModuleType:
    """Build a minimal fake ``torch`` module the detector probes.

    Includes the dtype attributes the wider engine reaches for at
    pipeline-build time (``float16`` / ``float32``) so the
    pipeline-factory smoke test can exercise the full code path.
    """
    fake = types.ModuleType("torch")
    fake.__version__ = version  # type: ignore[attr-defined]

    cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    fake.cuda = cuda  # type: ignore[attr-defined]

    mps = types.SimpleNamespace(is_available=lambda: mps_available)
    fake.backends = types.SimpleNamespace(mps=mps)  # type: ignore[attr-defined]

    # Sentinels — _resolve_torch_dtype just needs ``is`` identity.
    fake.float16 = object()  # type: ignore[attr-defined]
    fake.float32 = object()  # type: ignore[attr-defined]
    return fake


@pytest.fixture
def install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Inject a stub ``torch`` into ``sys.modules`` for the duration
    of a test. Restores the real module after."""

    def _install(**kw: Any) -> types.ModuleType:
        fake = _stub_torch(**kw)
        monkeypatch.setitem(sys.modules, "torch", fake)
        return fake

    return _install


@pytest.fixture
def stub_nvidia_probe(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Override the GPU probe so tests don't depend on the host
    machine actually having (or not having) an NVIDIA card."""

    def _set(present: bool) -> None:
        monkeypatch.setattr(
            embedded_image_client,
            "_has_nvidia_gpu",
            lambda: present,
        )

    return _set


# ---------- Happy paths -----------------------------------------------------


def test_cuda_available_returns_cuda_with_no_diagnostic(
    install_fake_torch,
    stub_nvidia_probe,
) -> None:
    install_fake_torch(cuda_available=True)
    stub_nvidia_probe(True)
    device, diagnostic = embedded_image_client.detect_compute_device()
    assert device == "cuda"
    assert diagnostic == ""


def test_mps_available_returns_mps_with_no_diagnostic(
    install_fake_torch,
    stub_nvidia_probe,
) -> None:
    install_fake_torch(cuda_available=False, mps_available=True)
    stub_nvidia_probe(False)
    device, diagnostic = embedded_image_client.detect_compute_device()
    assert device == "mps"
    assert diagnostic == ""


# ---------- CPU diagnostic dispatch ----------------------------------------


def test_cpu_only_wheel_with_nvidia_present_emits_pip_install_fix(
    install_fake_torch,
    stub_nvidia_probe,
) -> None:
    """The reported case: torch X.Y.Z+cpu installed, RTX-class GPU
    sitting idle. Diagnostic must name the +cpu build AND include
    the install command so the user can copy-paste it."""
    install_fake_torch(cuda_available=False, version="2.11.0+cpu")
    stub_nvidia_probe(True)
    device, diagnostic = embedded_image_client.detect_compute_device()
    assert device == "cpu"
    assert "+cpu" in diagnostic
    assert "NVIDIA" in diagnostic
    # The fix is the LITERAL string the user runs — pin it.
    assert "pip install --force-reinstall torch" in diagnostic
    assert "https://download.pytorch.org/whl/cu130" in diagnostic


def test_cpu_only_wheel_no_gpu_still_recommends_cuda_install(
    install_fake_torch,
    stub_nvidia_probe,
) -> None:
    """User on a CPU-only laptop. Still nudge them toward CUDA in
    case they're going to run on a different machine — but the
    "NVIDIA detected" line is suppressed because there isn't one."""
    install_fake_torch(cuda_available=False, version="2.11.0+cpu")
    stub_nvidia_probe(False)
    device, diagnostic = embedded_image_client.detect_compute_device()
    assert device == "cpu"
    assert "+cpu" in diagnostic
    # Doesn't falsely claim an NVIDIA GPU is present.
    assert "NVIDIA GPU was detected" not in diagnostic
    # Still includes the fix command for users who plan to run on
    # an NVIDIA machine later.
    assert "https://download.pytorch.org/whl/cu130" in diagnostic


def test_cuda_torch_but_runtime_unavailable_blames_driver(
    install_fake_torch,
    stub_nvidia_probe,
) -> None:
    """The "right wheel, wrong driver" case: torch is the CUDA
    build but ``cuda.is_available()`` says False, usually because
    the installed driver is older than the wheel's CUDA runtime.
    Diagnostic must name the driver-update path, not falsely
    claim the wheel is the wrong build."""
    install_fake_torch(cuda_available=False, version="2.5.1+cu128")
    stub_nvidia_probe(True)
    device, diagnostic = embedded_image_client.detect_compute_device()
    assert device == "cpu"
    # NOT a +cpu wheel diagnostic.
    assert "CPU-ONLY build" not in diagnostic
    # Names the driver-mismatch theory and the fix URL.
    assert "driver" in diagnostic.lower()
    assert "https://download.pytorch.org/whl/cu130" in diagnostic


def test_no_gpu_at_all_recommends_machine_with_gpu(
    install_fake_torch,
    stub_nvidia_probe,
) -> None:
    """No CUDA wheel marker, no NVIDIA hardware: the user is
    actually trying to run on a CPU-only machine. Diagnostic
    explains SDXL won't be usable, doesn't pretend a fix exists
    short of new hardware."""
    install_fake_torch(cuda_available=False, version="2.5.1")
    stub_nvidia_probe(False)
    device, diagnostic = embedded_image_client.detect_compute_device()
    assert device == "cpu"
    assert "NVIDIA GPU" in diagnostic
    assert "extremely slow" in diagnostic.lower() or "slow" in diagnostic.lower()


def test_no_torch_at_all_returns_actionable_install_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch isn't installed (rare, but the path needs to surface
    a clear error rather than crashing)."""
    # Force ImportError on ``import torch``.
    monkeypatch.setitem(sys.modules, "torch", None)  # type: ignore[arg-type]
    device, diagnostic = embedded_image_client.detect_compute_device()
    assert device == "cpu"
    assert "PyTorch is not installed" in diagnostic
    assert "pip install --force-reinstall torch" in diagnostic


# ---------- Pipeline factory uses the detected device -----------------------


def test_pipeline_factory_resolves_device_via_detector(
    install_fake_torch,
    stub_nvidia_probe,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: when the factory is called with ``device=None``
    it MUST consult ``detect_compute_device``, not skip the .to()
    call. Earlier shapes left the pipeline at its from_single_file
    default whenever cuda was unavailable; that's a footgun if a
    future torch version changes the default home device."""
    install_fake_torch(cuda_available=False, version="2.11.0+cpu")
    stub_nvidia_probe(True)

    captured: list[str] = []

    class _Pipe:
        scheduler = types.SimpleNamespace(config={})

        def to(self, dev: str) -> _Pipe:
            captured.append(dev)
            return self

    fake_pipe = _Pipe()

    fake_diffusers = types.ModuleType("diffusers")

    class _StableDiffusionXLPipeline:
        @classmethod
        def from_single_file(cls, _path: str, **_kw: Any) -> _Pipe:
            return fake_pipe

    class _EulerAncestralDiscreteScheduler:
        @classmethod
        def from_config(cls, _cfg: Any) -> Any:
            return types.SimpleNamespace()

    fake_diffusers.StableDiffusionXLPipeline = _StableDiffusionXLPipeline  # type: ignore[attr-defined]
    fake_diffusers.EulerAncestralDiscreteScheduler = (  # type: ignore[attr-defined]
        _EulerAncestralDiscreteScheduler
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    from pathlib import Path

    embedded_image_client._default_pipeline_factory(
        Path("ignored.safetensors"),
        None,
    )

    # Detector said "cpu" → factory called .to("cpu").
    assert captured == ["cpu"]


def test_startup_banner_warns_on_cpu_install(
    install_fake_torch,
    stub_nvidia_probe,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The startup banner in ``app._log_compute_device_diagnostics``
    must emit at WARNING level (not INFO) when the device is CPU
    and an NVIDIA GPU is present — that's the loud-enough signal
    Electron's [backend] log routes to the renderer-visible
    console. INFO would scroll off the top before the user sees
    it; the only loud channel for backend is WARNING+ in the
    current logging config."""
    import logging

    from lucidium.app import _log_compute_device_diagnostics

    install_fake_torch(cuda_available=False, version="2.11.0+cpu")
    stub_nvidia_probe(True)

    caplog.set_level(logging.WARNING, logger="lucidium")
    _log_compute_device_diagnostics(logging.getLogger("lucidium"))

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings, "expected a WARNING log line"
    msg = warnings[0].getMessage()
    assert "CPU" in msg
    assert "https://download.pytorch.org/whl/cu130" in msg


def test_startup_banner_quiet_on_cuda(
    install_fake_torch,
    stub_nvidia_probe,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Counterpart: when the user is on CUDA, the banner is INFO,
    not WARNING. Don't cry wolf when everything is fine."""
    import logging

    from lucidium.app import _log_compute_device_diagnostics

    install_fake_torch(cuda_available=True)
    stub_nvidia_probe(True)

    caplog.set_level(logging.INFO, logger="lucidium")
    _log_compute_device_diagnostics(logging.getLogger("lucidium"))

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == []
    infos = [r for r in caplog.records if r.levelname == "INFO"]
    assert any("cuda" in r.getMessage().lower() for r in infos)
