"""AMD / DirectML runtime-ergonomics pins.

Two milestones bundled here, both pure-Python and exercised with NO
real GPU / torch_directml / model downloads:

  * ``HSA_OVERRIDE_GFX_VERSION`` auto-set for unofficially-supported
    AMD cards on Linux (Radeon 6700 XT = gfx1031 needs 10.3.0 for
    ROCm). Must be set in ``app._configure_amd_rocm_env`` BEFORE the
    first torch import, only on Linux + AMD, and never clobber a
    user-provided value.

  * Z-Image-Turbo routing: it needs bf16 + ~18 GiB, which DirectML
    can't provide. The model-resolution path must NOT attempt the
    load — it falls back to an SDXL checkpoint when one is available,
    else raises a clear ``ProviderValidationError``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lucidium import app
from lucidium.api.errors import ProviderValidationError
from lucidium.providers.embedded_image_client import EmbeddedImageClient

# ---------------------------------------------------------------------------
# Task 4 — HSA_OVERRIDE_GFX_VERSION auto-set
# ---------------------------------------------------------------------------


def test_hsa_override_set_on_linux_amd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux + AMD detected + env var unset → set to 10.3.0."""
    monkeypatch.setattr(app.sys, "platform", "linux")
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    from lucidium.providers import embedded_image_client as eic

    monkeypatch.setattr(eic, "_has_amd_gpu", lambda: True)

    app._configure_amd_rocm_env()
    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0"


def test_hsa_override_not_clobbered_when_already_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user-provided value MUST survive — a power user on a
    different arch (e.g. 9.0.0) may have set it deliberately."""
    monkeypatch.setattr(app.sys, "platform", "linux")
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "9.0.0")
    from lucidium.providers import embedded_image_client as eic

    # Should never even reach the probe, but pin the value regardless.
    monkeypatch.setattr(eic, "_has_amd_gpu", lambda: True)

    app._configure_amd_rocm_env()
    assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == "9.0.0"


def test_hsa_override_not_set_when_no_amd_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux but no AMD card → don't set a ROCm-specific var on an
    NVIDIA/Intel box."""
    monkeypatch.setattr(app.sys, "platform", "linux")
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    from lucidium.providers import embedded_image_client as eic

    monkeypatch.setattr(eic, "_has_amd_gpu", lambda: False)

    app._configure_amd_rocm_env()
    assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_hsa_override_not_set_off_linux(
    platform: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The override is a ROCm/HSA concept — never set on Windows
    (DirectML) or macOS (MPS), even with an AMD card present."""
    monkeypatch.setattr(app.sys, "platform", platform)
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    from lucidium.providers import embedded_image_client as eic

    # Probe must not even be consulted off-Linux; make it loud if it is.
    monkeypatch.setattr(
        eic,
        "_has_amd_gpu",
        lambda: (_ for _ in ()).throw(AssertionError("AMD probe must not run off-Linux")),
    )

    app._configure_amd_rocm_env()
    assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ


# ---------------------------------------------------------------------------
# Task 5 — Z-Image routing away from DirectML
# ---------------------------------------------------------------------------


def _write(models_dir: Path, name: str) -> None:
    (models_dir / name).write_bytes(b"stub")


@pytest.mark.asyncio
async def test_z_image_on_dml_falls_back_to_sdxl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Z-Image pinned, device is DML, an SDXL checkpoint is also on
    disk → generate must re-point at the SDXL file and never load
    Z-Image. We assert via the pipeline_factory: it must be handed the
    SDXL path, not the Z-Image one."""
    from PIL import Image  # type: ignore

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write(models_dir, "zImageTurbo_turbo.safetensors")
    _write(models_dir, "dreamshaperXL_v10.safetensors")

    loaded_paths: list[Path] = []

    def fake_factory(path: Path, _device: str | None):
        loaded_paths.append(path)
        pipe = type("P", (), {})()

        def _call(**kwargs):
            img = Image.new("RGB", (kwargs["width"], kwargs["height"]))
            return type("R", (), {"images": [img]})()

        pipe.__call__ = _call  # type: ignore[attr-defined]
        # Bound so pipeline(**kwargs) dispatches.
        type(pipe).__call__ = staticmethod(_call)  # type: ignore[assignment]
        return pipe

    client = EmbeddedImageClient(
        models_dir=str(models_dir),
        character_model_name="zImageTurbo_turbo.safetensors",
        environment_model_name="zImageTurbo_turbo.safetensors",
        device="dml",
        pipeline_factory=fake_factory,
        bg_remover=lambda b: b,
    )

    out = await client.generate(
        "character.json",
        {"positive_prompt": "a figure under a streetlight"},
        seed=1,
    )
    assert out.startswith(b"\x89PNG\r\n\x1a\n")
    assert loaded_paths, "factory was never called"
    assert loaded_paths[0].name == "dreamshaperXL_v10.safetensors", (
        f"DML render loaded {loaded_paths[0].name!r}; expected the SDXL "
        "fallback, not the Z-Image checkpoint"
    )


@pytest.mark.asyncio
async def test_z_image_on_dml_raises_when_no_sdxl_fallback(
    tmp_path: Path,
) -> None:
    """Z-Image is the ONLY checkpoint on disk and device is DML →
    nothing to fall back to, so generate raises a clear
    ProviderValidationError naming the GPU limitation and the fix."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write(models_dir, "zImageTurbo_turbo.safetensors")

    def fail_factory(_path: Path, _device: str | None):
        raise AssertionError("factory must not be called — Z-Image can't load on DML")

    client = EmbeddedImageClient(
        models_dir=str(models_dir),
        character_model_name="zImageTurbo_turbo.safetensors",
        environment_model_name="zImageTurbo_turbo.safetensors",
        device="dml",
        pipeline_factory=fail_factory,
        bg_remover=lambda b: b,
    )

    with pytest.raises(ProviderValidationError) as excinfo:
        await client.generate(
            "character.json",
            {"positive_prompt": "a figure under a streetlight"},
            seed=1,
        )
    msg = str(excinfo.value)
    assert "Z-Image" in msg
    assert "SDXL" in msg


@pytest.mark.asyncio
async def test_z_image_on_cuda_is_not_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate is DML-only: a Z-Image checkpoint on a CUDA device
    (which is also how a capable AMD-on-Linux ROCm card presents) must
    still load Z-Image, NOT get bounced to SDXL."""
    from PIL import Image  # type: ignore

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write(models_dir, "zImageTurbo_turbo.safetensors")
    _write(models_dir, "dreamshaperXL_v10.safetensors")

    loaded_paths: list[Path] = []

    def fake_factory(path: Path, _device: str | None):
        loaded_paths.append(path)
        pipe = type("P", (), {})()

        def _call(**kwargs):
            img = Image.new("RGB", (kwargs["width"], kwargs["height"]))
            return type("R", (), {"images": [img]})()

        type(pipe).__call__ = staticmethod(_call)  # type: ignore[assignment]
        return pipe

    client = EmbeddedImageClient(
        models_dir=str(models_dir),
        character_model_name="zImageTurbo_turbo.safetensors",
        environment_model_name="zImageTurbo_turbo.safetensors",
        device="cuda",
        pipeline_factory=fake_factory,
        bg_remover=lambda b: b,
    )

    await client.generate(
        "character.json",
        {"positive_prompt": "a figure under a streetlight"},
        seed=1,
    )
    assert loaded_paths[0].name == "zImageTurbo_turbo.safetensors", (
        "CUDA must still load Z-Image; the DML gate leaked onto CUDA"
    )
