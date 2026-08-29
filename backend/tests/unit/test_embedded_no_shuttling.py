"""Pin: the embedded image client does NOT shuttle pipelines
between CPU and GPU on every render — neither when the music
client is disabled (the common case) nor when it's enabled-but-
idle (no eviction has happened yet).

Z-Image-Turbo on a 24 GiB card is too large to tolerate even a
single needless CPU↔GPU round-trip per render: the move briefly
double-allocates the pipeline's components and the user already
has ~1.8 GiB headroom after one render. A regression that
unconditionally called ``pipeline.to('cpu')`` / ``pipeline.to('cuda')``
inside ``generate`` would OOM on the next render and leave a
stranded half-CPU pipeline that all subsequent renders would
trip over.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from lucidium.providers.embedded_image_client import EmbeddedImageClient


class _TrackingPipeline:
    """Pipeline stub that records every ``to(device)`` call so the
    test can prove no shuttle happened during inference."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.moves: list[str] = []
        self.config = type("Cfg", (), {"_name_or_path": "stub"})()

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        img = Image.new(
            "RGB",
            (kwargs.get("width", 64), kwargs.get("height", 64)),
            color=(10, 10, 10),
        )
        return type("R", (), {"images": [img]})()

    def to(self, device: str) -> _TrackingPipeline:
        self.moves.append(device)
        return self


@pytest.fixture
def stocked_models_dir(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    (models / "zImageTurbo_turbo.safetensors").write_bytes(b"stub")
    return models


def _build_client(
    models_dir: Path,
    pipeline: _TrackingPipeline,
    *,
    gpu_lock: Any = None,
) -> EmbeddedImageClient:
    return EmbeddedImageClient(
        models_dir=str(models_dir),
        character_model_name="zImageTurbo_turbo.safetensors",
        environment_model_name="zImageTurbo_turbo.safetensors",
        pipeline_factory=lambda _path, _device: pipeline,
        bg_remover=lambda b: b,
        gpu_lock=gpu_lock,
    )


@pytest.mark.asyncio
async def test_no_shuttle_when_gpu_lock_is_none(
    stocked_models_dir: Path,
) -> None:
    """The app path when music + local_gpu_coordination are disabled:
    ``gpu_lock`` is None. ``generate`` must NOT call ``restore_to_gpu``
    on this path and the pipeline must never see a CPU/GPU move
    after construction."""
    pipeline = _TrackingPipeline()
    client = _build_client(stocked_models_dir, pipeline, gpu_lock=None)

    for i in range(3):
        await client.generate(
            "character.json" if i % 2 == 0 else "background.json",
            {"positive_prompt": f"r{i}"},
            seed=100 + i,
        )

    # The factory is a test stub that hands back ``pipeline`` directly,
    # so ``moves`` should be EMPTY: no ``.to`` ever gets invoked on
    # the stub. If a shuttle ever sneaks in, that list would grow.
    assert pipeline.moves == [], (
        f"pipeline was moved between devices during inference; observed moves: {pipeline.moves}"
    )


@pytest.mark.asyncio
async def test_no_shuttle_when_gpu_lock_set_but_nothing_evicted(
    stocked_models_dir: Path,
) -> None:
    """The app path with ``local_gpu_coordination=True`` but no
    actual music render in flight: ``gpu_lock`` is set, but the
    ``_evicted`` set stays empty. ``restore_to_gpu`` is called every
    generate; it MUST short-circuit and not move the pipeline.
    """
    import asyncio

    pipeline = _TrackingPipeline()
    client = _build_client(
        stocked_models_dir,
        pipeline,
        gpu_lock=asyncio.Lock(),
    )

    for i in range(3):
        await client.generate(
            "character.json",
            {"positive_prompt": f"r{i}"},
            seed=200 + i,
        )

    assert pipeline.moves == [], (
        "restore_to_gpu shuttled the pipeline despite _evicted being "
        f"empty; observed moves: {pipeline.moves}"
    )
    # Eviction bookkeeping must also stay empty — a stray ``add()``
    # would silently arm a future restore.
    assert client._evicted == set()
