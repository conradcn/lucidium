"""Pin: when character_model_name == environment_model_name, every
workflow (character + background) AND every ``force_environment``
toggle MUST share a SINGLE loaded pipeline.

Z-Image-Turbo is ~18 GiB resident on a 24 GiB card. Loading it
twice (once for the character workflow, once for the background)
OOMs. The user reported a real CUDA OOM after the embedded backend
finished one background render and tried a second workflow — even
though the pipeline cache key (model path) is identical for both
workflows.

These tests would fail loud if any of the following regressed:

  * ``pick_default_model`` returned a slightly-different path for
    the same configured name (case mismatch, extension drift) and
    the cache lookup missed.
  * The ``force_environment=True`` branch (nonhuman subject_kind)
    bypassed the cache.
  * A future ``_resolve_target_name`` refactor reordered the
    fallback chain in a way that returned ``""`` for one workflow
    and the real name for the other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from lucidium.providers.embedded_image_client import EmbeddedImageClient


class _CountingPipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.config = type("Cfg", (), {"_name_or_path": "shared-stub"})()
        self.moved_to: list[str] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        img = Image.new(
            "RGB",
            (kwargs.get("width", 64), kwargs.get("height", 64)),
            color=(40, 40, 40),
        )
        return type("R", (), {"images": [img]})()

    def to(self, device: str) -> _CountingPipeline:
        self.moved_to.append(device)
        return self


@pytest.fixture
def stocked_models_dir(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    # The shared file. Same one the user pinned in settings.json.
    (models / "zImageTurbo_turbo.safetensors").write_bytes(b"stub")
    # Decoy: an alphabetically-earlier file that WOULD win if the
    # name resolution silently dropped to the default-pick path.
    (models / "Aaa-decoy.safetensors").write_bytes(b"stub")
    return models


def _client_with_shared_model(
    models_dir: Path,
    factory_calls: list[Path],
) -> tuple[EmbeddedImageClient, _CountingPipeline]:
    pipeline = _CountingPipeline()

    def factory(path: Path, _device: Any) -> Any:
        factory_calls.append(path)
        return pipeline

    client = EmbeddedImageClient(
        models_dir=str(models_dir),
        character_model_name="zImageTurbo_turbo.safetensors",
        environment_model_name="zImageTurbo_turbo.safetensors",
        pipeline_factory=factory,
        bg_remover=lambda b: b,
    )
    return client, pipeline


@pytest.mark.asyncio
async def test_character_then_background_share_one_pipeline_load(
    stocked_models_dir: Path,
) -> None:
    """Character workflow then background workflow with the same
    pinned model must produce ONE factory call. A duplicate factory
    call here would mean the cache key drifted — exactly the
    regression that would re-OOM the user."""
    factory_calls: list[Path] = []
    client, _pipeline = _client_with_shared_model(stocked_models_dir, factory_calls)

    await client.generate(
        "character.json",
        {"positive_prompt": "a quiet figure"},
        seed=1,
    )
    await client.generate(
        "background.json",
        {"positive_prompt": "a stone harbor at dawn"},
        seed=2,
    )

    assert len(factory_calls) == 1, (
        "expected ONE pipeline load when character + environment "
        f"models are the same; got {len(factory_calls)}: {factory_calls}"
    )
    assert factory_calls[0].name == "zImageTurbo_turbo.safetensors"
    # Only ONE cache entry — confirms no second pipeline got
    # silently created under a different key.
    assert len(client._pipelines) == 1


@pytest.mark.asyncio
async def test_force_environment_reuses_shared_pipeline(
    stocked_models_dir: Path,
) -> None:
    """The ``force_environment=True`` path (subject_kind="nonhuman")
    picks the environment model. When character_model_name and
    environment_model_name resolve to the same file, that branch
    must hit the cache too — otherwise a beat with a nonhuman
    descriptor right after a human one triggers a second load and
    OOMs the user."""
    factory_calls: list[Path] = []
    client, _pipeline = _client_with_shared_model(stocked_models_dir, factory_calls)

    # First render: regular character.
    await client.generate(
        "character.json",
        {"positive_prompt": "a tall figure"},
        seed=3,
    )
    # Second render: same workflow but subject_kind="nonhuman" forces
    # the environment-model branch internally.
    await client.generate(
        "character.json",
        {
            "positive_prompt": "a glittering crystalline entity",
            "subject_kind": "nonhuman",
        },
        seed=4,
    )

    assert len(factory_calls) == 1, (
        "force_environment=True triggered a second pipeline load "
        "even though env_model == char_model"
    )
    assert len(client._pipelines) == 1


@pytest.mark.asyncio
async def test_third_and_fourth_renders_still_cache_hit(
    stocked_models_dir: Path,
) -> None:
    """Belt-and-braces: every successive render after the first
    must remain a cache hit. The bug here would be a future refactor
    that pops + re-inserts under a NEW path key on each call,
    eventually evicting the cache entry via OOM logic. Pin that the
    factory is called exactly once across many renders."""
    factory_calls: list[Path] = []
    client, _pipeline = _client_with_shared_model(stocked_models_dir, factory_calls)

    for i in range(4):
        await client.generate(
            "character.json" if i % 2 == 0 else "background.json",
            {"positive_prompt": f"render {i}"},
            seed=10 + i,
        )

    assert len(factory_calls) == 1, (
        f"expected ONE pipeline load across 4 alternating renders; "
        f"got {len(factory_calls)} loads at {factory_calls}"
    )


@pytest.mark.asyncio
async def test_short_circuit_immune_to_path_resolution_drift(
    stocked_models_dir: Path,
) -> None:
    """The new ``_shared_model_short_circuit`` returns the cached
    pipeline without re-resolving the path. Simulate a path-resolution
    bug by mutating ``pick_default_model``'s output mid-flight: even
    if the second call would have resolved to a DIFFERENT path,
    the short-circuit MUST keep returning the same pipeline.
    """
    from lucidium.providers import embedded_image_client as eic

    factory_calls: list[Path] = []
    client, _pipeline = _client_with_shared_model(stocked_models_dir, factory_calls)

    # First render loads the pipeline normally.
    await client.generate(
        "character.json",
        {"positive_prompt": "x"},
        seed=5,
    )
    assert len(factory_calls) == 1
    assert len(client._pipelines) == 1

    # Now break the path resolver. A future refactor that ships a
    # subtle case-mismatch bug — for example always returning the
    # decoy file — should NOT be able to bypass the short-circuit.
    original = eic.pick_default_model
    try:
        eic.pick_default_model = lambda _dir, _name: stocked_models_dir / "Aaa-decoy.safetensors"  # type: ignore[assignment]
        await client.generate(
            "background.json",
            {"positive_prompt": "y"},
            seed=6,
        )
    finally:
        eic.pick_default_model = original  # type: ignore[assignment]

    # No second factory call — the short-circuit returned the existing
    # pipeline before the (poisoned) resolver ran.
    assert len(factory_calls) == 1, (
        "path-resolution drift produced a duplicate load — the "
        "shared-model short-circuit must guard against this"
    )
    assert len(client._pipelines) == 1
