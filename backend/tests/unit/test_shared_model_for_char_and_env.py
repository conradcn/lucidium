"""User reported: rendering doesn't seem to work when character
and environment models are the same. Pin the same-model code
path so a regression doesn't recur, and use these tests as the
repro target while hunting the bug.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from lucidium.providers.embedded_image_client import EmbeddedImageClient


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.config = type("Cfg", (), {"_name_or_path": "test-stub"})()

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        width = kwargs.get("width", 512)
        height = kwargs.get("height", 512)
        image = Image.new("RGB", (width, height), color=(50, 50, 50))
        result = type("Result", (), {})()
        result.images = [image]
        return result


def _placeholder_png(w: int, h: int) -> bytes:
    img = Image.new("RGBA", (w, h), color=(0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def stocked_models_dir(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    (models / "shared-model.safetensors").write_bytes(b"fake")
    return models


@pytest.mark.asyncio
async def test_same_model_loaded_once_for_both_workflows(
    stocked_models_dir: Path,
) -> None:
    """Sanity baseline: when character and environment names point
    at the same file, the pipeline factory fires exactly once and
    both workflows reuse the cached pipeline."""
    factory_calls: list[Path] = []

    def factory(model_path: Path, _device: Any) -> Any:
        factory_calls.append(model_path)
        return _FakePipeline()

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        character_model_name="shared-model.safetensors",
        environment_model_name="shared-model.safetensors",
        pipeline_factory=factory,
        bg_remover=lambda _b: _placeholder_png(832, 1216),
    )

    await client.generate(
        "character.json",
        {"positive_prompt": "a paladin"},
        seed=1,
    )
    await client.generate(
        "background.json",
        {"positive_prompt": "a harbor at dawn"},
        seed=2,
    )

    assert len(factory_calls) == 1, "shared model should load exactly once across both workflows"


@pytest.mark.asyncio
async def test_same_model_renders_at_correct_dims_per_workflow(
    stocked_models_dir: Path,
) -> None:
    """The shared pipeline must still receive workflow-correct
    dimensions: portrait (832x1216) for character.json,
    landscape (1536x1024) for background.json. A regression
    where both renders went through at the same dims would
    produce mis-sized backgrounds and mis-sized portraits."""
    pipeline = _FakePipeline()

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        character_model_name="shared-model.safetensors",
        environment_model_name="shared-model.safetensors",
        pipeline_factory=lambda _p, _d: pipeline,
        bg_remover=lambda _b: _placeholder_png(832, 1216),
    )

    char_bytes = await client.generate(
        "character.json",
        {"positive_prompt": "a paladin"},
        seed=1,
    )
    bg_bytes = await client.generate(
        "background.json",
        {"positive_prompt": "a harbor at dawn"},
        seed=2,
    )

    assert char_bytes
    assert bg_bytes
    assert len(pipeline.calls) == 2
    # Pipeline saw the right dims per call.
    assert (pipeline.calls[0]["width"], pipeline.calls[0]["height"]) == (832, 1216)
    assert (pipeline.calls[1]["width"], pipeline.calls[1]["height"]) == (1536, 1024)


@pytest.mark.asyncio
async def test_same_model_renders_distinct_prompts(
    stocked_models_dir: Path,
) -> None:
    """The shared pipeline must see distinct positive prompts
    for char vs env. A regression where the prompt got cached
    on the pipeline would produce identical / wrong-subject
    images for the second call."""
    pipeline = _FakePipeline()

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        character_model_name="shared-model.safetensors",
        environment_model_name="shared-model.safetensors",
        pipeline_factory=lambda _p, _d: pipeline,
        bg_remover=lambda _b: _placeholder_png(832, 1216),
    )

    await client.generate(
        "character.json",
        {"positive_prompt": "a paladin in shining plate"},
        seed=1,
    )
    await client.generate(
        "background.json",
        {"positive_prompt": "a stone harbor at dawn"},
        seed=2,
    )

    # Both calls must have flowed through with their own prompt
    # — either as ``prompt`` (no Compel) or as ``prompt_embeds``
    # (Compel-encoded). With the test stub Compel is unavailable
    # so we expect the plain ``prompt`` field on both calls.
    prompts_seen: set[str] = set()
    for call in pipeline.calls:
        if "prompt" in call:
            prompts_seen.add(call["prompt"])
    assert "a paladin in shining plate" in prompts_seen
    assert "a stone harbor at dawn" in prompts_seen


@pytest.mark.asyncio
async def test_same_model_serialises_concurrent_calls(
    stocked_models_dir: Path,
) -> None:
    """Concurrent character + environment renders against the
    SAME pipeline must serialise via the per-pipeline lock —
    diffusers schedulers are not thread-safe and an off-by-one
    IndexError surfaces when two callers race the same
    scheduler. The lock is path-keyed; both workflows resolve
    to the same path so they share the same lock.

    This test verifies the lock is actually being held by
    sleeping the pipeline call and asserting that overlapping
    asyncio.gather calls don't interleave."""
    import asyncio

    in_flight = 0
    max_in_flight = 0

    class _SlowPipeline(_FakePipeline):
        async def _run(self, **kwargs: Any) -> Any:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                await asyncio.sleep(0.05)
                return _FakePipeline.__call__(self, **kwargs)
            finally:
                in_flight -= 1

        def __call__(self, **kwargs: Any) -> Any:
            # The client runs pipelines via asyncio.to_thread,
            # so __call__ is sync. Cheat with run_until on a
            # nested loop — simpler: count synchronously.
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            try:
                import time

                time.sleep(0.05)
                return _FakePipeline.__call__(self, **kwargs)
            finally:
                in_flight -= 1

    pipeline = _SlowPipeline()
    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        character_model_name="shared-model.safetensors",
        environment_model_name="shared-model.safetensors",
        pipeline_factory=lambda _p, _d: pipeline,
        bg_remover=lambda _b: _placeholder_png(832, 1216),
    )

    await asyncio.gather(
        client.generate(
            "character.json",
            {"positive_prompt": "char"},
            seed=1,
        ),
        client.generate(
            "background.json",
            {"positive_prompt": "env"},
            seed=2,
        ),
    )

    assert max_in_flight == 1, (
        f"per-pipeline lock should serialize calls; saw {max_in_flight} "
        "concurrent invocations against the shared pipeline"
    )


@pytest.mark.asyncio
async def test_same_model_face_inpaint_works_after_env_render(
    stocked_models_dir: Path,
) -> None:
    """Regression guard: an env render through the shared
    pipeline must NOT break the next character render's face-
    inpaint pass. The inpaint pipeline cache is attached to
    the text2img pipeline, so a poor lifecycle interaction
    between env render → cached inpaint could surface as
    "rendering doesn't work" on subsequent character calls."""
    pipeline = _FakePipeline()
    inpaint_calls: list[Any] = []

    def inpaint_runner(_pipe: Any, image_bytes: bytes, **_kwargs: Any) -> bytes:
        inpaint_calls.append(_kwargs)
        return image_bytes

    client = EmbeddedImageClient(
        models_dir=str(stocked_models_dir),
        character_model_name="shared-model.safetensors",
        environment_model_name="shared-model.safetensors",
        pipeline_factory=lambda _p, _d: pipeline,
        bg_remover=lambda _b: _placeholder_png(832, 1216),
        face_detail=True,
        face_inpaint_runner=inpaint_runner,
    )

    # 1) Env render first to warm up the shared pipeline.
    await client.generate(
        "background.json",
        {"positive_prompt": "harbor"},
        seed=1,
    )
    # 2) Character render with face_prompt — should trigger inpaint.
    await client.generate(
        "character.json",
        {
            "positive_prompt": "a paladin",
            "face_prompt": "calm expression",
        },
        seed=2,
    )

    assert len(inpaint_calls) == 1, (
        "face inpaint must fire on character render even after "
        "an env render warmed the shared pipeline"
    )
