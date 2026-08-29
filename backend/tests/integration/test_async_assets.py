"""Asset generation runs OFF the Continue critical path.

The user reported the renderer freezing during play; one cause was
``_walk_existing_child`` and ``_generate_and_walk_chain`` awaiting the
full image-generation pass before yielding a single state update. The
fix moves portrait/background work into a background task that pushes
state/full + image_ready via ``session.emit`` when it finishes; the
foreground walk emits its message sequence immediately.

These tests pin three guarantees:

1. **Continue never blocks on assets.** A play_advance returns its
   complete handler-yielded sequence before the image client's
   generate() has had a chance to complete — even when image
   generation is artificially slow.

2. **Speculation pre-renders portraits.** When a speculative branch
   introduces a new character, the engine starts generating that
   character's portrait BEFORE the player walks the branch. By the
   time the player picks the option, the portrait is on disk.

3. **Async assets emit through ``session.emit``.** When the
   background task finishes, it pushes a state/full + image_ready
   so the renderer can swap the now-stale character.images[0] for
   the freshly generated one.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from lucidium.api.handlers import HandlerContext, build_default_registry
from lucidium.api.messages import (
    Envelope,
    InterviewStep,
    MessageType,
)
from lucidium.orchestration.session import Session


class _FastLlm:
    """Returns queued JSON responses and a no-op fallback so speculation tasks
    don't starve the foreground."""

    _SPEC_FALLBACK = json.dumps(
        {
            "beats": [
                {
                    "text": "(speculative).",
                    "speaker_id": None,
                    "entering_character_ids": [],
                    "leaving_character_ids": [],
                    "new_characters": [],
                    "location_id": None,
                    "location_prompt": None,
                    "character_changes": [],
                }
            ],
            "options": [],
        }
    )

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def complete(self, *_a, **_kw) -> AsyncIterator[str]:
        nv = self._responses.pop(0) if self._responses else self._SPEC_FALLBACK

        async def gen() -> AsyncIterator[str]:
            yield nv

        return gen()


class _SlowImage:
    """Image client whose generate() takes a controlled amount of time and
    counts every call. The slow window proves the foreground walk emission
    finished BEFORE this generate() returned."""

    def __init__(self, *, delay_s: float = 0.2) -> None:
        self.delay_s = delay_s
        self.calls: list[tuple] = []

    async def generate(self, workflow, prompts, *, seed) -> bytes:
        self.calls.append((workflow, dict(prompts), seed))
        await asyncio.sleep(self.delay_s)
        return _PNG_BYTES


class _NullImage:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def generate(self, workflow, prompts, *, seed) -> bytes:
        self.calls.append((workflow, dict(prompts), seed))
        return _PNG_BYTES


# 8-byte PNG magic + minimal IHDR/IEND so atomic_write_bytes succeeds and
# downstream code (if any) doesn't choke on a corrupt file.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _opt(items: list[str]) -> str:
    return json.dumps({"options": items})


def _world_init_payload() -> str:
    return json.dumps(
        {
            "game_name": "Salt Lantern",
            "overall_plot_direction": "Investigate the keeper.",
            "active_plot_threads": [],
            "opening_node": {
                "beats": [
                    {
                        "text": "The harbor wakes slow.",
                        "speaker_id": None,
                        "entering_character_ids": [],
                        "leaving_character_ids": [],
                        "new_characters": [],
                        "location_id": "harbor",
                        "location_prompt": "stone harbor at dawn",
                        "character_changes": [],
                    },
                ],
                "options": [{"id": "left", "text": "Walk to the inn."}],
            },
            "player_character": _player_payload(),
        }
    )


def _player_payload() -> dict:
    return {
        "name": "Iris",
        "description": "wry archivist",
        "gender": "female",
        "age": 28,
        "ethnicity": "local",
        "skin": "pale",
        "hair_color": "auburn",
        "hairstyle": "braid",
        "eye_color": "grey",
        "build": "slight",
        "bust": "moderate",
        "outfit": "wool coat",
        "pose": "standing",
        "expression": "alert",
    }


def _new_character_descriptor(*, id_: str, name: str) -> dict:
    return {
        "id": id_,
        "name": name,
        "description": "a stranger from the road",
        "gender": "male",
        "age": 50,
        "ethnicity": "local",
        "skin": "weathered",
        "hair_color": "iron grey",
        "hairstyle": "short",
        "eye_color": "hazel",
        "build": "lean",
        "bust": "n/a",
        "outfit": "oilskin coat",
        "pose": "standing",
        "expression": "guarded",
    }


def _branch_with_new_character(text: str, char_id: str) -> str:
    return json.dumps(
        {
            "beats": [
                {
                    "text": text,
                    "speaker_id": None,
                    "entering_character_ids": [char_id],
                    "leaving_character_ids": [],
                    "new_characters": [_new_character_descriptor(id_=char_id, name="Mira Quill")],
                    "location_id": None,
                    "location_prompt": None,
                    "character_changes": [],
                }
            ],
            "options": [],
        }
    )


async def _drain(handler_result):
    out = []
    async for msg in handler_result:
        out.append(msg)
    return out


async def _walk_through_interview(ctx: HandlerContext, registry) -> None:
    await _drain(registry.dispatch(Envelope(type=MessageType.c2s_new_game_start, payload={}), ctx))
    for step, ans in [
        (InterviewStep.setting, "stone harbor"),
        (InterviewStep.visual_style, "ink wash"),
        (InterviewStep.genre, "mystery"),
        (InterviewStep.character_description, "wry archivist"),
        (InterviewStep.name, "Iris"),
    ]:
        await _drain(
            registry.dispatch(
                Envelope(
                    type=MessageType.c2s_new_game_answer,
                    payload={"step": step.value, "answer": ans, "is_free_text": False},
                ),
                ctx,
            )
        )


# ---------------------------------------------------------------------------
# 1. Continue does not block on asset generation.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_play_advance_returns_before_image_generation_completes(
    tmp_app_data: Path,
) -> None:
    """``play_advance`` yields its full handler sequence (state/patch +
    text + state/full) BEFORE the image client's slow generate() call
    has had time to finish.
    """

    image = _SlowImage(delay_s=0.5)
    llm = _FastLlm(
        responses=[
            _opt([f"c{i}" for i in range(6)]),
            _opt([f"n{i}" for i in range(8)]),
            _world_init_payload(),
            _branch_with_new_character("Mira steps from the fog.", "mira-quill"),
        ]
    )
    session = Session(llm_client=llm, image_client=image)
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _walk_through_interview(ctx, registry)
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )
    # Drain the speculation tasks the confirm spawned so they don't
    # interfere with the timing measurement below.
    for _ in range(20):
        await asyncio.sleep(0)
    image.calls.clear()

    # Now the timing-sensitive part: a play_advance MUST return all
    # its emissions before a new image generate() completes.
    started = asyncio.get_event_loop().time()
    advance_messages = await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_play_advance, payload={"option_id": "left"}),
            ctx,
        )
    )
    elapsed = asyncio.get_event_loop().time() - started

    # The handler yielded its full sequence in well under the
    # 0.5-second image-generation budget. If asset generation were
    # blocking, this would have taken at least 0.5s.
    assert elapsed < 0.4, (
        f"play_advance took {elapsed:.3f}s — should have returned without "
        f"waiting for the slow image client (0.5s)"
    )
    # And the foreground emissions are complete: state/patch + text
    # streaming + text complete + state/full.
    types = [t for t, _ in advance_messages]
    assert MessageType.s2c_state_patch in types
    assert MessageType.s2c_state_full in types
    # No image_ready in the foreground emission anymore — those land
    # asynchronously through session.emit instead.
    assert MessageType.s2c_image_ready not in types


# ---------------------------------------------------------------------------
# 2. Speculation pre-renders portraits for new speculative characters.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_speculation_pre_renders_portraits_for_new_characters(
    tmp_app_data: Path,
) -> None:
    """When a speculative branch introduces a character via
    ``new_characters``, the engine starts generating that character's
    portrait in the background. By the time the player walks the
    branch, the portrait is already on disk.
    """

    image = _NullImage()
    llm = _FastLlm(
        responses=[
            _opt([f"c{i}" for i in range(6)]),
            _opt([f"n{i}" for i in range(8)]),
            _world_init_payload(),
            _branch_with_new_character("Mira steps from the fog.", "mira-quill"),
        ]
    )
    session = Session(llm_client=llm, image_client=image)
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _walk_through_interview(ctx, registry)
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )
    # Wait for the speculation chain → speculative portrait task →
    # image_client.generate() for Mira. Use real sleep (not sleep(0))
    # so the chained tasks all get a chance to run.
    # The loop waits for the image to LAND on the character record, not
    # merely for ``generate()`` to have been called: everything after
    # the render — the safety check, the crop, the atomic write — runs
    # on worker threads, so the call and the result are no longer in the
    # same scheduling slice.
    # Generous bound: the output safety filter constructs NudeDetector
    # and insightface on first use in the process, which is seconds of
    # ONNX model loading. It breaks out as soon as the image lands.
    for _ in range(600):
        await asyncio.sleep(0.05)
        mira = (session.game.characters or {}).get("mira-quill")
        if mira is not None and mira.images:
            break

    portrait_calls = [c for c in image.calls if c[0] == session.settings.image.portrait_workflow]
    assert len(portrait_calls) >= 1, (
        f"expected speculative portrait pre-render; image.calls={image.calls}"
    )
    # The speculative character is in game.characters thanks to the
    # pre-materialization step in _speculate_branch.
    assert "mira-quill" in session.game.characters
    mira = session.game.characters["mira-quill"]
    # And the pre-rendered image landed on her record (so when the
    # player walks the speculative chain, the renderer's
    # character.images[0] is already the new portrait).
    assert len(mira.images) >= 1


def call_seed_matches(image: _NullImage, name_substr: str) -> bool:
    """Best-effort heuristic — is at least one portrait call recorded?"""
    return len(image.calls) >= 1


# ---------------------------------------------------------------------------
# 3. Async asset task pushes state/full + image_ready through session.emit.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_assets_push_state_full_via_session_emit(
    tmp_app_data: Path,
) -> None:
    """When the background asset task finishes, it pushes a state/full
    so the renderer picks up the new character.images[0]."""

    emitted: list[tuple[MessageType, object]] = []

    def emit(message_type: MessageType, payload):
        emitted.append((message_type, payload))

    image = _NullImage()
    llm = _FastLlm(
        responses=[
            _opt([f"c{i}" for i in range(6)]),
            _opt([f"n{i}" for i in range(8)]),
            _world_init_payload(),
            _branch_with_new_character("Mira steps from the fog.", "mira-quill"),
        ]
    )
    session = Session(llm_client=llm, image_client=image, emit=emit)
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _walk_through_interview(ctx, registry)
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_play_advance, payload={"option_id": "left"}),
            ctx,
        )
    )
    # Drain the background asset task. Real sleeps, not ``sleep(0)``:
    # the render's post-processing, safety check, atomic write and the
    # save commit all run on worker threads now, and a zero-delay yield
    # never lets the executor finish.
    for _ in range(100):
        if any(t == MessageType.s2c_state_full for t, _ in emitted):
            break
        await asyncio.sleep(0.02)

    # The emit channel should have received at least one state/full
    # AND at least one image_ready from the background asset task.
    state_fulls = [p for t, p in emitted if t == MessageType.s2c_state_full]
    image_readies = [p for t, p in emitted if t == MessageType.s2c_image_ready]
    assert len(state_fulls) >= 1, f"expected at least one state/full pushed via emit; got {emitted}"
    assert len(image_readies) >= 1, (
        f"expected at least one image_ready pushed via emit; got {emitted}"
    )
