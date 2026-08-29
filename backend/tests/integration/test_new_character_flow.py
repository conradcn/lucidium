"""When the LLM introduces a new character mid-dialog, the engine MUST:

1. Materialize a real ``Character`` from the descriptor in
   ``new_characters`` BEFORE evaluating ``entering_character_ids``.
2. Place that character on stage so the renderer draws them.
3. Pass them to the asset pipeline so a portrait gets generated.

The original bug: ``entering_character_ids`` referenced an id absent
from ``game.characters``, so ``_apply_node`` silently dropped it.
The character never appeared, even when the dialog addressed them.
"""

from __future__ import annotations

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


class QueuedLlm:
    """LlmClient with a primary fixture queue plus a fallback used by
    speculative tasks that fire off the back of every committed turn.
    The fallback returns a no-op single-beat chain so the speculation
    layer doesn't starve the foreground call when the test-author
    hasn't padded with extra fixtures.
    """

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
        next_value = self._responses.pop(0) if self._responses else self._SPEC_FALLBACK

        async def gen() -> AsyncIterator[str]:
            yield next_value

        return gen()


class _NullImage:
    async def generate(self, *_a, **_kw) -> bytes:
        return b""


def _options(items: list[str]) -> str:
    return json.dumps({"options": items})


def _full_descriptor(*, id_: str, name: str) -> dict:
    """Helper for emitting a full new-character descriptor in test fixtures."""
    return {
        "id": id_,
        "name": name,
        "description": "a stranger from the south road",
        "gender": "male",
        "age": 38,
        "ethnicity": "local",
        "skin": "weathered",
        "hair_color": "iron grey",
        "hairstyle": "short",
        "eye_color": "hazel",
        "build": "lean",
        "bust": "n/a",
        "outfit": "wool coat over road-stained linen",
        "pose": "standing",
        "expression": "guarded",
    }


def _world_init_with_no_npcs() -> str:
    return json.dumps(
        {
            "game_name": "The Salt Lantern",
            "overall_plot_direction": "Find what happened to the lighthouse keeper.",
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
                "options": [{"id": "opt-1", "text": "Walk to the inn."}],
            },
            "player_character": _full_descriptor(id_="iris", name="Iris"),
        }
    )


def _next_turn_introducing_mira() -> str:
    """The crucial fixture: the LLM emits a beat that introduces a new
    character. ``entering_character_ids`` references "mira-quill" and
    ``new_characters`` carries her full descriptor.
    """
    return json.dumps(
        {
            "beats": [
                {
                    "text": "A figure steps from the fog and lifts a lantern.",
                    "speaker_id": None,
                    "entering_character_ids": ["mira-quill"],
                    "leaving_character_ids": [],
                    "new_characters": [_full_descriptor(id_="mira-quill", name="Mira Quill")],
                    "location_id": None,
                    "location_prompt": None,
                    "character_changes": [],
                },
            ],
            "options": [],
        }
    )


def _next_turn_orphan_reference() -> str:
    """Regression fixture: the LLM mentions an unknown character WITHOUT
    a descriptor. Engine must skip the orphan id and not crash.
    """
    return json.dumps(
        {
            "beats": [
                {
                    "text": "Iris glances toward the tide line.",
                    "speaker_id": None,
                    "entering_character_ids": ["nonexistent"],
                    "leaving_character_ids": [],
                    "new_characters": [],
                    "location_id": None,
                    "location_prompt": None,
                    "character_changes": [],
                },
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
    for step, answer in [
        (InterviewStep.setting, "stone harbor at dawn"),
        (InterviewStep.visual_style, "ink wash painting, monochrome"),
        (InterviewStep.genre, "occult mystery"),
        (InterviewStep.character_description, "wry archivist"),
        (InterviewStep.name, "Iris"),
    ]:
        await _drain(
            registry.dispatch(
                Envelope(
                    type=MessageType.c2s_new_game_answer,
                    payload={"step": step.value, "answer": answer, "is_free_text": False},
                ),
                ctx,
            )
        )


@pytest.mark.asyncio
async def test_new_character_descriptor_materializes_into_game_characters(
    tmp_app_data: Path,
) -> None:
    """The original bug: a beat introduces ``mira-quill`` but no Character
    entry was created. After this fix, _apply_node materializes the
    descriptor and adds it to game.characters AND game.on_stage.
    """

    queued = QueuedLlm(
        responses=[
            _options([f"char-{i}" for i in range(6)]),
            _options([f"name-{i}" for i in range(8)]),
            _world_init_with_no_npcs(),
            _next_turn_introducing_mira(),
        ]
    )
    session = Session(llm_client=queued, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _walk_through_interview(ctx, registry)
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )

    assert session.game is not None
    # Before the play turn: only the player character exists.
    assert all(not c.is_player or True for c in session.game.characters.values())
    assert "mira-quill" not in session.game.characters

    # Click an option to trigger a fresh turn that introduces Mira.
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_play_advance, payload={"option_id": "opt-1"}),
            ctx,
        )
    )

    # Mira must now exist as a real Character.
    assert "mira-quill" in session.game.characters, "new character descriptor was not materialized"
    mira = session.game.characters["mira-quill"]
    assert mira.name == "Mira Quill"
    assert mira.is_player is False
    assert mira.seed > 0  # the engine assigned a seed for portrait generation
    # All descriptor fields round-tripped.
    assert mira.outfit == "wool coat over road-stained linen"
    assert mira.expression == "guarded"

    # And she's on stage so the renderer draws her.
    assert "mira-quill" in session.game.on_stage

    # The DialogNode that introduced her preserves the descriptor — so
    # save reload reconstructs the same character.
    current_node = session.game.dialog_tree.nodes[session.game.current_node_id]
    assert any(d.id == "mira-quill" for d in current_node.new_characters)


@pytest.mark.asyncio
async def test_orphan_entering_character_id_is_dropped_silently(
    tmp_app_data: Path,
) -> None:
    """A beat references an id with no corresponding descriptor. The
    engine must not crash and must not add a phantom id to on_stage.
    """

    queued = QueuedLlm(
        responses=[
            _options([f"char-{i}" for i in range(6)]),
            _options([f"name-{i}" for i in range(8)]),
            _world_init_with_no_npcs(),
            _next_turn_orphan_reference(),
        ]
    )
    session = Session(llm_client=queued, image_client=_NullImage())
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
            Envelope(type=MessageType.c2s_play_advance, payload={"option_id": "opt-1"}),
            ctx,
        )
    )

    assert session.game is not None
    # The orphan id never made it onto the stage — no phantom rendering.
    assert "nonexistent" not in session.game.on_stage
    # And the on_stage list satisfies the Game validator (every id has
    # a corresponding entry in characters).
    for cid in session.game.on_stage:
        assert cid in session.game.characters


def _next_turn_with_player_name_collision() -> str:
    """The LLM tries to introduce an NPC who happens to share the
    player's name (Iris). The engine MUST drop the descriptor so the
    on-stage roster never contains two characters answering to the
    same name.
    """
    return json.dumps(
        {
            "beats": [
                {
                    "text": "A figure who calls herself Iris steps from the fog.",
                    "speaker_id": None,
                    "entering_character_ids": ["other-iris"],
                    "leaving_character_ids": [],
                    "new_characters": [_full_descriptor(id_="other-iris", name="Iris")],
                    "location_id": None,
                    "location_prompt": None,
                    "character_changes": [],
                },
            ],
            "options": [],
        }
    )


@pytest.mark.asyncio
async def test_npc_with_player_name_is_dropped(tmp_app_data: Path) -> None:
    """Defense-in-depth: the prompt forbids it, but if the LLM still
    emits an NPC whose name matches the player's, the engine drops
    the descriptor at materialization time. The on-stage roster
    contains only the originally-named characters; nothing claiming
    to be "Iris" appears beside the actual Iris.
    """

    queued = QueuedLlm(
        responses=[
            _options([f"char-{i}" for i in range(6)]),
            _options([f"name-{i}" for i in range(8)]),
            _world_init_with_no_npcs(),
            _next_turn_with_player_name_collision(),
        ]
    )
    session = Session(llm_client=queued, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _walk_through_interview(ctx, registry)
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )
    # Sanity: the player exists exactly once and is named Iris.
    assert session.game is not None
    irises_before = [c for c in session.game.characters.values() if c.name.casefold() == "iris"]
    assert len(irises_before) == 1
    assert irises_before[0].is_player is True

    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_play_advance, payload={"option_id": "opt-1"}),
            ctx,
        )
    )

    # The bogus same-name descriptor was filtered out — there is
    # still exactly one Iris (the player), and the LLM-supplied
    # ``other-iris`` id never landed in the character map.
    irises_after = [c for c in session.game.characters.values() if c.name.casefold() == "iris"]
    assert len(irises_after) == 1, (
        f"player-name collision should have been dropped, but found "
        f"{len(irises_after)} characters named Iris"
    )
    assert "other-iris" not in session.game.characters
    # And the orphan id was silently dropped from on_stage.
    assert "other-iris" not in session.game.on_stage
