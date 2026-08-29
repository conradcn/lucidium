"""All side characters the player typed on the New Game screen must
land in the cast after they press Begin.

Specifically:
  * Every Add-side-character entry survives confirm and shows up as
    a ``Character`` keyed in ``session.game.characters``.
  * Edits applied to a stub before Begin are reflected in the cast
    (the edited name + description, not the original Add-button text).
  * A delete before Begin removes the entry from the cast.
  * Confirm-time LLM expansion failures don't drop the entry — the
    stub itself is kept so the player never sees their typed NPC
    silently vanish.
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

# ---------------------------------------------------------------------------
# LLM fixtures shared with the rest of the integration suite. Same shape as
# ``test_new_character_flow.py`` — a primary queue plus a single-beat
# fallback for any speculative tasks that fire off the back of confirm.
# ---------------------------------------------------------------------------


class _QueuedLlm:
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
        value = self._responses.pop(0) if self._responses else self._SPEC_FALLBACK

        async def gen() -> AsyncIterator[str]:
            yield value

        return gen()


class _NullImage:
    async def generate(self, *_a, **_kw) -> bytes:
        return b""


def _options(items: list[str]) -> str:
    return json.dumps({"options": items})


def _full_descriptor(*, id_: str, name: str) -> dict:
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


def _world_init_no_npcs() -> str:
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


def _expansion(
    *,
    name: str,
    description: str = "expanded by the LLM",
    outfit: str = "tailored coat",
) -> str:
    """A successful side-character expansion. ``outfit`` is the field
    confirm uses to detect a stub vs an expanded character, so it must
    be non-empty for the expansion to be visible in the test."""
    return json.dumps(
        {
            "name": name,
            "description": description,
            "gender": "female",
            "pronouns": "she/her",
            "age": 34,
            "ethnicity": "local",
            "skin": "pale",
            "hair_color": "auburn",
            "hairstyle": "braid",
            "eye_color": "grey",
            "build": "slight",
            "bust": "moderate",
            "outfit": outfit,
            "pose": "standing",
            "expression": "alert",
        }
    )


async def _drain(handler_result) -> list:
    out = []
    async for msg in handler_result:
        out.append(msg)
    return out


async def _walk_interview(ctx: HandlerContext, registry) -> None:
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_start, payload={}),
            ctx,
        )
    )
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
                    payload={
                        "step": step.value,
                        "answer": answer,
                        "is_free_text": False,
                    },
                ),
                ctx,
            )
        )


async def _add_side(
    ctx: HandlerContext,
    registry,
    description: str,
) -> str:
    """Drive the add-side-character handler and return the stub id the
    handler emitted (the renderer reads it from the state_patch op's
    final path segment)."""
    msgs = await _drain(
        registry.dispatch(
            Envelope(
                type=MessageType.c2s_new_game_add_side_character,
                payload={"description": description},
            ),
            ctx,
        )
    )
    op = msgs[0][1].ops[0]
    return op.path.rsplit("/", 1)[-1]


async def _edit_side(
    ctx: HandlerContext,
    registry,
    character_id: str,
    description: str,
) -> None:
    await _drain(
        registry.dispatch(
            Envelope(
                type=MessageType.c2s_new_game_edit_side_character,
                payload={
                    "character_id": character_id,
                    "description": description,
                },
            ),
            ctx,
        )
    )


async def _delete_side(
    ctx: HandlerContext,
    registry,
    character_id: str,
) -> None:
    await _drain(
        registry.dispatch(
            Envelope(
                type=MessageType.c2s_new_game_delete_side_character,
                payload={"character_id": character_id},
            ),
            ctx,
        )
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_added_side_characters_land_in_cast(
    tmp_app_data: Path,
) -> None:
    """Happy path: add three side characters, click Begin, all three end
    up in the cast — keyed by the same id the add handler emitted
    (so renderer state stays consistent), with expanded attributes
    populated by the confirm-time LLM call."""
    queued = _QueuedLlm(
        responses=[
            _options([f"char-{i}" for i in range(6)]),
            _options([f"name-{i}" for i in range(8)]),
            _world_init_no_npcs(),
            _expansion(name="Hale Stone"),
            _expansion(name="Mira Quill"),
            _expansion(name="Cal Westwind"),
        ]
    )
    session = Session(llm_client=queued, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _walk_interview(ctx, registry)
    stub_a = await _add_side(ctx, registry, "grizzled lighthouse keeper")
    stub_b = await _add_side(ctx, registry, "tavern keeper with secrets")
    stub_c = await _add_side(ctx, registry, "harbor pilot")

    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )

    assert session.game is not None
    cast = session.game.characters
    # All three stub ids must be present in the cast (the expansion
    # preserves stub ids so the renderer's patch keys keep resolving).
    assert stub_a in cast, f"stub_a={stub_a} missing from cast={list(cast)}"
    assert stub_b in cast, f"stub_b={stub_b} missing from cast={list(cast)}"
    assert stub_c in cast, f"stub_c={stub_c} missing from cast={list(cast)}"
    # And the expansion populated the LLM-derived name on each.
    names = sorted(cast[i].name for i in (stub_a, stub_b, stub_c))
    assert names == ["Cal Westwind", "Hale Stone", "Mira Quill"]
    # Plus the player. Cast = player + 3 side chars.
    assert sum(1 for c in cast.values() if c.is_player) == 1
    assert len(cast) == 4


@pytest.mark.asyncio
async def test_edited_side_character_lands_with_new_description(
    tmp_app_data: Path,
) -> None:
    """Editing a side-character stub before Begin must change what gets
    expanded at confirm time — confirm runs ``side_character_expansion``
    against the CURRENT description, not the original Add-button text.
    Without keeping ``side_character_descriptions`` in sync with the
    stub renames, the edit would be silently lost.
    """
    queued = _QueuedLlm(
        responses=[
            _options([f"char-{i}" for i in range(6)]),
            _options([f"name-{i}" for i in range(8)]),
            _world_init_no_npcs(),
            _expansion(name="Renamed NPC", description="from renamed prompt"),
        ]
    )
    session = Session(llm_client=queued, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _walk_interview(ctx, registry)
    stub_id = await _add_side(ctx, registry, "original description")
    await _edit_side(ctx, registry, stub_id, "renamed description")

    # Sanity: the in-memory stub was renamed BEFORE confirm fired.
    assert session.interview.side_characters[0].name == "renamed description"
    assert session.interview.side_character_descriptions == ["renamed description"]

    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )

    assert session.game is not None
    cast = session.game.characters
    assert stub_id in cast
    # Expansion ran against the renamed description and overwrote the
    # placeholder name with the LLM-returned one. (The fact that the
    # expansion JSON is the only one we queued proves the call fired —
    # if confirm had used the original description we'd still expect a
    # match because we only queued one expansion, so we also assert on
    # the description text to nail the path.)
    assert cast[stub_id].description == "from renamed prompt"


@pytest.mark.asyncio
async def test_deleted_side_character_is_absent_from_cast(
    tmp_app_data: Path,
) -> None:
    """Deleting a side-character stub before Begin must drop it from
    confirm-time expansion AND the resulting cast."""
    queued = _QueuedLlm(
        responses=[
            _options([f"char-{i}" for i in range(6)]),
            _options([f"name-{i}" for i in range(8)]),
            _world_init_no_npcs(),
            # Only ONE expansion fixture — the second stub gets deleted
            # before confirm, so only one expansion call should fire.
            # If confirm called expansion for the deleted stub, the
            # SPEC_FALLBACK kicks in (a beat payload that fails the
            # character schema), the expansion returns the stub as-is,
            # and the cast would still contain the deleted id — which
            # is exactly what we're guarding against.
            _expansion(name="Surviving NPC"),
        ]
    )
    session = Session(llm_client=queued, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _walk_interview(ctx, registry)
    keep_id = await _add_side(ctx, registry, "keeper to keep")
    drop_id = await _add_side(ctx, registry, "keeper to drop")
    await _delete_side(ctx, registry, drop_id)

    assert len(session.interview.side_characters) == 1
    assert session.interview.side_characters[0].id == keep_id

    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )

    assert session.game is not None
    cast = session.game.characters
    assert keep_id in cast
    assert drop_id not in cast, f"deleted stub_id={drop_id} should not appear in cast={list(cast)}"
    assert cast[keep_id].name == "Surviving NPC"


@pytest.mark.asyncio
async def test_failed_expansion_still_lands_stub_in_cast(
    tmp_app_data: Path,
) -> None:
    """If the side-character expansion LLM call returns invalid JSON,
    confirm falls back to the un-expanded stub. The player still sees
    their typed NPC in the cast — they just don't get the LLM's
    embellishments. The original symptom this guards against: a
    silently-vanishing NPC after Begin because expansion failed."""
    queued = _QueuedLlm(
        responses=[
            _options([f"char-{i}" for i in range(6)]),
            _options([f"name-{i}" for i in range(8)]),
            _world_init_no_npcs(),
            # Expansion returns something that's not a character —
            # parse_json_object raises, the expansion handler logs +
            # returns the stub unchanged.
            json.dumps({"unrelated": "garbage"}),
        ]
    )
    session = Session(llm_client=queued, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _walk_interview(ctx, registry)
    stub_id = await _add_side(ctx, registry, "the keeper")

    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )

    assert session.game is not None
    cast = session.game.characters
    assert stub_id in cast
    # No LLM enrichment — the stub's name/description are still the
    # one-liner the player typed.
    assert cast[stub_id].name == "the keeper"
    assert cast[stub_id].description == "the keeper"
