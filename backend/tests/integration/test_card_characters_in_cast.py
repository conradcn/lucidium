"""Character cards added on the New Game Review step land in the cast.

Same flow as ``test_side_characters_in_cast.py`` but with library cards
instead of typed one-liners:
  * a foreign card is adapted at Begin (one LLM call), keeps the stub
    id the renderer already holds, and carries its card into the game;
  * a card Lucidium exported earlier needs no LLM call at all;
  * cards and typed side characters mix in one confirm.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lucidium.api.handlers import HandlerContext, build_default_registry
from lucidium.api.messages import Envelope, MessageType
from lucidium.domain.character import Character
from lucidium.domain.settings import Settings
from lucidium.orchestration.card_import import card_from_character
from lucidium.orchestration.session import Session
from lucidium.persistence import character_cards as cc

from .test_side_characters_in_cast import (
    _add_side,
    _drain,
    _expansion,
    _NullImage,
    _options,
    _QueuedLlm,
    _walk_interview,
    _world_init_no_npcs,
)

_ADAPTATION = json.dumps(
    {
        "description": "A forest guardian, gentle but wary.",
        "kind": "human",
        "gender": "female",
        "pronouns": "she/her",
        "age": 300,
        "hair_color": "silver",
        "hairstyle": "long",
        "eye_color": "green",
        "outfit": "moss-green cloak",
        "facts": ["Guards the old glade."],
    }
)


def _session(tmp_app_data: Path, responses: list[str]) -> tuple[Session, Path]:
    lib = tmp_app_data / "my_cards"
    lib.mkdir()
    session = Session(
        settings=Settings(character_cards_dir=str(lib)),
        llm_client=_QueuedLlm(responses),
        image_client=_NullImage(),
    )
    return session, lib


async def _add_card(ctx: HandlerContext, registry, file: str) -> str:
    msgs = await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_add_card_character, payload={"file": file}),
            ctx,
        )
    )
    return msgs[0][1].ops[0].value["id"]


async def _confirm(ctx: HandlerContext, registry) -> None:
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )


@pytest.mark.asyncio
async def test_foreign_card_adapted_at_begin(tmp_app_data: Path) -> None:
    session, lib = _session(
        tmp_app_data,
        [
            _options([f"char-{i}" for i in range(6)]),
            _options([f"name-{i}" for i in range(8)]),
            _world_init_no_npcs(),
            _ADAPTATION,
        ],
    )
    card = cc.parse_card_json(
        {
            "spec": "chara_card_v2",
            "data": {"name": "Seraphine", "personality": "wary", "mes_example": "{{char}}: Hm."},
        }
    )
    (lib / "Seraphine.png").write_bytes(cc.embed_card_in_png(None, card))
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _walk_interview(ctx, registry)
    stub_id = await _add_card(ctx, registry, "Seraphine.png")
    await _confirm(ctx, registry)

    assert session.game is not None
    cast = session.game.characters
    assert stub_id in cast
    sera = cast[stub_id]
    assert sera.name == "Seraphine"
    assert sera.outfit == "moss-green cloak"
    assert sera.card is not None and sera.card.personality == "wary"
    # The card survives a save/load round trip.
    assert Character.model_validate_json(sera.model_dump_json()).card == sera.card


@pytest.mark.asyncio
async def test_lucidium_card_and_typed_side_character_mix(tmp_app_data: Path) -> None:
    # No adaptation reply queued: only the typed side character expands.
    session, lib = _session(
        tmp_app_data,
        [
            _options([f"char-{i}" for i in range(6)]),
            _options([f"name-{i}" for i in range(8)]),
            _world_init_no_npcs(),
            _expansion(name="Hale Stone"),
        ],
    )
    exported_from = Character(
        name="Mira",
        description="a sharp-eyed smuggler",
        gender="female",
        age=31,
        hair_color="black",
        hairstyle="cropped",
        outfit="tar-black oilskin coat",
        pose="leaning",
        expression="smirking",
        seed=77,
    )
    (lib / "Mira.png").write_bytes(
        cc.embed_card_in_png(None, card_from_character(exported_from, None))
    )
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _walk_interview(ctx, registry)
    mira_id = await _add_card(ctx, registry, "Mira.png")
    hale_id = await _add_side(ctx, registry, "grizzled lighthouse keeper")
    await _confirm(ctx, registry)

    assert session.game is not None
    cast = session.game.characters
    assert cast[hale_id].name == "Hale Stone"
    mira = cast[mira_id]
    assert mira.outfit == "tar-black oilskin coat"
    assert mira.seed == 77
    assert mira.description == "a sharp-eyed smuggler"
