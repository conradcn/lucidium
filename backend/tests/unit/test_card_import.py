"""Card <-> Character conversion (``orchestration.card_import``).

The two properties that matter most: a card Lucidium exported comes
back as the *same* character without an LLM call, and a foreign card
always yields a usable cast member — even when the adaptation reply is
garbage.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from lucidium.domain.character import DEFAULT_OUTFIT, CharacterKind
from lucidium.domain.character_card import LUCIDIUM_EXTENSION_KEY, CharacterCardData
from lucidium.orchestration import card_import as ci
from lucidium.orchestration.prompts.common import render_character_full

from .handler_harness import make_character, make_game


class _LlmSession:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[Any] = []

    async def llm_text(self, prompt: Any, **_kw: Any) -> tuple[str, list[str]]:
        self.prompts.append(prompt)
        return self.reply, []


def _foreign_card(**kw: Any) -> CharacterCardData:
    fields: dict[str, Any] = {
        "name": "Seraphine",
        "description": "{{char}} guards the forest and distrusts {{user}}.",
        "personality": "gentle but wary",
        "mes_example": "<START>\n{{char}}: Stay close, {{user}}.",
        "character_book": {
            "entries": [{"content": "The glade is old."}, {"enabled": False, "content": "hidden"}]
        },
    }
    fields.update(kw)
    return CharacterCardData(**fields)


def _adaptation(**overrides: Any) -> str:
    body = {
        "description": "A forest guardian, gentle but wary.",
        "kind": "human",
        "gender": "female",
        "pronouns": "she/her",
        "age": 300,
        "hair_color": "silver",
        "hairstyle": "long",
        "eye_color": "green",
        "outfit": "moss-green cloak",
        "facts": ["Guards the old glade.", "  ", "Distrusts outsiders."],
    }
    body.update(overrides)
    return json.dumps(body)


def test_expand_macros() -> None:
    text = "{{char}} greets {{ User }} — <BOT> and <USER>"
    assert ci.expand_macros(text, char_name="Sera", user_name="Iris") == (
        "Sera greets Iris — Sera and Iris"
    )


@pytest.mark.asyncio
async def test_adapt_foreign_card_uses_llm_and_keeps_card() -> None:
    session = _LlmSession(_adaptation())
    card = _foreign_card()
    character = await ci.adapt_card(session, card, setting="harbor", player_name="Iris")  # type: ignore[arg-type]

    assert len(session.prompts) == 1
    user_prompt = session.prompts[0][-1]["content"]
    # Macros are expanded with the player's name; disabled lore is dropped.
    assert "distrusts Iris" in user_prompt
    assert "The glade is old." in user_prompt
    assert "hidden" not in user_prompt

    assert character.name == "Seraphine"
    assert character.hair_color == "silver"
    assert character.outfit == "moss-green cloak"
    assert [f.text for f in character.facts] == ["Guards the old glade.", "Distrusts outsiders."]
    assert character.card == card


@pytest.mark.asyncio
async def test_adapt_keeps_stub_identity() -> None:
    card = _foreign_card()
    stub = ci.card_stub(card)
    character = await ci.adapt_card(
        _LlmSession(_adaptation()),  # type: ignore[arg-type]
        card,
        setting="",
        player_name="",
        base=stub,
    )
    assert character.id == stub.id
    assert character.seed == stub.seed


@pytest.mark.asyncio
async def test_adapt_garbage_reply_falls_back_to_card_text() -> None:
    card = _foreign_card()
    character = await ci.adapt_card(
        _LlmSession("I'm sorry, I can't do JSON today"),  # type: ignore[arg-type]
        card,
        setting="",
        player_name="",
    )
    assert character.name == "Seraphine"
    assert "guards the forest" in character.description
    assert character.outfit == DEFAULT_OUTFIT
    assert character.kind == CharacterKind.human


@pytest.mark.asyncio
async def test_adapt_unknown_kind_defaults_to_human() -> None:
    character = await ci.adapt_card(
        _LlmSession(_adaptation(kind="dragon")),  # type: ignore[arg-type]
        _foreign_card(),
        setting="",
        player_name="",
    )
    assert character.kind == CharacterKind.human


@pytest.mark.asyncio
async def test_lucidium_export_reimports_exactly_without_llm() -> None:
    game, _ids, _player, npc = make_game()
    npc = npc.model_copy(update={"pronouns": "she/her", "decals": "a scar"})
    exported = ci.card_from_character(npc, game)

    session = _LlmSession("should not be called")
    back = await ci.adapt_card(session, exported, setting="", player_name="")  # type: ignore[arg-type]
    assert session.prompts == []
    for field in (
        "name",
        "description",
        "kind",
        "gender",
        "pronouns",
        "age",
        "hair_color",
        "hairstyle",
        "eye_color",
        "outfit",
        "pose",
        "expression",
        "decals",
        "seed",
    ):
        assert getattr(back, field) == getattr(npc, field), field
    # A fresh id — re-importing is a new cast member, not the same one.
    assert back.id != npc.id


def test_bad_lucidium_record_is_ignored() -> None:
    card = CharacterCardData(
        name="X", extensions={LUCIDIUM_EXTENSION_KEY: {"character": {"age": "not a number"}}}
    )
    assert ci.character_from_extension(card) is None
    assert ci.character_from_extension(CharacterCardData(name="Y")) is None


def test_export_invented_character_composes_card() -> None:
    game, ids, _player, npc = make_game(node_count=3)
    nodes = dict(game.dialog_tree.nodes)
    nodes[ids[1]] = nodes[ids[1]].model_copy(update={"speaker_id": npc.id, "text": "First line."})
    nodes[ids[2]] = nodes[ids[2]].model_copy(update={"speaker_id": npc.id, "text": "Second line."})
    game = game.model_copy(
        update={"dialog_tree": game.dialog_tree.model_copy(update={"nodes": nodes})}
    )

    card = ci.card_from_character(npc, game)
    assert card.name == "Mira"
    assert "[Appearance]" in card.description
    assert "Hair: auburn braid" in card.description
    assert card.first_mes == "First line."
    assert "{{char}}: Second line." in card.mes_example
    assert "Setting: stone harbor" in card.scenario
    assert "mystery" in card.tags
    assert "Embers" in card.creator_notes
    assert card.extensions[LUCIDIUM_EXTENSION_KEY]["character"]["outfit"] == "wool coat"


def test_export_card_backed_character_preserves_original() -> None:
    original = _foreign_card(tags=["fantasy"], extensions={"talkativeness": "0.5"})
    npc = make_character(name="Sera (renamed)").model_copy(update={"card": original})
    card = ci.card_from_character(npc, None)
    assert card.name == "Sera (renamed)"
    assert card.description == original.description
    assert card.character_book == original.character_book
    assert card.extensions["talkativeness"] == "0.5"
    assert LUCIDIUM_EXTENSION_KEY in card.extensions


def test_card_prompt_lines_feed_storyteller() -> None:
    plain = make_character(name="Plain")
    assert ci.card_prompt_lines(plain) == ""
    assert "from character card" not in render_character_full(plain)

    carded = make_character(name="Sera").model_copy(
        update={"card": _foreign_card(personality="x" * 5000)}
    )
    rendered = render_character_full(carded)
    assert "personality (from character card)" in rendered
    assert "speech sample (from character card): Sera: Stay close, the protagonist." in rendered
    # Bounded so one huge card can't crowd out the prompt.
    assert len(ci.card_prompt_lines(carded)) < 1500


def test_resolve_cards_dir(tmp_path) -> None:
    saves = tmp_path / "app" / "saves"
    assert ci.resolve_cards_dir("", saves) == (tmp_path / "app" / "character_cards").resolve()
    assert ci.resolve_cards_dir(str(tmp_path / "mine"), saves) == (tmp_path / "mine").resolve()
