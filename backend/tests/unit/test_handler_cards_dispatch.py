"""The character-card handlers, driven through ``dispatch``.

``c2s/cards/list``, ``c2s/new_game/add_card_character``,
``c2s/edit/character/add_from_card`` and
``c2s/edit/character/export_card``. The ``file`` handle is the only
path-like thing on the wire, so the path-confinement cases go through
dispatch on purpose: the payload-pattern check there is the first of
the two layers that keep a request inside the card library.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from lucidium.api.errors import NotFoundError, SchemaError
from lucidium.api.handlers import HandlerContext
from lucidium.api.messages import MessageType
from lucidium.domain.character_card import LUCIDIUM_EXTENSION_KEY
from lucidium.domain.settings import Settings
from lucidium.persistence import character_cards as cc

from .handler_harness import ScriptedLlm, dispatch, make_game, make_registry, make_session

_ADAPTATION = json.dumps(
    {
        "description": "A forest guardian, gentle but wary.",
        "kind": "human",
        "gender": "female",
        "age": 300,
        "hair_color": "silver",
        "outfit": "moss-green cloak",
        "facts": ["Guards the old glade."],
    }
)


def _write_v2_png(path: Path, name: str = "Seraphine") -> None:
    card = cc.parse_card_json(
        {"spec": "chara_card_v2", "data": {"name": name, "description": "{{char}} guards."}}
    )
    path.write_bytes(cc.embed_card_in_png(None, card))


def _session(tmp_path: Path, *, llm: Any | None = None):
    lib = tmp_path / "cards"
    lib.mkdir(exist_ok=True)
    session = make_session(
        tmp_path, settings=Settings(character_cards_dir=str(lib)), llm_client=llm
    )
    return session, lib


# ---------------------------------------------------------------------------
# c2s/cards/list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_cards(tmp_path: Path) -> None:
    session, lib = _session(tmp_path)
    _write_v2_png(lib / "Seraphine.png")
    (lib / "bad.json").write_text("{", encoding="utf-8")
    ctx = HandlerContext(session=session)

    msgs = await dispatch(make_registry(), ctx, MessageType.c2s_cards_list)
    assert [m[0] for m in msgs] == [MessageType.s2c_cards_list]
    reply = msgs[0][1]
    assert Path(reply.cards_dir) == lib.resolve()
    assert [c.file for c in reply.cards] == ["Seraphine.png"]
    card = reply.cards[0]
    assert card.name == "Seraphine"
    assert card.summary == "Seraphine guards."
    assert card.has_lucidium_data is False
    # Thumbnails live under the saves root so the asset protocol serves them.
    assert Path(card.thumbnail_path).is_relative_to(session.saves_root)
    assert [e.file for e in reply.errors] == ["bad.json"]


@pytest.mark.asyncio
async def test_list_creates_default_library(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    msgs = await dispatch(
        make_registry(), HandlerContext(session=session), MessageType.c2s_cards_list
    )
    cards_dir = Path(msgs[0][1].cards_dir)
    assert cards_dir == (session.saves_root.parent / "character_cards").resolve()
    assert cards_dir.is_dir()  # noqa: ASYNC240


@pytest.mark.asyncio
async def test_list_takes_no_directory_from_the_wire(tmp_path: Path) -> None:
    session, _lib = _session(tmp_path)
    with pytest.raises(SchemaError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_cards_list,
            {"directory": "C:\\Windows"},
        )


# ---------------------------------------------------------------------------
# path confinement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file",
    ["../outside.png", "..\\outside.png", "C:\\x.png", "sub/x.png", "a\x00.png", ""],
)
@pytest.mark.parametrize(
    "message_type",
    [MessageType.c2s_new_game_add_card_character, MessageType.c2s_edit_character_add_from_card],
)
async def test_file_handle_must_be_bare_name(
    tmp_path: Path, file: str, message_type: MessageType
) -> None:
    session, _lib = _session(tmp_path)
    _write_v2_png(tmp_path / "outside.png")
    game, *_ = make_game()
    session.install_game(game)
    with pytest.raises(SchemaError):
        await dispatch(
            make_registry(), HandlerContext(session=session), message_type, {"file": file}
        )


@pytest.mark.asyncio
async def test_symlink_out_of_library_refused(tmp_path: Path) -> None:
    session, lib = _session(tmp_path)
    target = tmp_path / "outside.png"
    _write_v2_png(target)
    try:
        os.symlink(target, lib / "link.png")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform/user")
    with pytest.raises(SchemaError, match="outside"):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_new_game_add_card_character,
            {"file": "link.png"},
        )


@pytest.mark.asyncio
async def test_missing_and_unreadable_cards(tmp_path: Path) -> None:
    session, lib = _session(tmp_path)
    ctx = HandlerContext(session=session)
    with pytest.raises(NotFoundError):
        await dispatch(
            make_registry(), ctx, MessageType.c2s_new_game_add_card_character, {"file": "nope.png"}
        )
    (lib / "plain.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(SchemaError, match=r"plain.json"):
        await dispatch(
            make_registry(),
            ctx,
            MessageType.c2s_new_game_add_card_character,
            {"file": "plain.json"},
        )


# ---------------------------------------------------------------------------
# c2s/new_game/add_card_character (+ edit / delete of the resulting row)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_game_add_card_defers_adaptation(tmp_path: Path) -> None:
    llm = ScriptedLlm([])
    session, lib = _session(tmp_path, llm=llm)
    _write_v2_png(lib / "Seraphine.png")
    registry = make_registry()
    ctx = HandlerContext(session=session)

    msgs = await dispatch(
        registry, ctx, MessageType.c2s_new_game_add_card_character, {"file": "Seraphine.png"}
    )
    assert llm.prompts == []  # no LLM call until Begin
    op = msgs[0][1].ops[0]
    assert op.op == "add"
    assert op.value["from_card"] is True
    stub_id = op.value["id"]
    interview = session.interview
    assert interview.pending_card_ids == [stub_id]
    assert interview.side_characters[0].card is not None
    assert interview.side_character_descriptions[0].startswith("Seraphine (from an imported")

    # Rename keeps the card text, carries the name onto the card.
    await dispatch(
        registry,
        ctx,
        MessageType.c2s_new_game_edit_side_character,
        {"character_id": stub_id, "description": "Sera"},
    )
    side = interview.side_characters[0]
    assert side.name == "Sera"
    assert side.card is not None and side.card.name == "Sera"
    assert side.description != "Sera"
    assert interview.side_character_descriptions[0].startswith("Sera (from an imported")

    # Delete drops it from the pending list too.
    await dispatch(
        registry, ctx, MessageType.c2s_new_game_delete_side_character, {"character_id": stub_id}
    )
    assert interview.side_characters == []
    assert interview.pending_card_ids == []


@pytest.mark.asyncio
async def test_new_game_add_lucidium_card_is_final_immediately(tmp_path: Path) -> None:
    from lucidium.orchestration.card_import import card_from_character

    session, lib = _session(tmp_path)
    game, _ids, _player, npc = make_game()
    (lib / "Mira.png").write_bytes(cc.embed_card_in_png(None, card_from_character(npc, game)))

    await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_new_game_add_card_character,
        {"file": "Mira.png"},
    )
    side = session.interview.side_characters[0]
    assert session.interview.pending_card_ids == []
    assert side.outfit == npc.outfit
    assert side.hair_color == npc.hair_color


# ---------------------------------------------------------------------------
# c2s/edit/character/add_from_card
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_from_card_mid_story(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = ScriptedLlm([_ADAPTATION])
    session, lib = _session(tmp_path, llm=llm)
    _write_v2_png(lib / "Seraphine.png")
    game, *_ = make_game()
    session.install_game(game)
    redraws: list[str] = []
    monkeypatch.setattr(session.render_scheduler, "request_redraw", redraws.append)

    msgs = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_edit_character_add_from_card,
        {"file": "Seraphine.png"},
    )
    assert [m[0] for m in msgs] == [MessageType.s2c_state_full]
    assert len(llm.prompts) == 1
    assert session.game is not None
    added = next(c for c in session.game.characters.values() if c.name == "Seraphine")
    assert added.outfit == "moss-green cloak"
    assert added.card is not None and added.card.source_file == "Seraphine.png"
    assert added.id in session.game.on_stage
    assert redraws == [added.id]
    # Undoable: the snapshot from before the add is on the stack.
    assert session.undo_stack
    assert added.id not in session.undo_stack[-1].characters


@pytest.mark.asyncio
async def test_add_from_card_off_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session, lib = _session(tmp_path, llm=ScriptedLlm([_ADAPTATION]))
    _write_v2_png(lib / "Seraphine.png")
    game, *_ = make_game()
    session.install_game(game)
    monkeypatch.setattr(session.render_scheduler, "request_redraw", lambda _id: None)

    await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_edit_character_add_from_card,
        {"file": "Seraphine.png", "on_stage": False},
    )
    assert session.game is not None
    added = next(c for c in session.game.characters.values() if c.name == "Seraphine")
    assert added.id not in session.game.on_stage


@pytest.mark.asyncio
async def test_add_from_card_requires_game(tmp_path: Path) -> None:
    session, lib = _session(tmp_path)
    _write_v2_png(lib / "Seraphine.png")
    with pytest.raises(NotFoundError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_edit_character_add_from_card,
            {"file": "Seraphine.png"},
        )


# ---------------------------------------------------------------------------
# c2s/edit/character/export_card
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_export_card_writes_png_and_reimports(tmp_path: Path) -> None:
    session, lib = _session(tmp_path)
    game, _ids, _player, npc = make_game()
    portrait = tmp_path / "mira.png"
    portrait.write_bytes(cc.embed_card_in_png(None, cc.parse_card_json({"name": "x"})))
    from lucidium.domain.character import CharacterImage

    images = [CharacterImage(path=str(portrait), prompt_hash="h" * 64, attributes_snapshot={})]
    npc = npc.model_copy(update={"images": images})
    session.install_game(game.model_copy(update={"characters": {**game.characters, npc.id: npc}}))
    registry = make_registry()
    ctx = HandlerContext(session=session)

    msgs = await dispatch(
        registry, ctx, MessageType.c2s_edit_character_export_card, {"character_id": npc.id}
    )
    assert [m[0] for m in msgs] == [MessageType.s2c_cards_exported]
    reply = msgs[0][1]
    assert reply.file == "Mira.png"
    written = lib / "Mira.png"
    assert Path(reply.path) == written
    loaded = cc.load_card_file(written)
    assert loaded.card.name == "Mira"
    assert LUCIDIUM_EXTENSION_KEY in loaded.card.extensions

    # Both chunks present so V2-only readers (older SillyTavern) see it.
    data = written.read_bytes()
    keys = {t[0] for c in cc._iter_png_chunks(data) if (t := cc._text_chunk(c)) is not None}
    assert keys == {"chara", "ccv3"}
    v2 = next(
        t[1] for c in cc._iter_png_chunks(data) if (t := cc._text_chunk(c)) and t[0] == "chara"
    )
    assert json.loads(base64.b64decode(v2))["data"]["name"] == "Mira"

    # A second export doesn't overwrite the first.
    msgs = await dispatch(
        registry, ctx, MessageType.c2s_edit_character_export_card, {"character_id": npc.id}
    )
    assert msgs[0][1].file == "Mira (2).png"

    # The listing flags it as instantly importable.
    listing = await dispatch(registry, ctx, MessageType.c2s_cards_list)
    assert all(c.has_lucidium_data for c in listing[0][1].cards)


@pytest.mark.asyncio
async def test_export_unknown_character(tmp_path: Path) -> None:
    session, _lib = _session(tmp_path)
    game, *_ = make_game()
    session.install_game(game)
    with pytest.raises(NotFoundError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_edit_character_export_card,
            {"character_id": "ghost"},
        )


def test_payload_pattern_rejects_separators() -> None:
    from lucidium.api.messages import C2SNewGameAddCardCharacter

    for bad in ("a/b.png", "a\\b.png", "c:x.png", "x\ny.png"):
        with pytest.raises(ValidationError):
            C2SNewGameAddCardCharacter(file=bad)
    assert C2SNewGameAddCardCharacter(file="Zoë (2).png").file == "Zoë (2).png"
