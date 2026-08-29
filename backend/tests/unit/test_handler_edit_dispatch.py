"""``c2s/edit/world``, ``c2s/edit/history`` and the character
show/dismiss pair, through dispatch.

``edit/world`` writes straight into the persisted ``WorldState``, so it
carries an explicit field allowlist (``_EDITABLE_WORLD_FIELDS``). Two
things depend on that allowlist and neither had a test: a field outside
it must be refused, and a field inside it must still go through the
model's own validators rather than ``model_copy``'s unchecked
``__dict__`` write — the latter is what would otherwise let one edit
frame persist a ``game.json`` that no longer loads.

``edit/history`` has no field surface (it names a node and new text), so
its equivalent guard is the node-id lookup.

show/dismiss are inverses that both mutate ``removed`` + ``on_stage`` and
both push an undo snapshot; they are tested as a round trip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lucidium.api.errors import NotFoundError, SchemaError
from lucidium.api.handlers import HandlerContext
from lucidium.api.messages import MessageType
from lucidium.persistence import save_store

from .handler_harness import (
    dispatch,
    make_game,
    make_registry,
    make_session,
    types_of,
)


def _loaded_session(tmp_app_data: Path, *, node_count: int = 3):
    session = make_session(tmp_app_data)
    game, ids, player, npc = make_game(node_count=node_count)
    session.install_game(game)
    return session, ids, player, npc


# ---------------------------------------------------------------------------
# c2s/edit/world
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("game_name", "Ashes of the Quay"),
        ("setting", "a drowned lighthouse"),
        ("genre", "gothic horror"),
        ("visual_style", "charcoal"),
        ("overall_plot_direction", "the tide never goes out"),
        ("prompt_history_clamp_chars", 4000),
    ],
)
async def test_edit_world_updates_an_allowlisted_field_and_commits(
    tmp_app_data: Path, field: str, value: object
) -> None:
    session, _ids, _player, _npc = _loaded_session(tmp_app_data)

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_edit_world,
        {"field": field, "value": value},
    )

    assert types_of(messages) == [MessageType.s2c_state_patch]
    op = messages[0][1].ops[0]
    assert op.op == "replace"
    assert op.path == f"/world/{field}"
    assert op.value == value
    assert getattr(session.game.world, field) == value
    # It was persisted, not just held in memory.
    persisted = save_store.load_save(session.game.id, root=session.saves_root)
    assert getattr(persisted.world, field) == value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    [
        # Never editable: identity, engine-managed state, and a few
        # plausible-looking near-misses.
        "id",
        "music_prompt",
        "facts",
        "__class__",
        "model_config",
        "../../etc/passwd",
        "",
    ],
)
async def test_edit_world_refuses_a_field_outside_the_allowlist(
    tmp_app_data: Path, field: str
) -> None:
    """The abuse case: an arbitrary attribute name off the wire must not
    reach ``WorldState``. The game must be left exactly as it was."""
    session, _ids, _player, _npc = _loaded_session(tmp_app_data)
    before = session.game.world.model_dump()

    with pytest.raises(SchemaError) as excinfo:
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_edit_world,
            {"field": field, "value": "anything"},
        )

    assert "not editable" in str(excinfo.value)
    assert session.game.world.model_dump() == before
    # Nothing was committed either.
    assert not (session.saves_root / session.game.id).exists()


@pytest.mark.asyncio
async def test_edit_world_refuses_a_value_the_model_rejects(
    tmp_app_data: Path,
) -> None:
    """``prompt_history_clamp_chars`` is an int. A string for it is an
    allowlisted FIELD with an invalid VALUE — caught by re-validating the
    model, not by the allowlist."""
    session, _ids, _player, _npc = _loaded_session(tmp_app_data)

    with pytest.raises(SchemaError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_edit_world,
            {"field": "prompt_history_clamp_chars", "value": "not-a-number"},
        )


@pytest.mark.asyncio
async def test_edit_world_without_a_game_is_a_not_found(tmp_app_data: Path) -> None:
    session = make_session(tmp_app_data)
    with pytest.raises(NotFoundError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_edit_world,
            {"field": "genre", "value": "mystery"},
        )


# ---------------------------------------------------------------------------
# c2s/edit/history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_history_rewrites_one_beat_and_commits(tmp_app_data: Path) -> None:
    session, ids, _player, _npc = _loaded_session(tmp_app_data)
    target = ids[0]

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_edit_history,
        {"node_id": target, "new_text": "The harbor was already burning."},
    )

    assert types_of(messages) == [MessageType.s2c_state_patch]
    op = messages[0][1].ops[0]
    assert op.path == f"/dialog_tree/nodes/{target}/text"
    assert op.value == "The harbor was already burning."
    assert session.game.dialog_tree.nodes[target].text == "The harbor was already burning."
    # Other beats are untouched.
    assert session.game.dialog_tree.nodes[ids[1]].text == "beat 1"
    persisted = save_store.load_save(session.game.id, root=session.saves_root)
    assert persisted.dialog_tree.nodes[target].text == "The harbor was already burning."


@pytest.mark.asyncio
@pytest.mark.parametrize("node_id", ["no-such-node", "../../etc/passwd", "", "n99"])
async def test_edit_history_refuses_an_unknown_node_id(tmp_app_data: Path, node_id: str) -> None:
    """``edit/history`` has no field allowlist — its analogous guard is
    the node-id membership check. An id that isn't in the tree must not
    create one, and must not commit."""
    session, _ids, _player, _npc = _loaded_session(tmp_app_data)
    before = set(session.game.dialog_tree.nodes)

    with pytest.raises(NotFoundError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_edit_history,
            {"node_id": node_id, "new_text": "injected"},
        )

    assert set(session.game.dialog_tree.nodes) == before
    assert not (session.saves_root / session.game.id).exists()


@pytest.mark.asyncio
async def test_edit_history_without_a_game_is_a_not_found(tmp_app_data: Path) -> None:
    session = make_session(tmp_app_data)
    with pytest.raises(NotFoundError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_edit_history,
            {"node_id": "n0", "new_text": "x"},
        )


# ---------------------------------------------------------------------------
# c2s/edit/character/dismiss + show
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dismiss_marks_removed_and_drops_from_stage(tmp_app_data: Path) -> None:
    session, _ids, _player, npc = _loaded_session(tmp_app_data)
    assert npc.id in session.game.on_stage

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_edit_character_dismiss,
        {"character_id": npc.id, "reason": "wandered off"},
    )

    assert types_of(messages) == [MessageType.s2c_state_patch]
    paths = {op.path for op in messages[0][1].ops}
    assert paths == {
        f"/characters/{npc.id}/removed",
        f"/characters/{npc.id}/removed_reason",
        "/on_stage",
    }
    updated = session.game.characters[npc.id]
    assert updated.removed is True
    assert updated.removed_reason == "wandered off"
    assert npc.id not in session.game.on_stage
    # Reversible: the pre-dismiss game is on the undo stack.
    assert session.undo_stack


@pytest.mark.asyncio
async def test_dismiss_defaults_the_reason(tmp_app_data: Path) -> None:
    session, _ids, _player, npc = _loaded_session(tmp_app_data)

    await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_edit_character_dismiss,
        {"character_id": npc.id},
    )

    assert session.game.characters[npc.id].removed_reason == "dismissed"


@pytest.mark.asyncio
async def test_dismiss_refuses_the_player_character(tmp_app_data: Path) -> None:
    """The PC is the camera lens — removing them leaves the run with no
    viewpoint at all."""
    session, _ids, player, _npc = _loaded_session(tmp_app_data)

    with pytest.raises(SchemaError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_edit_character_dismiss,
            {"character_id": player.id},
        )

    assert session.game.characters[player.id].removed is False


@pytest.mark.asyncio
async def test_show_restores_a_dismissed_character(tmp_app_data: Path) -> None:
    session, _ids, _player, npc = _loaded_session(tmp_app_data)
    registry = make_registry()
    ctx = HandlerContext(session=session)

    await dispatch(
        registry,
        ctx,
        MessageType.c2s_edit_character_dismiss,
        {"character_id": npc.id, "reason": "wandered off"},
    )
    messages = await dispatch(
        registry,
        ctx,
        MessageType.c2s_edit_character_show,
        {"character_id": npc.id},
    )

    assert types_of(messages) == [MessageType.s2c_state_patch]
    restored = session.game.characters[npc.id]
    assert restored.removed is False
    assert restored.removed_reason == ""
    assert npc.id in session.game.on_stage
    # A round trip is a true inverse for the fields it owns.
    assert session.game.on_stage == [npc.id]


@pytest.mark.asyncio
async def test_show_does_not_put_the_player_character_on_stage(
    tmp_app_data: Path,
) -> None:
    """The PC is never an on-stage portrait — showing them must clear
    ``removed`` without adding them to the stage list."""
    session, _ids, player, _npc = _loaded_session(tmp_app_data)

    await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_edit_character_show,
        {"character_id": player.id},
    )

    assert session.game.characters[player.id].removed is False
    assert player.id not in session.game.on_stage


@pytest.mark.asyncio
async def test_show_is_idempotent_for_a_character_already_on_stage(
    tmp_app_data: Path,
) -> None:
    session, _ids, _player, npc = _loaded_session(tmp_app_data)

    await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_edit_character_show,
        {"character_id": npc.id},
    )

    assert session.game.on_stage.count(npc.id) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message_type",
    [
        MessageType.c2s_edit_character_show,
        MessageType.c2s_edit_character_dismiss,
    ],
)
async def test_show_and_dismiss_refuse_an_unknown_character(
    tmp_app_data: Path, message_type: MessageType
) -> None:
    session, _ids, _player, _npc = _loaded_session(tmp_app_data)
    before = set(session.game.characters)

    with pytest.raises(NotFoundError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            message_type,
            {"character_id": "nobody"},
        )

    assert set(session.game.characters) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message_type",
    [
        MessageType.c2s_edit_character_show,
        MessageType.c2s_edit_character_dismiss,
    ],
)
async def test_show_and_dismiss_without_a_game_are_not_found(
    tmp_app_data: Path, message_type: MessageType
) -> None:
    session = make_session(tmp_app_data)
    with pytest.raises(NotFoundError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            message_type,
            {"character_id": "anyone"},
        )
