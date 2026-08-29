"""``c2s/edit/*`` field updates run through Pydantic validation.

Bug shape: the three live-game edit handlers built their updated
model with ``model_copy(update={payload.field: payload.value})``.
In Pydantic v2 that writes straight into ``__dict__`` with no
validators, and the payloads are ``field: str`` / ``value: Any``
with no allow-list. So one ``c2s/edit/environment`` frame carrying
``{"field": "seed", "value": "x"}`` committed a ``game.json``
holding ``"seed": "x"`` — which can never be re-validated, i.e. the
save is permanently unloadable, with no repair path.

Secondarily, ``Environment.image_path`` (and every other
filesystem-path field) was settable this way and flows into the
asset handler, making the edit frame a local-file-read primitive.

Fix: an explicit per-model frozenset of editable fields, plus
``Model.model_validate({**existing, field: value})`` so validators
run. Both failure modes surface as ``SchemaError`` — a client
mistake — and nothing is installed or committed.
"""

from __future__ import annotations

import pytest

from lucidium.api.errors import SchemaError
from lucidium.api.handlers import (
    HandlerContext,
    edit_character_handler,
    edit_environment_handler,
    edit_world_handler,
)
from lucidium.api.messages import (
    C2SEditCharacter,
    C2SEditEnvironment,
    C2SEditWorld,
)
from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import DialogNode, DialogNodeState, DialogTree
from lucidium.domain.environment import Environment
from lucidium.domain.game import Game
from lucidium.domain.settings import Settings
from lucidium.domain.world import WorldState


class _StubSession:
    def __init__(self, *, game: Game) -> None:
        self.game = game
        self.commits = 0
        self.settings = Settings()

    def install_game(self, g: Game) -> None:
        self.game = g

    async def commit(self) -> None:
        self.commit_blocking()

    def commit_blocking(self) -> None:
        self.commits += 1

    def invalidate_speculation_from(self, root_id: str) -> list[str]:
        return []


def _npc() -> Character:
    return Character(
        name="Hale",
        description="the keeper",
        gender="male",
        age=50,
        ethnicity="local",
        skin="tan",
        hair_color="grey",
        hairstyle="short",
        eye_color="brown",
        build="stocky",
        bust="n/a",
        outfit="oilskin",
        pose="leaning",
        expression="watchful",
        seed=7,
        kind=CharacterKind.human,
    )


def _env() -> Environment:
    return Environment(
        location_label="lighthouse",
        prompt="a lamp room at dusk",
        prompt_hash="h" * 64,
        image_path="/real/render.png",
        seed=11,
    )


def _make() -> tuple[_StubSession, Character, Environment]:
    npc, env = _npc(), _env()
    node = DialogNode(
        id="n1",
        text="x",
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )
    game = Game(
        world=WorldState(
            game_name="t",
            setting="harbor",
            genre="Mystery",
            visual_style="ink wash",
        ),
        characters={npc.id: npc},
        environments={env.id: env},
        dialog_tree=DialogTree(
            nodes={node.id: node},
            root_id=node.id,
            committed_path=[node.id],
        ),
        current_node_id=node.id,
    )
    return _StubSession(game=game), npc, env


async def _drain(gen):
    return [m async for m in gen]


@pytest.mark.asyncio
async def test_bad_environment_seed_is_a_schema_error_and_commits_nothing() -> None:
    """The exact frame that used to poison a save: a string seed."""
    session, _unused_npc, env = _make()
    before = session.game.model_dump_json()
    ctx = HandlerContext(session=session)

    with pytest.raises(SchemaError):
        await edit_environment_handler(
            C2SEditEnvironment(environment_id=env.id, field="seed", value="x"),
            ctx,
        )

    # Nothing installed, nothing written — the save on disk is what
    # ``session.game`` would serialise, and it is byte-identical.
    assert session.game.model_dump_json() == before
    assert session.commits == 0
    # And it still loads.
    Game.model_validate_json(session.game.model_dump_json())


@pytest.mark.asyncio
async def test_environment_image_path_is_not_editable() -> None:
    """Filesystem-path fields are off the allow-list entirely, even
    with a well-typed value."""
    session, _unused_npc, env = _make()
    ctx = HandlerContext(session=session)

    with pytest.raises(SchemaError):
        await edit_environment_handler(
            C2SEditEnvironment(
                environment_id=env.id,
                field="image_path",
                value="../../../../etc/passwd",
            ),
            ctx,
        )
    assert session.game.environments[env.id].image_path == "/real/render.png"
    assert session.commits == 0


@pytest.mark.asyncio
async def test_legitimate_environment_edit_round_trips() -> None:
    session, _unused_npc, env = _make()
    ctx = HandlerContext(session=session)

    await _drain(
        await edit_environment_handler(
            C2SEditEnvironment(
                environment_id=env.id,
                field="location_label",
                value="the quay",
            ),
            ctx,
        )
    )

    assert session.game.environments[env.id].location_label == "the quay"
    assert session.commits == 1
    reloaded = Game.model_validate_json(session.game.model_dump_json())
    assert reloaded.environments[env.id].location_label == "the quay"
    # A numeric seed edit is legitimate too (the Rerender docstring
    # points players at it to pin a draw they like).
    await _drain(
        await edit_environment_handler(
            C2SEditEnvironment(environment_id=env.id, field="seed", value=99),
            ctx,
        )
    )
    assert session.game.environments[env.id].seed == 99


@pytest.mark.asyncio
async def test_bad_character_age_is_a_schema_error() -> None:
    session, npc, _unused_env = _make()
    ctx = HandlerContext(session=session)

    with pytest.raises(SchemaError):
        await edit_character_handler(
            C2SEditCharacter(character_id=npc.id, field="age", value="ancient"),
            ctx,
        )
    assert session.game.characters[npc.id].age == 50
    assert session.commits == 0


@pytest.mark.asyncio
async def test_character_non_editable_field_rejected() -> None:
    """``images`` is engine-owned and carries file paths."""
    session, npc, _unused_env = _make()
    ctx = HandlerContext(session=session)

    with pytest.raises(SchemaError):
        await edit_character_handler(
            # ``value`` is a scalar (``EditValue``); the rejection under test is
            # of the field name, not the payload shape.
            C2SEditCharacter(character_id=npc.id, field="images", value=""),
            ctx,
        )
    assert session.commits == 0


@pytest.mark.asyncio
async def test_legitimate_character_edit_round_trips() -> None:
    session, npc, _unused_env = _make()
    ctx = HandlerContext(session=session)

    await _drain(
        await edit_character_handler(
            C2SEditCharacter(character_id=npc.id, field="outfit", value="storm cloak"),
            ctx,
        )
    )

    reloaded = Game.model_validate_json(session.game.model_dump_json())
    assert reloaded.characters[npc.id].outfit == "storm cloak"


@pytest.mark.asyncio
async def test_bad_world_clamp_is_a_schema_error_and_good_edit_round_trips() -> None:
    session, _unused_npc, _unused_env = _make()
    ctx = HandlerContext(session=session)

    with pytest.raises(SchemaError):
        await edit_world_handler(
            C2SEditWorld(field="prompt_history_clamp_chars", value="lots"),
            ctx,
        )
    with pytest.raises(SchemaError):
        await edit_world_handler(C2SEditWorld(field="music_path", value="/etc/passwd"), ctx)
    assert session.commits == 0

    await _drain(
        await edit_world_handler(
            C2SEditWorld(field="setting", value="a drowned harbour"),
            ctx,
        )
    )
    reloaded = Game.model_validate_json(session.game.model_dump_json())
    assert reloaded.world.setting == "a drowned harbour"
