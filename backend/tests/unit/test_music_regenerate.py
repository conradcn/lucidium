"""``c2s/music/regenerate`` re-fires the ACE-Step pipeline for the
live game's background track.

The Story panel's Music tab calls this when the player clicks
Regenerate. The actual render is dispatched as a background task
via ``_spawn_async_music``; this handler's job is to update the
stored ``world.music_prompt`` (when supplied), enforce that music
gen is enabled, and emit a state/patch for the prompt change so
the renderer's stored value stays in lockstep.

These tests stub out the spawn helper so no audio backend is
required; they only verify the handler's shape: prompt-update
patch, error on disabled music, error on missing prompt.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lucidium.api.errors import NotFoundError, SchemaError
from lucidium.api.handlers import (
    HandlerContext,
    music_regenerate_handler,
)
from lucidium.api.messages import (
    C2SMusicRegenerate,
    MessageType,
)
from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import (
    DialogNode,
    DialogNodeState,
    DialogTree,
)
from lucidium.domain.game import Game
from lucidium.domain.settings import MusicSettings, Settings
from lucidium.domain.world import WorldState


class _StubSession:
    def __init__(self, *, game: Game, music_enabled: bool) -> None:
        self.game = game
        self.commits = 0
        self.settings = Settings(
            music=MusicSettings(enabled=music_enabled),
        )

    def install_game(self, g: Game) -> None:
        self.game = g

    async def commit(self) -> None:
        self.commit_blocking()

    def commit_blocking(self) -> None:
        self.commits += 1


def _world(*, music_prompt: str = "") -> WorldState:
    return WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink wash",
        music_prompt=music_prompt,
    )


def _player() -> Character:
    return Character(
        name="Iris",
        description="archivist",
        gender="female",
        age=28,
        ethnicity="local",
        skin="pale",
        hair_color="auburn",
        hairstyle="braid",
        eye_color="grey",
        build="slight",
        bust="moderate",
        outfit="wool coat",
        pose="standing",
        expression="alert",
        seed=1,
        is_player=True,
        kind=CharacterKind.human,
    )


def _make_session(*, music_enabled: bool, music_prompt: str = "") -> _StubSession:
    node = DialogNode(
        id="n1",
        text="x",
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )
    tree = DialogTree(
        nodes={node.id: node},
        root_id=node.id,
        committed_path=[node.id],
    )
    player = _player()
    game = Game(
        world=_world(music_prompt=music_prompt),
        characters={player.id: player},
        dialog_tree=tree,
        current_node_id=node.id,
    )
    return _StubSession(game=game, music_enabled=music_enabled)


async def _drain(gen):
    return [m async for m in gen]


@pytest.mark.asyncio
async def test_regenerate_with_new_prompt_patches_world_and_spawns_task() -> None:
    session = _make_session(music_enabled=True, music_prompt="ambient drone")
    ctx = HandlerContext(session=session)

    with patch("lucidium.api.handlers._spawn_async_music") as spawn:
        result = await music_regenerate_handler(
            C2SMusicRegenerate(prompt="brooding piano, slow tempo"),
            ctx,
        )
        messages = await _drain(result)

    # World prompt updated.
    assert session.game.world.music_prompt == "brooding piano, slow tempo"
    assert session.commits == 1
    # State patch carries the prompt change so the renderer's
    # stored value stays in sync.
    assert len(messages) == 1
    msg_type, payload = messages[0]
    assert msg_type == MessageType.s2c_state_patch
    assert payload.ops[0].path == "/world/music_prompt"
    assert payload.ops[0].value == "brooding piano, slow tempo"
    # Background render task spawned with the new prompt.
    spawn.assert_called_once()
    args, _ = spawn.call_args
    assert args[1] == "brooding piano, slow tempo"


@pytest.mark.asyncio
async def test_regenerate_without_new_prompt_reuses_stored_prompt() -> None:
    """Empty prompt = retry the existing track. World stays
    unchanged; no state/patch is needed."""
    session = _make_session(music_enabled=True, music_prompt="ambient drone")
    ctx = HandlerContext(session=session)

    with patch("lucidium.api.handlers._spawn_async_music") as spawn:
        result = await music_regenerate_handler(
            C2SMusicRegenerate(prompt=""),
            ctx,
        )
        messages = await _drain(result)

    assert session.game.world.music_prompt == "ambient drone"
    assert session.commits == 0
    assert messages == []
    spawn.assert_called_once()
    args, _ = spawn.call_args
    assert args[1] == "ambient drone"


@pytest.mark.asyncio
async def test_regenerate_refused_when_music_disabled() -> None:
    session = _make_session(music_enabled=False, music_prompt="ambient drone")
    ctx = HandlerContext(session=session)
    with pytest.raises(SchemaError):
        await music_regenerate_handler(
            C2SMusicRegenerate(prompt=""),
            ctx,
        )


@pytest.mark.asyncio
async def test_regenerate_refused_when_no_game() -> None:
    session = _make_session(music_enabled=True)
    session.game = None
    ctx = HandlerContext(session=session)
    with pytest.raises(NotFoundError):
        await music_regenerate_handler(
            C2SMusicRegenerate(prompt="anything"),
            ctx,
        )
