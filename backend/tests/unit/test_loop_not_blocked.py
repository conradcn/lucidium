"""The measured blocking work stays off the event loop.

These pin behaviour, not wall-clock: each test puts a known-slow
synchronous call on the path and asserts the loop kept running other
coroutines while it was in flight. A regression that moves the work back
onto the loop thread starves the watcher and fails here, on any machine.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import DialogNode, DialogNodeState, DialogTree
from lucidium.domain.game import Game
from lucidium.domain.settings import Settings
from lucidium.domain.world import WorldState
from lucidium.orchestration.assets import _render_or_await
from lucidium.orchestration.session import Session

# Long enough that a loop-thread call would obviously starve the
# watcher, short enough not to drag the suite out.
_BLOCK_S = 0.25


class _Ticker:
    """Counts how many times it got to run while something else works."""

    def __init__(self) -> None:
        self.ticks = 0
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> _Ticker:
        self._task = asyncio.create_task(self._run())
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self._stop.set()
        assert self._task is not None
        await self._task

    async def _run(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(0.005)
            self.ticks += 1


async def test_render_post_processing_does_not_block_the_loop(tmp_path: Path) -> None:
    """``safety_check`` and ``post_process`` run on worker threads."""

    async def render() -> bytes:
        return b"payload"

    def slow_safety_check(_data: bytes) -> bool:
        time.sleep(_BLOCK_S)
        return True

    def slow_post_process(data: bytes) -> bytes:
        time.sleep(_BLOCK_S)
        return data

    async with _Ticker() as ticker:
        wrote = await _render_or_await(
            tmp_path / "portrait.png",
            render=render,
            post_process=slow_post_process,
            safety_check=slow_safety_check,
        )

    assert wrote
    assert (tmp_path / "portrait.png").read_bytes() == b"payload"
    # Two 250 ms blocks at a 5 ms tick is ~100 ticks if the loop stayed
    # free; a loop-thread call would leave it in the low single digits.
    assert ticker.ticks > 20, f"loop was starved during post-processing (only {ticker.ticks} ticks)"


def _game() -> Game:
    npc = Character(
        name="Hale",
        description="the harbormaster",
        seed=7,
        kind=CharacterKind.human,
    )
    node = DialogNode(
        parent_id=None,
        speaker_id=None,
        text="The harbor wakes.",
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )
    return Game(
        world=WorldState(
            game_name="Embers",
            setting="harbor",
            genre="mystery",
            visual_style="ink",
        ),
        characters={npc.id: npc},
        dialog_tree=DialogTree(
            nodes={node.id: node},
            root_id=node.id,
            committed_path=[node.id],
        ),
        current_node_id=node.id,
    )


async def test_commit_does_not_block_the_loop(tmp_path: Path) -> None:
    session = Session(settings=Settings(), saves_root=tmp_path)
    session.game = _game()

    async with _Ticker() as ticker:
        await asyncio.gather(*(session.commit() for _ in range(6)))

    assert ticker.ticks > 3, f"loop was starved during commit (only {ticker.ticks} ticks)"
    # Every commit landed and the save is readable — the serialising
    # lock means concurrent commits don't corrupt each other or lose the
    # ``os.replace`` race that Windows fails with WinError 5.
    assert (tmp_path / session.game.id / "game.json").exists()
    assert (tmp_path / session.game.id / "meta.json").exists()


async def test_commit_is_a_noop_without_a_game(tmp_path: Path) -> None:
    session = Session(settings=Settings(), saves_root=tmp_path)
    await session.commit()
    assert not list(tmp_path.iterdir())  # noqa: ASYNC240 - sync fs in test assertion


def test_commit_blocking_still_works_off_loop(tmp_path: Path) -> None:
    """The sync escape hatch the minor-nudity guard uses."""
    session = Session(settings=Settings(), saves_root=tmp_path)
    session.game = _game()
    session.commit_blocking()
    assert (tmp_path / session.game.id / "game.json").exists()


class _StubSession:
    def __init__(self, game: Game) -> None:
        self.game = game
        self.settings = Settings()
        self.undo_stack: list[Game] = []

    def install_game(self, g: Game) -> None:
        self.game = g

    async def commit(self) -> None:
        pass

    def commit_blocking(self) -> None:
        pass


@pytest.mark.parametrize(
    ("handler_name", "message_name", "expect_removed", "expect_on_stage"),
    [
        ("edit_character_dismiss_handler", "C2SEditCharacterDismiss", True, False),
        ("edit_character_show_handler", "C2SEditCharacterShow", False, True),
    ],
)
async def test_cast_tab_toggle_emits_targeted_ops_not_a_full_state(
    handler_name: str,
    message_name: str,
    expect_removed: bool,
    expect_on_stage: bool,
) -> None:
    """Dismiss / show ship three ops. They used to ship the whole game,
    which is ~1 MB on a thousand-node save for a one-character change."""
    from lucidium.api import handlers, messages
    from lucidium.api.messages import MessageType

    game = _game()
    npc_id = next(iter(game.characters))
    if expect_on_stage:
        # ``show`` starts from a dismissed character.
        char = game.characters[npc_id].model_copy(
            update={"removed": True, "removed_reason": "dismissed"}
        )
        game = game.model_copy(update={"characters": {npc_id: char}})
    else:
        game = game.model_copy(update={"on_stage": [npc_id]})

    session = _StubSession(game)
    ctx = handlers.HandlerContext(session=session)
    payload = getattr(messages, message_name)(character_id=npc_id)

    emitted = [message async for message in await getattr(handlers, handler_name)(payload, ctx)]

    assert len(emitted) == 1
    message_type, patch = emitted[0]
    assert message_type is MessageType.s2c_state_patch
    by_path = {op.path: op.value for op in patch.ops}
    assert by_path[f"/characters/{npc_id}/removed"] is expect_removed
    assert by_path["/on_stage"] == ([npc_id] if expect_on_stage else [])
    # And the server-side state agrees with what we told the renderer.
    assert session.game.characters[npc_id].removed is expect_removed
    assert session.game.on_stage == by_path["/on_stage"]
