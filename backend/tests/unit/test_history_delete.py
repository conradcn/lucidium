"""``c2s/edit/history/delete`` — splice a single committed beat
out of the history.

Removes the node from ``dialog_tree.nodes`` and ``committed_path``,
re-parents the next committed beat to the deleted node's parent
so the chain stays linked, and invalidates any speculative
descendants left dangling.

Refusal paths (raise SchemaError):
  * deleting the root beat (the chain has to start somewhere).
  * deleting the current beat (the player is reading it; would
    leave ``current_node_id`` pointing at nothing).

These tests run against a stub session so no live LLM / WebSocket
is needed.
"""

from __future__ import annotations

import pytest

from lucidium.api.errors import NotFoundError, SchemaError
from lucidium.api.handlers import (
    HandlerContext,
    edit_history_delete_handler,
)
from lucidium.api.messages import (
    C2SEditHistoryDelete,
    MessageType,
)
from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import (
    DialogNode,
    DialogNodeState,
    DialogTree,
)
from lucidium.domain.game import Game
from lucidium.domain.settings import Settings
from lucidium.domain.world import WorldState


class _StubSession:
    def __init__(self, *, game: Game) -> None:
        self.game = game
        self.commits = 0
        self.invalidated_from: list[str] = []
        self.settings = Settings()

    def install_game(self, g: Game) -> None:
        self.game = g

    async def commit(self) -> None:
        self.commit_blocking()

    def commit_blocking(self) -> None:
        self.commits += 1

    def invalidate_speculation_from(self, root_id: str) -> list[str]:
        self.invalidated_from.append(root_id)
        return []


def _world() -> WorldState:
    return WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink wash",
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


def _node(*, node_id: str, parent_id: str | None, text: str = "x") -> DialogNode:
    return DialogNode(
        id=node_id,
        parent_id=parent_id,
        text=text,
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )


def _make_session(committed_count: int) -> tuple[_StubSession, list[str]]:
    """Build a session with N committed beats in a linear chain.
    Returns (session, list of node ids in chain order)."""
    ids = [f"n{i}" for i in range(committed_count)]
    nodes = {}
    for i, nid in enumerate(ids):
        parent = ids[i - 1] if i > 0 else None
        nodes[nid] = _node(node_id=nid, parent_id=parent)
    tree = DialogTree(
        nodes=nodes,
        root_id=ids[0],
        committed_path=list(ids),
    )
    player = _player()
    game = Game(
        world=_world(),
        characters={player.id: player},
        dialog_tree=tree,
        current_node_id=ids[-1],
    )
    return _StubSession(game=game), ids


async def _drain(gen):
    return [m async for m in gen]


@pytest.mark.asyncio
async def test_delete_middle_beat_splices_chain() -> None:
    session, _ids = _make_session(4)  # n0, n1, n2, n3 (current)
    ctx = HandlerContext(session=session)

    result = await edit_history_delete_handler(
        C2SEditHistoryDelete(node_id="n1"),
        ctx,
    )
    messages = await _drain(result)

    tree = session.game.dialog_tree
    assert "n1" not in tree.nodes
    assert tree.committed_path == ["n0", "n2", "n3"]
    # Successor's parent_id was rewritten to skip the deleted beat.
    assert tree.nodes["n2"].parent_id == "n0"
    assert session.commits == 1
    # Speculation past the new tail is invalidated.
    assert session.invalidated_from == ["n3"]
    # Frontend gets a full state push (linked structure changed too
    # broadly for a clean patch op set).
    assert len(messages) == 1
    assert messages[0][0] == MessageType.s2c_state_full


@pytest.mark.asyncio
async def test_delete_first_non_root_beat_keeps_root_as_parent_of_new_head() -> None:
    session, _ = _make_session(3)  # n0 (root), n1, n2 (current)
    ctx = HandlerContext(session=session)

    await _drain(
        await edit_history_delete_handler(
            C2SEditHistoryDelete(node_id="n1"),
            ctx,
        )
    )

    tree = session.game.dialog_tree
    assert tree.committed_path == ["n0", "n2"]
    assert tree.nodes["n2"].parent_id == "n0"


@pytest.mark.asyncio
async def test_delete_root_is_refused() -> None:
    session, _ = _make_session(3)
    ctx = HandlerContext(session=session)
    with pytest.raises(SchemaError):
        await edit_history_delete_handler(
            C2SEditHistoryDelete(node_id="n0"),
            ctx,
        )


@pytest.mark.asyncio
async def test_delete_current_beat_is_refused() -> None:
    session, _ = _make_session(3)  # current is n2
    ctx = HandlerContext(session=session)
    with pytest.raises(SchemaError):
        await edit_history_delete_handler(
            C2SEditHistoryDelete(node_id="n2"),
            ctx,
        )


@pytest.mark.asyncio
async def test_delete_unknown_node_raises_not_found() -> None:
    session, _ = _make_session(3)
    ctx = HandlerContext(session=session)
    with pytest.raises(NotFoundError):
        await edit_history_delete_handler(
            C2SEditHistoryDelete(node_id="ghost"),
            ctx,
        )
