"""``c2s/edit/environment/apply`` — swap the current scene's
backdrop to a chosen environment.

The handler mutates the current dialog node's ``location_id`` to
the supplied environment id. Renderer's location-walk
(``activeEnvironmentId`` in EnvironmentsTab) then resolves to
this env's pre-rendered image and the on-screen backdrop swaps.

Tested by driving the handler against a minimal session stub
so the test suite runs without a live WebSocket / image client.
"""

from __future__ import annotations

import pytest

from lucidium.api.errors import NotFoundError
from lucidium.api.handlers import (
    HandlerContext,
    edit_environment_apply_handler,
)
from lucidium.api.messages import (
    C2SEditEnvironmentApply,
    MessageType,
)
from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import (
    DialogNode,
    DialogNodeState,
    DialogTree,
)
from lucidium.domain.environment import Environment
from lucidium.domain.game import Game
from lucidium.domain.world import WorldState


class _StubSession:
    def __init__(self, *, game: Game) -> None:
        self.game = game
        self.commits = 0
        self.emit = None

    def install_game(self, g: Game) -> None:
        self.game = g

    async def commit(self) -> None:
        self.commit_blocking()

    def commit_blocking(self) -> None:
        self.commits += 1


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


def _make_node(*, node_id: str, location_id: str | None) -> DialogNode:
    return DialogNode(
        id=node_id,
        text="x",
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
        location_id=location_id,
    )


def _make_session(
    *,
    environments: list[Environment],
    current_location: str | None,
) -> tuple[_StubSession, str]:
    """Build a session with a single committed dialog node whose
    ``location_id`` points at ``current_location``. Returns the
    session and the id of the current node so tests can assert
    the mutation lands on it."""
    node = _make_node(node_id="n1", location_id=current_location)
    tree = DialogTree(
        nodes={node.id: node},
        root_id=node.id,
        committed_path=[node.id],
    )
    player = _player()
    game = Game(
        world=_world(),
        characters={player.id: player},
        environments={env.id: env for env in environments},
        dialog_tree=tree,
        current_node_id=node.id,
    )
    return _StubSession(game=game), node.id


async def _drain(gen) -> list:
    """Collect all messages from an async generator."""
    out = []
    async for msg in gen:
        out.append(msg)
    return out


def _env(id: str, label: str = "", image_path: str | None = "/img.png") -> Environment:
    return Environment(
        id=id,
        location_label=label or id,
        prompt=f"prompt for {id}",
        image_path=image_path,
        seed=1,
        prompt_hash="h" * 16,
    )


# ---------- Happy path -------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_mutates_current_node_location_id() -> None:
    """The core happy path: current node sits in env A, player
    clicks Apply on env B → node's location_id flips to B."""
    session, node_id = _make_session(
        environments=[_env("env-a"), _env("env-b")],
        current_location="env-a",
    )
    ctx = HandlerContext(session=session)

    result = await edit_environment_apply_handler(
        C2SEditEnvironmentApply(environment_id="env-b"),
        ctx,
    )
    messages = await _drain(result)

    assert session.game.dialog_tree.nodes[node_id].location_id == "env-b"
    # Persisted to disk so the swap survives a save reload.
    assert session.commits == 1
    # Echoed to the renderer via a state/patch op so the
    # location-walk picks up the change.
    assert len(messages) == 1
    msg_type, payload = messages[0]
    assert msg_type == MessageType.s2c_state_patch
    ops = payload.ops
    assert len(ops) == 1
    assert ops[0].path == f"/dialog_tree/nodes/{node_id}/location_id"
    assert ops[0].value == "env-b"


@pytest.mark.asyncio
async def test_apply_doesnt_modify_other_nodes_in_history() -> None:
    """Past beats keep their original backdrops. Only the CURRENT
    node's location_id flips — apply is a one-shot for the
    current scene, not a global override."""
    past = _make_node(node_id="n0", location_id="env-a")
    current = _make_node(node_id="n1", location_id="env-a")
    tree = DialogTree(
        nodes={past.id: past, current.id: current},
        root_id=past.id,
        committed_path=[past.id, current.id],
    )
    player = _player()
    game = Game(
        world=_world(),
        characters={player.id: player},
        environments={
            "env-a": _env("env-a"),
            "env-b": _env("env-b"),
        },
        dialog_tree=tree,
        current_node_id=current.id,
    )
    session = _StubSession(game=game)
    ctx = HandlerContext(session=session)

    result = await edit_environment_apply_handler(
        C2SEditEnvironmentApply(environment_id="env-b"),
        ctx,
    )
    await _drain(result)

    # Past beat is untouched.
    assert session.game.dialog_tree.nodes["n0"].location_id == "env-a"
    # Current beat flipped.
    assert session.game.dialog_tree.nodes["n1"].location_id == "env-b"


# ---------- Edge cases -------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_already_active_env_is_a_noop_but_echoes_state() -> None:
    """Player clicked Apply on the env that's ALREADY the
    current backdrop. We don't error — we just emit a state
    patch echo for renderer consistency and skip the install /
    commit so we don't touch the save."""
    session, node_id = _make_session(
        environments=[_env("env-a")],
        current_location="env-a",
    )
    ctx = HandlerContext(session=session)

    result = await edit_environment_apply_handler(
        C2SEditEnvironmentApply(environment_id="env-a"),
        ctx,
    )
    messages = await _drain(result)

    # No new commit — the save is unchanged.
    assert session.commits == 0
    # location_id still points at env-a.
    assert session.game.dialog_tree.nodes[node_id].location_id == "env-a"
    # Renderer still gets the echo so its mirror stays consistent.
    assert len(messages) == 1
    assert messages[0][0] == MessageType.s2c_state_patch


@pytest.mark.asyncio
async def test_apply_unknown_env_raises_not_found() -> None:
    session, _ = _make_session(
        environments=[_env("env-a")],
        current_location="env-a",
    )
    ctx = HandlerContext(session=session)

    with pytest.raises(NotFoundError):
        await edit_environment_apply_handler(
            C2SEditEnvironmentApply(environment_id="env-ghost"),
            ctx,
        )
    # No mutation, no commit on the error path.
    assert session.commits == 0


@pytest.mark.asyncio
async def test_apply_with_no_game_raises_not_found() -> None:
    session = _StubSession(game=None)  # type: ignore[arg-type]
    ctx = HandlerContext(session=session)

    with pytest.raises(NotFoundError):
        await edit_environment_apply_handler(
            C2SEditEnvironmentApply(environment_id="env-a"),
            ctx,
        )


@pytest.mark.asyncio
async def test_apply_with_no_current_node_raises_not_found() -> None:
    """A game with environments but no current_node_id (e.g.
    mid-onboarding) can't have a backdrop applied — there's
    nothing whose location_id we'd flip."""
    player = _player()
    game = Game(
        world=_world(),
        characters={player.id: player},
        environments={"env-a": _env("env-a")},
        dialog_tree=DialogTree(),
        current_node_id=None,
    )
    session = _StubSession(game=game)
    ctx = HandlerContext(session=session)

    with pytest.raises(NotFoundError):
        await edit_environment_apply_handler(
            C2SEditEnvironmentApply(environment_id="env-a"),
            ctx,
        )


@pytest.mark.asyncio
async def test_apply_works_when_current_node_had_no_location() -> None:
    """A node with location_id=None (e.g. a narrator-interlude
    beat) can still receive an Apply — the backdrop swap fills
    in the previously-unset location."""
    session, node_id = _make_session(
        environments=[_env("env-a"), _env("env-b")],
        current_location=None,
    )
    ctx = HandlerContext(session=session)

    result = await edit_environment_apply_handler(
        C2SEditEnvironmentApply(environment_id="env-b"),
        ctx,
    )
    await _drain(result)

    assert session.game.dialog_tree.nodes[node_id].location_id == "env-b"
    assert session.commits == 1
