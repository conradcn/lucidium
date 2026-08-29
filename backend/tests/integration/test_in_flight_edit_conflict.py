"""When a player edits world or character state, in-flight speculative
text tasks become obsolete by virtue of the rescore() pass; their results
are discarded on return.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from lucidium.api.handlers import HandlerContext, build_default_registry
from lucidium.api.messages import (
    Envelope,
    MessageType,
)
from lucidium.domain.character import Character
from lucidium.domain.dialog import (
    DialogNode,
    DialogNodeState,
    DialogTree,
    premise_hash,
    world_snapshot_vector,
)
from lucidium.domain.game import Game
from lucidium.domain.world import WorldState
from lucidium.orchestration.session import Session


def _make_character() -> Character:
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
        outfit="cloak",
        pose="standing",
        expression="alert",
        seed=1,
    )


def _make_game(char: Character) -> Game:
    snapshot = world_snapshot_vector(
        on_stage_character_ids=[char.id],
        character_attributes={char.id: {"outfit": char.outfit}},
        world_fields={},
    )
    node = DialogNode(
        premise_hash=premise_hash(parent_id=None, chosen_option_id=None, snapshot_vector=snapshot),
        state=DialogNodeState.committed,
    )
    speculative = DialogNode(
        parent_id=node.id,
        premise_hash="some-old-hash",
        state=DialogNodeState.speculative,
    )
    return Game(
        world=WorldState(game_name="t", setting="s", genre="g", visual_style="v"),
        characters={char.id: char},
        dialog_tree=DialogTree(
            nodes={node.id: node, speculative.id: speculative},
            root_id=node.id,
            committed_path=[node.id],
        ),
        current_node_id=node.id,
        on_stage=[char.id],
    )


class _NoopLlm:
    async def complete(self, *_a, **_kw) -> AsyncIterator[str]:
        async def gen() -> AsyncIterator[str]:
            yield "{}"

        return gen()


@pytest.mark.asyncio
async def test_character_edit_invalidates_speculative_descendants(
    tmp_app_data: Path,
) -> None:
    char = _make_character()
    game = _make_game(char)
    session = Session(llm_client=_NoopLlm(), image_client=None)
    session.install_game(game)
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    speculative_id = next(
        nid
        for nid, node in game.dialog_tree.nodes.items()
        if node.state == DialogNodeState.speculative
    )

    out = []
    async for message in registry.dispatch(
        Envelope(
            type=MessageType.c2s_edit_character,
            payload={"character_id": char.id, "field": "outfit", "value": "torn cloak"},
        ),
        ctx,
    ):
        out.append(message)

    # The edit handler emits one s2c/state/patch describing the field change.
    assert any(t == MessageType.s2c_state_patch for t, _ in out)
    # The speculative descendant must now be invalidated.
    assert session.game.dialog_tree.nodes[speculative_id].state == DialogNodeState.invalidated
    # The actual character field was updated.
    assert session.game.characters[char.id].outfit == "torn cloak"
