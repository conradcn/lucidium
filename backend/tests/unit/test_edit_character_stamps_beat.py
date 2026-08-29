"""``c2s/edit/character`` stamps the current beat's
``character_changes`` so the edit survives replay/retcon.

Bug shape: player edits a NPC's outfit via the Cast tab. The edit
flips ``game.characters[id].outfit`` (good) and emits a state/patch
(good). But the CURRENT beat's ``character_changes`` list still
reflects the LLM's original choice. Any later pass that re-derives
character state from the dialog tree (retcon, undo+redo, future
replay) overwrites the edit with the stale beat-stamped value.

Fix: when the field maps onto ``CharacterAttributeField``, append (or
replace) a CharacterChange on the current node so the edit becomes
part of the canonical beat record. Fields that aren't in the enum
(``seed``, ``images``) are left alone — they don't participate in
the dialog state machine and don't need persisting via beats.
"""

from __future__ import annotations

import pytest

from lucidium.api.handlers import (
    HandlerContext,
    edit_character_handler,
)
from lucidium.api.messages import (
    C2SEditCharacter,
    MessageType,
)
from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import (
    CharacterAttributeField,
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


def _npc(*, name: str = "Hale", outfit: str = "oilskin") -> Character:
    return Character(
        name=name,
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
        outfit=outfit,
        pose="leaning",
        expression="watchful",
        seed=7,
        kind=CharacterKind.human,
    )


def _make_session(npc: Character) -> tuple[_StubSession, str]:
    node = DialogNode(
        id="n1",
        text="x",
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )
    tree = DialogTree(nodes={node.id: node}, root_id=node.id, committed_path=[node.id])
    player = _player()
    game = Game(
        world=_world(),
        characters={player.id: player, npc.id: npc},
        dialog_tree=tree,
        current_node_id=node.id,
    )
    return _StubSession(game=game), node.id


async def _drain(gen):
    return [m async for m in gen]


@pytest.mark.asyncio
async def test_outfit_edit_stamps_character_changes_on_current_beat() -> None:
    npc = _npc(outfit="oilskin")
    session, node_id = _make_session(npc)
    ctx = HandlerContext(session=session)

    result = await edit_character_handler(
        C2SEditCharacter(character_id=npc.id, field="outfit", value="storm cloak"),
        ctx,
    )
    messages = await _drain(result)

    # Character field flipped on the live game.
    assert session.game.characters[npc.id].outfit == "storm cloak"
    # Current beat now has a CharacterChange for the outfit edit.
    changes = session.game.dialog_tree.nodes[node_id].character_changes
    matching = [
        c for c in changes if c.character_id == npc.id and c.field == CharacterAttributeField.outfit
    ]
    assert len(matching) == 1
    assert matching[0].new_value == "storm cloak"
    # State patch carries both ops so the renderer applies them in
    # the same tick (no transient frame where the field flipped but
    # the beat hasn't).
    assert len(messages) == 1
    msg_type, payload = messages[0]
    assert msg_type == MessageType.s2c_state_patch
    paths = [op.path for op in payload.ops]
    assert f"/characters/{npc.id}/outfit" in paths
    assert f"/dialog_tree/nodes/{node_id}/character_changes" in paths


@pytest.mark.asyncio
async def test_repeat_outfit_edit_replaces_prior_change_does_not_stack() -> None:
    """Editing the same field twice should keep ONE entry on the
    beat — otherwise character_changes would grow unbounded across
    a long session of touch-ups."""
    npc = _npc(outfit="oilskin")
    session, node_id = _make_session(npc)
    ctx = HandlerContext(session=session)

    await _drain(
        await edit_character_handler(
            C2SEditCharacter(character_id=npc.id, field="outfit", value="storm cloak"),
            ctx,
        )
    )
    await _drain(
        await edit_character_handler(
            C2SEditCharacter(character_id=npc.id, field="outfit", value="grey wool"),
            ctx,
        )
    )

    changes = session.game.dialog_tree.nodes[node_id].character_changes
    outfit_changes = [
        c for c in changes if c.character_id == npc.id and c.field == CharacterAttributeField.outfit
    ]
    assert len(outfit_changes) == 1
    assert outfit_changes[0].new_value == "grey wool"


@pytest.mark.asyncio
async def test_edit_to_non_attribute_field_skips_beat_stamp() -> None:
    """Fields outside the dialog attribute enum (``seed``,
    ``images``) don't ride the character_changes path — there's
    nothing to persist through beat replay for those fields."""
    npc = _npc()
    session, node_id = _make_session(npc)
    ctx = HandlerContext(session=session)

    await _drain(
        await edit_character_handler(
            C2SEditCharacter(character_id=npc.id, field="seed", value=42),
            ctx,
        )
    )

    assert session.game.characters[npc.id].seed == 42
    # No character_changes entry — the field doesn't participate
    # in the beat-level state machine.
    assert session.game.dialog_tree.nodes[node_id].character_changes == []
