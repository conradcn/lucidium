from __future__ import annotations

import pytest

from lucidium.domain.character import Character
from lucidium.domain.dialog import DialogNode, DialogTree, premise_hash, world_snapshot_vector
from lucidium.domain.game import Game
from lucidium.domain.world import WorldState


def _make_character(*, is_player: bool = False, name: str = "Iris") -> Character:
    return Character(
        is_player=is_player,
        name=name,
        description="curious archivist",
        gender="female",
        age=28,
        ethnicity="local",
        skin="pale",
        hair_color="auburn",
        hairstyle="braid",
        eye_color="grey",
        build="slight",
        bust="moderate",
        outfit="travel cloak",
        pose="standing",
        expression="alert",
        seed=42,
    )


def _make_world() -> WorldState:
    return WorldState(game_name="Embers", setting="harbor", genre="mystery", visual_style="ink")


def _make_node(parent_id: str | None = None) -> DialogNode:
    snapshot = world_snapshot_vector(
        on_stage_character_ids=[],
        character_attributes={},
        world_fields={"plot": "opening"},
    )
    return DialogNode(
        parent_id=parent_id,
        premise_hash=premise_hash(
            parent_id=parent_id, chosen_option_id=None, snapshot_vector=snapshot
        ),
    )


def test_game_accepts_at_most_one_player() -> None:
    a = _make_character(is_player=True, name="Iris")
    b = _make_character(is_player=True, name="Hale")
    with pytest.raises(ValueError, match="player character"):
        Game(world=_make_world(), characters={a.id: a, b.id: b})


def test_game_rejects_unknown_current_node_id() -> None:
    with pytest.raises(ValueError, match="current_node_id"):
        Game(world=_make_world(), current_node_id="missing")


def test_game_rejects_on_stage_referencing_unknown_character() -> None:
    with pytest.raises(ValueError, match="on_stage references unknown"):
        Game(world=_make_world(), on_stage=["ghost"])


def test_game_accepts_consistent_committed_path() -> None:
    node = _make_node()
    tree = DialogTree(nodes={node.id: node}, root_id=node.id, committed_path=[node.id])
    game = Game(world=_make_world(), dialog_tree=tree, current_node_id=node.id)
    assert game.current_node_id == node.id


def test_game_rejects_committed_path_tail_mismatch() -> None:
    node = _make_node()
    other = _make_node(parent_id=node.id)
    tree = DialogTree(
        nodes={node.id: node, other.id: other},
        root_id=node.id,
        committed_path=[node.id],
    )
    with pytest.raises(ValueError, match="committed_path tail"):
        Game(world=_make_world(), dialog_tree=tree, current_node_id=other.id)
