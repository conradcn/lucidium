from __future__ import annotations

from lucidium.domain.dialog import (
    DialogNode,
    DialogNodeState,
    DialogTree,
    premise_hash,
    world_snapshot_vector,
)
from lucidium.domain.game import Game
from lucidium.domain.world import WorldState
from lucidium.orchestration import obsolescence


def _ph() -> str:
    return premise_hash(
        parent_id=None,
        chosen_option_id=None,
        snapshot_vector=world_snapshot_vector(
            on_stage_character_ids=[], character_attributes={}, world_fields={}
        ),
    )


def _make_game_with_speculative_chain(length: int = 3) -> Game:
    nodes: dict[str, DialogNode] = {}
    chain_ids: list[str] = []
    parent_id: str | None = None
    root_id: str | None = None
    for index in range(length):
        node = DialogNode(
            parent_id=parent_id,
            premise_hash=_ph(),
            state=DialogNodeState.committed if index == 0 else DialogNodeState.speculative,
        )
        nodes[node.id] = node
        chain_ids.append(node.id)
        if root_id is None:
            root_id = node.id
        parent_id = node.id
    tree = DialogTree(nodes=nodes, root_id=root_id, committed_path=[chain_ids[0]])
    return Game(
        world=WorldState(game_name="t", setting="s", genre="g", visual_style="v"),
        dialog_tree=tree,
        current_node_id=chain_ids[0],
    )


def test_invalidate_descendants_marks_speculative_children() -> None:
    game = _make_game_with_speculative_chain(4)
    assert game.current_node_id is not None
    invalidated = obsolescence.invalidate_descendants(game, game.current_node_id)
    # 3 speculative descendants should be invalidated.
    assert len(invalidated) == 3
    for nid in invalidated:
        assert game.dialog_tree.nodes[nid].state == DialogNodeState.invalidated


def test_descendants_finds_only_children_of_root() -> None:
    game = _make_game_with_speculative_chain(3)
    assert game.current_node_id is not None
    children = obsolescence.descendants(game, game.current_node_id)
    assert len(children) == 2  # 2 speculative children below the committed root


def test_filter_image_tasks_after_invalidation_keeps_unaffected() -> None:
    invalid = ["nodeA", "nodeB"]
    image_targets = ["nodeA", "characterX", "envY", "nodeB"]
    cancel = obsolescence.filter_image_tasks_after_invalidation(
        invalidated_node_ids=invalid, image_task_node_ids=image_targets
    )
    assert set(cancel) == {"nodeA", "nodeB"}
