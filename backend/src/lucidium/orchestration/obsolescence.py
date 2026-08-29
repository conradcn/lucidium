"""Task obsolescence rules.

A scheduled work item carries the *premise* it was generated under. When
state mutates (player advances, types free text, edits a character or
world field), some queued tasks become obsolete. This module is the
single place where that decision lives.

Rules (FR-013, FR-016, FR-017):
  * A queued task whose target's premise no longer matches is dropped
    before dispatch.
  * An in-flight task is allowed to finish; its result is applied only
    if it is still more current than what it would replace, otherwise
    it is discarded.
  * Image work tied to a character or environment that is *unchanged*
    survives a free-text invalidation; image work tied to invalidated
    nodes does not.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..domain.dialog import DialogNode, DialogNodeState
from ..domain.game import Game


def descendants(game: Game, root_id: str) -> list[str]:
    """All node ids reachable from ``root_id`` (excluding the root itself)."""
    children: dict[str, list[str]] = {}
    for node in game.dialog_tree.nodes.values():
        if node.parent_id is not None:
            children.setdefault(node.parent_id, []).append(node.id)

    result: list[str] = []
    stack = list(children.get(root_id, []))
    while stack:
        nid = stack.pop()
        result.append(nid)
        stack.extend(children.get(nid, []))
    return result


def speculative_descendants(game: Game, root_id: str) -> list[str]:
    return [
        nid
        for nid in descendants(game, root_id)
        if game.dialog_tree.nodes[nid].state
        in (DialogNodeState.speculative, DialogNodeState.pending_text, DialogNodeState.ready)
    ]


def invalidate_descendants(game: Game, root_id: str) -> list[str]:
    """Mark all non-committed descendants as ``invalidated``. Returns the ids."""
    invalidated: list[str] = []
    for nid in speculative_descendants(game, root_id):
        node = game.dialog_tree.nodes[nid]
        if node.state == DialogNodeState.committed:
            continue
        new_node = node.model_copy(update={"state": DialogNodeState.invalidated})
        game.dialog_tree.nodes[nid] = new_node
        invalidated.append(nid)
    return invalidated


def invalidate_unwalked_descendants(game: Game, root_id: str) -> list[str]:
    """Mark every descendant of ``root_id`` that the player hasn't
    walked yet as ``invalidated``. Used by ``c2s/play/free_text`` —
    when the player takes the story in a new direction, the
    LLM-prepared beats below them become obsolete (they were authored
    under a different premise).

    Returns the ids that were just invalidated.
    """
    walked = set(game.dialog_tree.committed_path)
    invalidated: list[str] = []
    for nid in descendants(game, root_id):
        if nid in walked:
            continue
        node = game.dialog_tree.nodes[nid]
        if node.state == DialogNodeState.invalidated:
            continue
        new_node = node.model_copy(update={"state": DialogNodeState.invalidated})
        game.dialog_tree.nodes[nid] = new_node
        invalidated.append(nid)
    return invalidated


def first_valid_child(
    game: Game, parent_id: str, *, chosen_option_id: str | None = None
) -> str | None:
    """Return the id of the parent's still-valid child node that
    matches ``chosen_option_id`` (or has no chosen option when
    ``chosen_option_id`` is None). Used to decide whether ``advance``
    can walk an existing pre-generated beat or has to call the LLM
    to generate the next chain.
    """
    for nid, node in game.dialog_tree.nodes.items():
        if node.parent_id != parent_id:
            continue
        if node.state == DialogNodeState.invalidated:
            continue
        if node.chosen_option_id != chosen_option_id:
            continue
        return nid
    return None


def is_node_task_obsolete(*, task_target_node_id: str, game: Game) -> bool:
    """Is the speculative text task for ``task_target_node_id`` still relevant?"""
    node = game.dialog_tree.nodes.get(task_target_node_id)
    if node is None:
        return True
    return node.state == DialogNodeState.invalidated


def filter_image_tasks_after_invalidation(
    *,
    invalidated_node_ids: Iterable[str],
    image_task_node_ids: Iterable[str],
) -> list[str]:
    """Return image-task node ids that should be cancelled.

    Per FR-017, image work that targets a *node* (not a character or
    environment whose state is unchanged) is cancelled when the node is
    invalidated. Character-portrait and environment-background work
    keyed by entity id rather than node id survives.
    """
    invalid = set(invalidated_node_ids)
    return [nid for nid in image_task_node_ids if nid in invalid]


def committed_path(game: Game) -> list[DialogNode]:
    return [game.dialog_tree.nodes[nid] for nid in game.dialog_tree.committed_path]
