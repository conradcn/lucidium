"""Deterministic backstop against the disappearing-character bug.

``handlers._recent_scene_change`` gates the summarizer's
``characters_to_offstage`` pass so a quiet-but-present character is only
offstaged when the scene has actually moved. Same-scene exits are handled
per-beat by the storyteller's ``leaving_character_ids``; the summarizer's
bulk pass is trusted only across an actual location change.
"""

from __future__ import annotations

from lucidium.api.handlers import _recent_scene_change
from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import DialogNode, DialogNodeState, DialogTree
from lucidium.domain.game import Game
from lucidium.domain.world import WorldState


def _world() -> WorldState:
    return WorldState(game_name="t", setting="harbor", genre="Mystery", visual_style="ink")


def _node(location_id: str | None) -> DialogNode:
    return DialogNode(
        parent_id=None,
        speaker_id=None,
        text="x",
        location_id=location_id,
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )


def _pc() -> Character:
    return Character(
        is_player=True,
        name="You",
        description="the player",
        gender="n/a",
        age=30,
        ethnicity="n/a",
        skin="n/a",
        hair_color="n/a",
        hairstyle="n/a",
        eye_color="n/a",
        build="n/a",
        bust="n/a",
        outfit="n/a",
        pose="n/a",
        expression="n/a",
        seed=1,
        kind=CharacterKind.human,
    )


def _game(location_ids: list[str | None]) -> Game:
    nodes = [_node(loc) for loc in location_ids]
    pc = _pc()
    tree = DialogTree(
        nodes={n.id: n for n in nodes},
        root_id=nodes[0].id,
        committed_path=[n.id for n in nodes],
    )
    return Game(
        world=_world(),
        characters={pc.id: pc},
        environments={},
        dialog_tree=tree,
        current_node_id=nodes[-1].id,
        on_stage=[],
    )


def test_no_scene_change_when_location_is_stable() -> None:
    game = _game(["room-a", "room-a", "room-a", "room-a"])
    assert _recent_scene_change(game) is False


def test_scene_change_detected_within_window() -> None:
    game = _game(["room-a", "room-a", "room-b"])
    assert _recent_scene_change(game) is True


def test_old_scene_change_outside_window_is_ignored() -> None:
    # The move happened 4 beats ago; the last `window`+1 beats are all
    # room-b, so no RECENT change -> summarizer offstaging stays gated.
    game = _game(["room-a", "room-b", "room-b", "room-b", "room-b"])
    assert _recent_scene_change(game, window=3) is False


def test_carried_forward_location_across_none_beats() -> None:
    # Beats that don't set a location inherit the previous one; a genuine
    # change still trips the guard even with None gaps in between.
    game = _game(["room-a", None, "room-b"])
    assert _recent_scene_change(game) is True


def test_single_node_is_not_a_scene_change() -> None:
    game = _game(["room-a"])
    assert _recent_scene_change(game) is False
