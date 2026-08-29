"""``_apply_node`` keeps on-stage characters on stage unless the
beat itself flags them as leaving.

The bug: when streaming a multi-beat chain, characters who weren't
the speaker on a given beat would sometimes vanish — the LLM had
listed them in ``leaving_character_ids`` even though narration
didn't move them anywhere. The renderer then filtered them out of
the next prompt's ON-STAGE list and the storyteller couldn't bring
them back without a fresh ``new_characters`` entry.

These tests pin the persistence semantics that ``_apply_node``
guarantees, so a future refactor can't accidentally remove an
on-stage character whose only "crime" was not speaking on the
current beat. The matching prompt-level guidance lives in
``ON-STAGE PERSISTENCE`` in ``prompts/text_gen.py``.
"""

from __future__ import annotations

from lucidium.api.handlers import _apply_node
from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import (
    DialogNode,
    DialogNodeState,
    DialogTree,
)
from lucidium.domain.game import Game
from lucidium.domain.world import WorldState


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


def _npc(name: str) -> Character:
    return Character(
        name=name,
        description=f"the {name}",
        gender="male",
        age=40,
        ethnicity="local",
        skin="tan",
        hair_color="grey",
        hairstyle="short",
        eye_color="brown",
        build="stocky",
        bust="n/a",
        outfit="oilskin",
        pose="leaning",
        expression="watchful",
        seed=7,
        kind=CharacterKind.human,
    )


def _node(
    *,
    parent_id: str | None,
    speaker_id: str | None,
    entering: list[str] | None = None,
    leaving: list[str] | None = None,
) -> DialogNode:
    return DialogNode(
        parent_id=parent_id,
        speaker_id=speaker_id,
        text="x",
        entering_character_ids=entering or [],
        leaving_character_ids=leaving or [],
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )


def _seed_game(npcs: list[Character], on_stage: list[str]) -> Game:
    player = _player()
    parent = _node(parent_id=None, speaker_id=None)
    chars = {player.id: player}
    for n in npcs:
        chars[n.id] = n
    return Game(
        world=_world(),
        characters=chars,
        dialog_tree=DialogTree(
            nodes={parent.id: parent},
            root_id=parent.id,
            committed_path=[parent.id],
        ),
        current_node_id=parent.id,
        on_stage=list(on_stage),
    )


def test_on_stage_character_persists_when_not_speaker_on_next_beat() -> None:
    """A on-stage NPC who isn't the speaker on the next beat stays
    on stage. Without this guarantee, multi-character scenes lose
    silent participants between beats."""
    hale = _npc("Hale")
    mira = _npc("Mira")
    game = _seed_game([hale, mira], on_stage=[hale.id, mira.id])
    parent = game.dialog_tree.nodes[game.current_node_id]
    # Mira speaks; Hale is silent. Beat carries no leaving list.
    next_beat = _node(parent_id=parent.id, speaker_id=mira.id)
    new_game = _apply_node(game, next_beat)
    assert hale.id in new_game.on_stage
    assert mira.id in new_game.on_stage


def test_explicit_leaving_id_drops_character_from_stage() -> None:
    """When the LLM does narrate a character leaving, the on-stage
    list reflects it — the persistence rule is "default keep,"
    not "always keep"."""
    hale = _npc("Hale")
    mira = _npc("Mira")
    game = _seed_game([hale, mira], on_stage=[hale.id, mira.id])
    parent = game.dialog_tree.nodes[game.current_node_id]
    next_beat = _node(
        parent_id=parent.id,
        speaker_id=mira.id,
        leaving=[hale.id],
    )
    new_game = _apply_node(game, next_beat)
    assert hale.id not in new_game.on_stage
    assert mira.id in new_game.on_stage


def test_speaker_implicitly_enters_when_not_listed() -> None:
    """A character speaking on a beat without being on stage gets
    auto-promoted — protects against an LLM that emits the speaker
    id but forgets to populate entering_character_ids."""
    hale = _npc("Hale")
    game = _seed_game([hale], on_stage=[])
    parent = game.dialog_tree.nodes[game.current_node_id]
    next_beat = _node(parent_id=parent.id, speaker_id=hale.id)
    new_game = _apply_node(game, next_beat)
    assert hale.id in new_game.on_stage


def test_listing_on_stage_speaker_in_entering_is_idempotent() -> None:
    """Re-adding a character who's already on stage doesn't dupe
    them — the on_stage list stays a set-like roster."""
    hale = _npc("Hale")
    game = _seed_game([hale], on_stage=[hale.id])
    parent = game.dialog_tree.nodes[game.current_node_id]
    next_beat = _node(
        parent_id=parent.id,
        speaker_id=hale.id,
        entering=[hale.id],
    )
    new_game = _apply_node(game, next_beat)
    assert new_game.on_stage.count(hale.id) == 1
