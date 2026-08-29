"""It is impossible for a character to be off-stage while talking.

A voice with no body on screen — a name tag over dialog while the
stage shows nobody, or shows everyone except the one speaking — is
the stage bug players notice instantly. Rather than police every
writer of ``Game.on_stage`` (per-beat entering/leaving lists, the
summarizer's idle-cleanup pass, retcon rewriting ``speaker_id``,
undo, save migration), the invariant is repaired at the single point
every mutation funnels through: ``Session.install_game`` ->
``domain.game.with_speaker_on_stage``.

These tests pin both halves:

  * the repair itself (promote a present speaker; drop the
    attribution for a ``removed`` one rather than resurrect them),
  * the DEFERRED EXIT rule in ``_apply_node`` — a character who
    speaks their own exit line stays visible for that beat and is
    retired on the next, instead of vanishing mid-sentence.
"""

from __future__ import annotations

from lucidium.api.handlers import _apply_node
from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import DialogNode, DialogNodeState, DialogTree
from lucidium.domain.game import Game, with_speaker_on_stage
from lucidium.domain.world import WorldState
from lucidium.orchestration.session import Session


def _world() -> WorldState:
    return WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink wash",
    )


def _char(name: str, *, is_player: bool = False, removed: bool = False) -> Character:
    return Character(
        name=name,
        description=f"the {name}",
        gender="female",
        age=30,
        ethnicity="local",
        skin="pale",
        hair_color="black",
        hairstyle="short",
        eye_color="grey",
        build="slight",
        bust="moderate",
        outfit="oilskin",
        pose="standing",
        expression="watchful",
        seed=3,
        kind=CharacterKind.human,
        is_player=is_player,
        removed=removed,
        removed_reason="dismissed" if removed else "",
    )


def _node(
    *,
    parent_id: str | None = None,
    speaker_id: str | None = None,
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


def _game(
    *,
    characters: list[Character],
    node: DialogNode,
    on_stage: list[str],
    extra_nodes: list[DialogNode] | None = None,
) -> Game:
    nodes = {n.id: n for n in [*(extra_nodes or []), node]}
    return Game(
        world=_world(),
        characters={c.id: c for c in characters},
        dialog_tree=DialogTree(
            nodes=nodes,
            root_id=node.id,
            committed_path=[node.id],
        ),
        current_node_id=node.id,
        on_stage=list(on_stage),
    )


# ---------- the repair -------------------------------------------------------


def test_offstage_speaker_is_put_on_stage() -> None:
    hale = _char("Hale")
    mira = _char("Mira")
    game = _game(
        characters=[hale, mira],
        node=_node(speaker_id=hale.id),
        on_stage=[mira.id],
    )
    assert with_speaker_on_stage(game).on_stage == [mira.id, hale.id]


def test_speaker_already_on_stage_is_untouched() -> None:
    hale = _char("Hale")
    game = _game(
        characters=[hale],
        node=_node(speaker_id=hale.id),
        on_stage=[hale.id],
    )
    # Identity: the common path must not churn the Game object.
    assert with_speaker_on_stage(game) is game


def test_narrator_beat_leaves_the_stage_alone() -> None:
    hale = _char("Hale")
    game = _game(characters=[hale], node=_node(speaker_id=None), on_stage=[])
    assert with_speaker_on_stage(game) is game


def test_unknown_speaker_id_leaves_the_stage_alone() -> None:
    """A hallucinated id (or the literal "narrator") resolves to no
    character, so the renderer already draws the line unattributed —
    there's nobody to put on stage."""
    hale = _char("Hale")
    game = _game(characters=[hale], node=_node(speaker_id="narrator"), on_stage=[])
    result = with_speaker_on_stage(game)
    assert result.on_stage == []


def test_player_speaker_is_never_pushed_on_stage() -> None:
    """The player character is the camera lens, not a stage actor
    (FR-034a) — they have no portrait to show."""
    you = _char("You", is_player=True)
    game = _game(characters=[you], node=_node(speaker_id=you.id), on_stage=[])
    assert with_speaker_on_stage(game).on_stage == []


def test_removed_speaker_loses_the_attribution_instead_of_returning() -> None:
    """``removed`` (narrated dead / dismissed from the cast panel) is
    an explicit fiat that outranks a stray beat. We don't resurrect
    them — we drop the speaker tag, so the line ships as narration
    and no name hangs over an empty stage."""
    ghost = _char("Hale", removed=True)
    node = _node(speaker_id=ghost.id)
    game = _game(characters=[ghost], node=node, on_stage=[])
    result = with_speaker_on_stage(game)
    assert result.on_stage == []
    assert result.dialog_tree.nodes[node.id].speaker_id is None
    assert result.dialog_tree.nodes[node.id].text == "x"


def test_install_game_enforces_the_invariant() -> None:
    """The choke point: any caller installing a state where the
    current beat's speaker is off-stage gets it repaired."""
    hale = _char("Hale")
    game = _game(characters=[hale], node=_node(speaker_id=hale.id), on_stage=[])
    session = Session.__new__(Session)  # no I/O; install_game only sets .game
    session.install_game(game)
    assert session.game is not None
    assert session.game.on_stage == [hale.id]


def test_summarizer_style_offstaging_cannot_mute_the_current_speaker() -> None:
    """The summarizer's idle-character cleanup writes a trimmed
    on_stage list; if it trims the character who is speaking the
    beat on screen right now, install_game puts them back."""
    hale = _char("Hale")
    mira = _char("Mira")
    game = _game(
        characters=[hale, mira],
        node=_node(speaker_id=hale.id),
        on_stage=[hale.id, mira.id],
    )
    pruned = game.model_copy(update={"on_stage": []})  # what the pass would install
    session = Session.__new__(Session)
    session.install_game(pruned)
    assert session.game is not None
    assert session.game.on_stage == [hale.id]


# ---------- deferred exit ----------------------------------------------------


def test_speaker_leaving_on_their_own_beat_stays_visible_for_it() -> None:
    """ "Fine — I'm leaving," she says. The LLM puts her in this
    beat's leaving list; she must still be on stage while the player
    reads the line she says."""
    hale = _char("Hale")
    mira = _char("Mira")
    parent = _node()
    game = _game(
        characters=[hale, mira],
        node=parent,
        on_stage=[hale.id, mira.id],
    )
    exit_beat = _node(parent_id=parent.id, speaker_id=mira.id, leaving=[mira.id])
    walked = _apply_node(game, exit_beat)
    assert mira.id in walked.on_stage
    assert hale.id in walked.on_stage


def test_deferred_exit_retires_the_speaker_on_the_next_beat() -> None:
    """Visible for the line they speak, gone on the next beat — the
    exit still happens, it just isn't allowed to land mid-sentence."""
    hale = _char("Hale")
    mira = _char("Mira")
    parent = _node()
    game = _game(
        characters=[hale, mira],
        node=parent,
        on_stage=[hale.id, mira.id],
    )
    exit_beat = _node(parent_id=parent.id, speaker_id=mira.id, leaving=[mira.id])
    walked = _apply_node(game, exit_beat)
    after = _apply_node(walked, _node(parent_id=exit_beat.id, speaker_id=hale.id))
    assert mira.id not in after.on_stage
    assert hale.id in after.on_stage


def test_deferred_exit_yields_to_a_character_who_walks_back_in() -> None:
    """She slams the door, then shoves it open again on the next
    beat. The deferred retire must not cancel a fresh entrance."""
    mira = _char("Mira")
    parent = _node()
    game = _game(characters=[mira], node=parent, on_stage=[mira.id])
    exit_beat = _node(parent_id=parent.id, speaker_id=mira.id, leaving=[mira.id])
    walked = _apply_node(game, exit_beat)
    back = _apply_node(
        walked,
        _node(parent_id=exit_beat.id, entering=[mira.id]),
    )
    assert back.on_stage == [mira.id]


def test_non_speaker_still_leaves_on_the_beat_that_says_so() -> None:
    """The deferral is only for the speaker — a silent character the
    beat walks off leaves immediately, and re-applying the list on
    the next beat is a no-op rather than a second removal."""
    hale = _char("Hale")
    mira = _char("Mira")
    parent = _node()
    game = _game(
        characters=[hale, mira],
        node=parent,
        on_stage=[hale.id, mira.id],
    )
    beat = _node(parent_id=parent.id, speaker_id=mira.id, leaving=[hale.id])
    walked = _apply_node(game, beat)
    assert walked.on_stage == [mira.id]
    after = _apply_node(walked, _node(parent_id=beat.id, speaker_id=mira.id))
    assert after.on_stage == [mira.id]
