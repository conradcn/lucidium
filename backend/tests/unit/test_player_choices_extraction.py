"""``collect_player_choices`` extracts player-driven moments from
committed history. The summarizer's profile-inference step uses
this filtered list (NOT the full history) so it can't accidentally
infer "likes haunted houses" from the engine deciding the setting
was a haunted house — taste signals must come from things the
player actually selected.

Two kinds of moments count:
  * Menu picks — node has ``chosen_option_id``; the option text
    is fetched from the parent's ``options``.
  * Typed inputs — same shape as ``collect_player_typed_quotes``.

Engine-authored beats (narrator / NPC dialogue / scene setup)
must NOT appear in the output.
"""

from __future__ import annotations

from lucidium.domain.dialog import (
    DialogNode,
    DialogNodeState,
    DialogOption,
    GenerationMetadata,
)
from lucidium.orchestration.summarizer import collect_player_choices


def _engine_beat(
    *,
    node_id: str,
    parent_id: str | None,
    speaker_id: str | None,
    text: str,
    options: list[DialogOption] | None = None,
    chosen_option_id: str | None = None,
) -> DialogNode:
    """A node the storyteller LLM authored. Has generation_metadata
    so it doesn't get mistaken for a player-typed input."""
    return DialogNode(
        id=node_id,
        parent_id=parent_id,
        chosen_option_id=chosen_option_id,
        speaker_id=speaker_id,
        text=text,
        options=options or [],
        state=DialogNodeState.committed,
        premise_hash="x",
        generation_metadata=GenerationMetadata(model="some-model"),
    )


def _typed_beat(*, node_id: str, parent_id: str, text: str) -> DialogNode:
    """A free-text submission from the player. ``speaker_id`` and
    ``chosen_option_id`` both None, no model in generation_metadata."""
    return DialogNode(
        id=node_id,
        parent_id=parent_id,
        chosen_option_id=None,
        speaker_id=None,
        text=text,
        state=DialogNodeState.committed,
        premise_hash="x",
        generation_metadata=GenerationMetadata(),
    )


def test_menu_picks_render_with_picked_prefix() -> None:
    """A node with ``chosen_option_id`` set produces a ``picked:
    <option text>`` line by looking up the option text on the
    parent's ``options`` list."""
    parent = _engine_beat(
        node_id="p1",
        parent_id=None,
        speaker_id=None,
        text="A door at the end of the hall.",
        options=[
            DialogOption(id="opt-open", text="Open the door."),
            DialogOption(id="opt-leave", text="Walk away."),
        ],
    )
    chosen = _engine_beat(
        node_id="p2",
        parent_id="p1",
        speaker_id=None,
        text="The door swings open.",
        chosen_option_id="opt-open",
    )
    out = collect_player_choices([parent, chosen])
    assert out == ["picked: Open the door."]


def test_typed_inputs_render_with_typed_prefix() -> None:
    """Free-text beats appear as ``typed: <text>``."""
    parent = _engine_beat(
        node_id="n1",
        parent_id=None,
        speaker_id=None,
        text="The hall stretches.",
    )
    typed = _typed_beat(node_id="n2", parent_id="n1", text="I shout for someone.")
    out = collect_player_choices([parent, typed])
    assert out == ["typed: I shout for someone."]


def test_engine_authored_beats_are_excluded() -> None:
    """Narrator beats and NPC dialogue with no chosen_option must
    not leak into player-choice extraction. They're engine output,
    not player taste signals."""
    nodes = [
        _engine_beat(
            node_id="n1",
            parent_id=None,
            speaker_id=None,
            text="The wind picks up.",
        ),
        _engine_beat(
            node_id="n2",
            parent_id="n1",
            speaker_id="mira-quill",
            text='"You are early."',
        ),
        _engine_beat(
            node_id="n3",
            parent_id="n2",
            speaker_id=None,
            text="A bell tolls in the distance.",
        ),
    ]
    out = collect_player_choices(nodes)
    assert out == []


def test_chosen_option_with_unknown_id_is_skipped() -> None:
    """If the parent's options list doesn't carry the chosen id
    (corrupted save / partial migration), skip the entry rather
    than crashing or emitting an empty ``picked:`` line."""
    parent = _engine_beat(
        node_id="p1",
        parent_id=None,
        speaker_id=None,
        text="X",
        options=[DialogOption(id="opt-known", text="Known.")],
    )
    chosen = _engine_beat(
        node_id="p2",
        parent_id="p1",
        speaker_id=None,
        text="Y",
        chosen_option_id="opt-MISSING",
    )
    assert collect_player_choices([parent, chosen]) == []


def test_mixed_history_returns_chronological_choices_only() -> None:
    """Realistic-shape history: narrator beats, NPC dialogue, a
    menu pick, more narration, a typed line. Output preserves
    chronological order and excludes everything that wasn't a
    direct player action."""
    n1 = _engine_beat(node_id="n1", parent_id=None, speaker_id=None, text="Foggy harbor.")
    n2 = _engine_beat(
        node_id="n2",
        parent_id="n1",
        speaker_id=None,
        text="A figure waits.",
        options=[
            DialogOption(id="opt-approach", text="Approach the figure."),
            DialogOption(id="opt-watch", text="Watch from cover."),
        ],
    )
    n3 = _engine_beat(
        node_id="n3",
        parent_id="n2",
        speaker_id=None,
        text="You step closer.",
        chosen_option_id="opt-approach",
    )
    n4 = _engine_beat(node_id="n4", parent_id="n3", speaker_id="mira-quill", text='"Late again."')
    n5 = _typed_beat(node_id="n5", parent_id="n4", text="I ask for her name.")
    out = collect_player_choices([n1, n2, n3, n4, n5])
    assert out == [
        "picked: Approach the figure.",
        "typed: I ask for her name.",
    ]


def test_limit_keeps_most_recent_choices() -> None:
    """When choice count exceeds ``limit``, the helper returns the
    most-recent slice in chronological order."""
    nodes: list[DialogNode] = [
        _engine_beat(
            node_id="root",
            parent_id=None,
            speaker_id=None,
            text="Start.",
            options=[DialogOption(id=f"opt-{i}", text=f"Option {i}.") for i in range(5)],
        ),
    ]
    parent_id = "root"
    for i in range(5):
        nodes.append(
            _engine_beat(
                node_id=f"c{i}",
                parent_id=parent_id,
                speaker_id=None,
                text=f"Beat {i}",
                chosen_option_id=f"opt-{i}",
                options=[DialogOption(id=f"opt-{j}", text=f"Option {j}.") for j in range(5)],
            )
        )
        parent_id = f"c{i}"
    out = collect_player_choices(nodes, limit=3)
    assert len(out) == 3
    assert out == [
        "picked: Option 2.",
        "picked: Option 3.",
        "picked: Option 4.",
    ]
