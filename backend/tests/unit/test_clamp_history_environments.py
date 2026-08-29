"""Location markers in committed-history rendering.

``clamp_history`` is the helper that renders ``DialogNode`` history
into the speaker+text block the storyteller / summarizer prompts
read from. Without environment-aware rendering, the LLM only sees
``narrator: ...`` / ``mira-quill: ...`` lines and loses track of
WHERE each beat happened — which makes a character introduced in
the harbor and then referenced in the noodle stall read as a
continuity drift to the model.

These tests pin the contract that ``clamp_history`` injects
``[LOCATION: ...]`` markers between beats whenever the
``location_id`` changes, with optional lighting suffix when the
environment carries it.
"""

from __future__ import annotations

from lucidium.domain.dialog import DialogNode, DialogNodeState
from lucidium.domain.environment import Environment
from lucidium.orchestration.prompts.common import clamp_history


def _node(*, text: str, speaker: str | None, location_id: str | None) -> DialogNode:
    return DialogNode(
        text=text,
        speaker_id=speaker,
        location_id=location_id,
        state=DialogNodeState.committed,
        premise_hash="x",
    )


def _env(*, id_: str, label: str, lighting: str = "") -> Environment:
    return Environment(
        id=id_,
        location_label=label,
        prompt=label,
        prompt_hash="h",
        lighting=lighting,
    )


def test_no_environments_uses_bare_id() -> None:
    """When ``environments`` isn't passed, the marker still appears
    using the raw ``location_id`` as the label — giving the LLM
    SOME scene heading is better than dropping the location info
    entirely. Lighting is omitted (nothing to read it from)."""
    nodes = [
        _node(text="The harbor wakes slow.", speaker=None, location_id="harbor"),
        _node(text='"Late again."', speaker="mira", location_id="harbor"),
    ]
    out = clamp_history(nodes, char_budget=10_000)
    assert out == ('[LOCATION: harbor]\nnarrator: The harbor wakes slow.\nmira: "Late again."')


def test_nodes_without_any_location_skip_markers() -> None:
    """When NO node has a location_id, no marker is emitted —
    backwards compatible with old recorded fixtures whose nodes
    never set the field."""
    nodes = [
        _node(text="Hi.", speaker=None, location_id=None),
        _node(text="There.", speaker=None, location_id=None),
    ]
    out = clamp_history(nodes, char_budget=10_000)
    assert "LOCATION" not in out
    assert out == "narrator: Hi.\nnarrator: There."


def test_location_marker_emitted_at_first_beat() -> None:
    """The first beat's location triggers an opening marker so the
    LLM has a scene heading even on a single-location playthrough."""
    nodes = [
        _node(text="The harbor wakes slow.", speaker=None, location_id="harbor"),
        _node(text='"Late again."', speaker="mira", location_id="harbor"),
    ]
    envs = {"harbor": _env(id_="harbor", label="Stone Harbor", lighting="dawn light")}
    out = clamp_history(nodes, char_budget=10_000, environments=envs)
    expected = (
        "[LOCATION: Stone Harbor — dawn light]\n"
        "narrator: The harbor wakes slow.\n"
        'mira: "Late again."'
    )
    assert out == expected


def test_location_marker_only_on_change() -> None:
    """Successive beats in the SAME location don't repeat the
    marker; only a real location change emits one."""
    nodes = [
        _node(text="A.", speaker=None, location_id="harbor"),
        _node(text="B.", speaker=None, location_id="harbor"),
        _node(text="C.", speaker=None, location_id="noodle-stall"),
        _node(text="D.", speaker=None, location_id="noodle-stall"),
    ]
    envs = {
        "harbor": _env(id_="harbor", label="Harbor"),
        "noodle-stall": _env(id_="noodle-stall", label="Noodle Stall"),
    }
    out = clamp_history(nodes, char_budget=10_000, environments=envs)
    expected = (
        "[LOCATION: Harbor]\n"
        "narrator: A.\n"
        "narrator: B.\n"
        "[LOCATION: Noodle Stall]\n"
        "narrator: C.\n"
        "narrator: D."
    )
    assert out == expected


def test_lighting_suffix_only_when_present() -> None:
    """Environments with empty lighting render as ``[LOCATION:
    label]``; non-empty lighting joins via em-dash."""
    nodes = [
        _node(text="Beat.", speaker=None, location_id="loc"),
    ]
    no_light = clamp_history(
        nodes,
        char_budget=10_000,
        environments={"loc": _env(id_="loc", label="Place", lighting="")},
    )
    assert no_light.startswith("[LOCATION: Place]\n")
    assert "—" not in no_light.split("\n", 1)[0]

    with_light = clamp_history(
        nodes,
        char_budget=10_000,
        environments={"loc": _env(id_="loc", label="Place", lighting="warm gloom")},
    )
    assert with_light.startswith("[LOCATION: Place — warm gloom]\n")


def test_unknown_location_id_falls_back_to_id_string() -> None:
    """When a node references a ``location_id`` that's not in the
    ``environments`` map (race condition: history was committed
    before the env got promoted, or save was partially migrated),
    the marker uses the bare id rather than skipping the marker
    entirely. Better to give the LLM a label even if it's the
    machine id."""
    nodes = [_node(text="Hi.", speaker=None, location_id="unknown-loc")]
    out = clamp_history(nodes, char_budget=10_000, environments={})
    assert out.startswith("[LOCATION: unknown-loc]\n")


def test_node_without_location_does_not_emit_marker() -> None:
    """A Continue beat with no ``location_id`` doesn't reset the
    scene heading — earlier marker stays implicit, no spurious
    transition emitted."""
    nodes = [
        _node(text="A.", speaker=None, location_id="harbor"),
        _node(text="B.", speaker=None, location_id=None),
        _node(text="C.", speaker=None, location_id="harbor"),
    ]
    envs = {"harbor": _env(id_="harbor", label="Harbor")}
    out = clamp_history(nodes, char_budget=10_000, environments=envs)
    # Single LOCATION marker at the top, no second one after the
    # nullable beat (which inherits the implicit scene).
    assert out.count("[LOCATION:") == 1
