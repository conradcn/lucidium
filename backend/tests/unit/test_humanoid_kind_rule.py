"""Pin the NONHUMAN RULE wording so humanoid-but-supernatural
characters keep rendering as people.

The earlier rule said "kind defaults to human" but didn't list
the edge cases — a Goddess, an angel, a vampire, an elf could
plausibly be classified either way, and the storyteller LLM
was sometimes choosing ``\"nonhuman\"`` and producing abstract
non-person renders for what should have been person-shaped
deities. The updated rule explicitly enumerates humanoid
supernatural beings as ``\"human\"`` and reserves
``\"nonhuman\"`` for entities a witness would call ``it``.
"""

from __future__ import annotations

from lucidium.domain.character import Character
from lucidium.domain.world import WorldState
from lucidium.orchestration.prompts import text_gen


def _world() -> WorldState:
    return WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink",
    )


def _player() -> Character:
    return Character(
        is_player=True,
        name="Iris",
        description="archivist",
        gender="female",
        age=28,
        ethnicity="local",
        skin="fair",
        hair_color="dark",
        hairstyle="braid",
        eye_color="hazel",
        build="slim",
        bust="small",
        outfit="coat",
        pose="standing",
        expression="alert",
        seed=1,
    )


def _build() -> str:
    msgs = text_gen.build(
        world=_world(),
        history=[],
        on_stage={},
        off_stage={},
        chosen_option_text=None,
    )
    return msgs[-1]["content"]


def test_rule_lists_humanoid_supernatural_examples_as_human() -> None:
    """The rule must call out the cases the storyteller used to
    miscategorise — Goddesses / angels / vampires / elves get
    explicit human classification so the LLM picks ``\"human\"``
    for them and the embedded backend renders person-shaped
    portraits with the structured anatomy fields."""
    text = _build()
    # "HUMANOID is HUMAN" header — pin the lede.
    assert "HUMANOID is HUMAN" in text, (
        "expected the explicit humanoid-is-human header in the "
        "NONHUMAN RULE — without it the storyteller drifts toward "
        "kind=nonhuman for divine / supernatural humanoids."
    )
    # Specific examples the rule must enumerate.
    for example in (
        "Goddess",
        "angel",
        "elf",
        "vampire",
        "demon",
    ):
        assert example.lower() in text.lower(), (
            f"expected {example!r} in the humanoid examples list"
        )


def test_rule_clarifies_when_to_use_nonhuman() -> None:
    """The rule must define the ``it`` heuristic — kind=nonhuman is
    for shapes a witness would point at and call ``it``, not
    ``she``/``he``/``they``. Pinning the heuristic keeps the LLM
    from over-applying nonhuman to anything supernatural."""
    text = _build()
    # The ``it`` heuristic must be present.
    assert "``it``" in text or '"it"' in text or "'it'" in text, (
        "expected the ``it``-vs-pronoun heuristic in the rule"
    )
    # Concrete nonhuman examples must include shape-types, not
    # just supernatural-types.
    assert "tentacled" in text.lower() or "swarm" in text.lower() or "orb" in text.lower(), (
        "expected at least one shape-based nonhuman example "
        "(tentacled / swarm / orb) so the LLM has a clear "
        "anchor for what counts."
    )
