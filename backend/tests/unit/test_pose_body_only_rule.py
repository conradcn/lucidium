"""The ``pose`` field's RENDER-FRIENDLY VALUES rule must
restrict pose to body position only — no scene context, no
objects, no other characters.

Without the restriction, the storyteller LLM emits pose values
like ``kneeling beside the altar`` which then go straight into
the SDXL portrait prompt — the renderer ends up trying to draw
the altar, and either smears it into the figure or pushes the
character out of frame. Items the character is holding belong
in ``effects`` (clutched ledger, drawn dagger); other
characters and locations live in the narration ``text`` field.
"""

from __future__ import annotations

from lucidium.domain.world import WorldState
from lucidium.orchestration.prompts import text_gen


def _world() -> WorldState:
    return WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink",
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


def test_rule_restricts_pose_to_body_position() -> None:
    text = _build()
    # The restriction header — pin the lede.
    assert "Pose describes ONLY the character's body position" in text, (
        "text_gen prompt must spell out that pose covers ONLY "
        "the body — without it, scene context bleeds into the "
        "SDXL portrait prompt."
    )


def test_rule_lists_concrete_body_only_examples() -> None:
    """Concrete examples anchor the LLM far better than the
    abstract rule alone — pin each example explicitly so a
    refactor that drops them fails the test."""
    text = _build()
    for example in (
        "kneeling, hands on knees",
        "arms crossed, head tilted",
        "leaning forward",
    ):
        assert example in text, f"expected body-only pose example {example!r} in the prompt"


def test_rule_lists_bad_examples_with_explanations() -> None:
    """The Bad list teaches the LLM what NOT to put in pose by
    showing the failure mode next to each rejected shape."""
    text = _build()
    # Scene leak — mentions of altar / doorway / location.
    assert "scene leak" in text.lower() or "kneeling beside" in text.lower()
    # Object held — pose isn't where held objects go.
    assert "holding a brass key" in text or "object" in text.lower()
    # Other characters in pose.
    assert "another character" in text.lower() or "arms around" in text.lower()


def test_rule_routes_held_objects_to_effects() -> None:
    """Items the character is holding belong in the ``effects``
    field, not pose. The rule must call this out so the LLM
    doesn't smuggle them into pose."""
    text = _build()
    assert "effects" in text.lower()
    # The specific routing instruction — held items go to effects.
    assert "holding" in text.lower() and ("effects" in text.lower()), (
        "rule must explicitly route held items into ``effects``"
    )
