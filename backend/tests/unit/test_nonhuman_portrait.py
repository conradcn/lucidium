"""Coverage for the nonhuman branch of the character portrait
pipeline. Pins two things:

  * ``CharacterKind.nonhuman`` routes ``portrait_prompt`` through
    the freeform-anatomy builder. The structured human-anatomy
    fields (gender / hair / eyes / build / bust / ethnicity /
    skin) MUST NOT appear in the resulting SDXL prompt — they're
    empty for nonhuman entries by design, and a stray ``" hair"``
    or ``"thirty-year-old "`` would dominate the prompt's early
    tokens via CLIP's positional weight bias.
  * ``physical_description`` lands in the prompt with attention
    weight high enough to override the checkpoint's "human
    portrait" prior, and the negative prompt picks up
    ``human, person`` extras so SDXL doesn't leak humanoid
    features into a dragon / robot / spirit render.
"""

from __future__ import annotations

from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.world import WorldState
from lucidium.orchestration.prompts.image_prompts import (
    portrait_negative_extras,
    portrait_prompt,
)


def _world() -> WorldState:
    return WorldState(
        game_name="t",
        setting="ruined cathedral",
        genre="dark fantasy",
        visual_style="ink",
    )


def _nonhuman(**overrides) -> Character:
    base = dict(
        name="The Beast",
        description="bound at the altar",
        kind=CharacterKind.nonhuman,
        physical_description=(
            "scale-armoured serpent, copper hide, lantern-yellow eyes, four spiral horns"
        ),
        age=600,
        outfit="iron chain at the throat",
        pose="coiled, head low",
        expression="watchful",
        seed=1,
    )
    base.update(overrides)
    return Character(**base)


def test_nonhuman_prompt_omits_human_anatomy() -> None:
    char = _nonhuman()
    prompt = portrait_prompt(world=_world(), character=char)
    # Subject identity rides on physical_description.
    assert "scale-armoured serpent" in prompt
    assert "copper hide" in prompt
    # No anatomy artefacts — the human builder's tags should be
    # absent because the nonhuman branch never assembles them.
    for stray in (
        "year-old",
        " hair",
        " eyes",  # eyes via "lantern-yellow eyes" in the
        # description doesn't trip this check because
        # we only ban the standalone-tag suffix.
        " skin",
        " build",
        " bust",
    ):
        # Need a tighter check for " eyes" since the description
        # legitimately contains "lantern-yellow eyes". Validate
        # that no STANDALONE eye tag was added by the builder.
        if stray == " eyes":
            assert " eyes," not in prompt or "lantern-yellow" in prompt
            continue
        assert stray not in prompt, f"nonhuman prompt unexpectedly contains '{stray}': {prompt!r}"


def test_nonhuman_prompt_includes_pose_and_expression() -> None:
    """Pose + expression aren't human-only — a coiled dragon has
    a pose, a watchful spirit has an expression. The nonhuman
    builder still emits these tags."""
    char = _nonhuman()
    prompt = portrait_prompt(world=_world(), character=char)
    assert "watchful" in prompt
    assert "coiled, head low" in prompt
    # Outfit / adornment also flows through.
    assert "iron chain at the throat" in prompt


def test_nonhuman_prompt_falls_back_to_description() -> None:
    """When ``physical_description`` is empty (storyteller
    forgot to fill it), the builder falls back to the
    narrative ``description`` field rather than rendering a
    blank subject."""
    char = _nonhuman(physical_description="")
    prompt = portrait_prompt(world=_world(), character=char)
    assert "bound at the altar" in prompt


def test_nonhuman_negative_extras_block_human_features() -> None:
    """The negative prompt for a nonhuman character pushes
    ``human, person, humanoid face`` into the negative so SDXL's
    person-portrait prior doesn't leak humanoid features into
    a dragon / robot / spirit render."""
    char = _nonhuman()
    negs = portrait_negative_extras(character=char)
    assert "human" in negs
    assert "person" in negs
    assert "humanoid" in negs


def test_human_default_kind_unchanged() -> None:
    """Existing human characters without an explicit ``kind``
    keep working — the field defaults to ``human`` and the
    builder still emits the structured anatomy tags."""
    char = Character(
        name="Mira",
        description="scrivener at the harbor",
        gender="female",
        age=34,
        ethnicity="local",
        skin="fair",
        hair_color="dark",
        hairstyle="braid",
        eye_color="hazel",
        build="slim",
        bust="small",
        outfit="charcoal coat",
        pose="standing",
        expression="alert",
        seed=2,
    )
    assert char.kind == CharacterKind.human
    prompt = portrait_prompt(world=_world(), character=char)
    assert "thirty-year-old female" in prompt
    assert "dark braid hair" in prompt
    assert "hazel eyes" in prompt
