"""Storyteller-prompt rules that pin character consistency.

The text_gen prompt has historically been the place where ad-hoc
"don't change pose / outfit unless the beat needs it" guidance has
landed. These tests pin the rule strings so a refactor that drops
the consistency guidance falls visibly instead of silently letting
character state drift back to per-beat churn.

Rules pinned here:

  * ``STATE STABILITY`` clause — instructs the LLM to leave
    pose/outfit/expression alone when the beat doesn't narratively
    require a change. Without this, the storyteller emits
    character_changes on every beat for tonal variety, the
    portrait re-renders on every change, and the player perceives
    the character as flipping clothes/posture every turn.
  * ``EFFECTS`` clause — separates injuries / soot / blood / soaked
    clothing from the wardrobe ``outfit`` field, so a fresh cut
    doesn't have to be re-narrated as part of the outfit string
    forever.
  * Schema includes the ``effects`` field — both new_characters
    (NewCharacterDescriptor) and the existing-character schema
    (LlmCharacterPayload) carry it, so character_changes and
    new_characters can both update it.
"""

from __future__ import annotations

from lucidium.domain.character import Character
from lucidium.domain.dialog import CharacterAttributeField
from lucidium.orchestration.prompts import text_gen as text_gen_prompts
from lucidium.orchestration.prompts.common import render_character_full


def test_text_gen_prompt_includes_state_stability_clause() -> None:
    """The prompt MUST tell the LLM not to churn pose/outfit/
    expression every beat. Pin the rule so a refactor that drops
    it falls a test instead of silently letting the storyteller
    rewrite character state on every turn."""
    rules = text_gen_prompts._TEXT_FORMAT_RULES
    assert "STATE STABILITY" in rules, (
        "text_gen rules dropped the STATE STABILITY clause that keeps "
        "character pose/outfit/expression from flipping every beat"
    )
    assert "Only emit a character_change when the BEAT NARRATIVELY" in rules


def test_text_gen_prompt_describes_effects_field() -> None:
    """Effects guidance must instruct the LLM to use the new
    ``effects`` field for cuts / bruises / soot / soaked clothing
    instead of bleeding those into ``outfit`` (which would force a
    re-render every time a transient effect appears)."""
    rules = text_gen_prompts._TEXT_FORMAT_RULES
    assert "EFFECTS" in rules
    assert "cuts" in rules.lower() or "cut" in rules.lower()
    # Effects must be explicitly named separate from outfit.
    assert "effects" in rules.lower()
    assert "outfit" in rules.lower()


def test_character_attribute_field_enum_includes_effects() -> None:
    """The validator that filters ``character_changes[].field`` must
    accept ``effects`` so the LLM can update an injury / cleanup
    state. Without this enum entry, every effects update silently
    drops at the handler boundary."""
    assert CharacterAttributeField.effects.value == "effects"


def test_render_character_full_lists_effects_under_canon_attributes() -> None:
    """Every storyteller / summarizer prompt's character block
    must show the current ``effects`` so the LLM doesn't
    re-narrate a healed injury or forget an open one."""
    char = Character(
        name="Mira",
        description="harbour scrivener",
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
        effects="blood soaking left sleeve",
        seed=12345,
    )
    block = render_character_full(char)
    assert "effects" in block.lower()
    assert "blood soaking left sleeve" in block


def test_character_with_no_effects_renders_none_placeholder() -> None:
    """When ``effects`` is empty, the canon block shows ``(none)``
    explicitly rather than dropping the field — the LLM needs to
    SEE that there are no current effects so it doesn't invent
    some via narration drift."""
    char = Character(
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
        seed=999,
    )
    block = render_character_full(char)
    assert "effects" in block.lower()
    assert "(none)" in block
