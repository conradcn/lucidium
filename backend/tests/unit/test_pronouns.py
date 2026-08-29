"""Pronouns are an optional free-text Character / NPC attribute
threaded through:
  * ``Character.pronouns`` and ``NewCharacterDescriptor.pronouns``
    domain fields,
  * ``LlmCharacterPayload.pronouns`` so LLM-generated NPCs can
    declare their own pronouns,
  * ``CharacterAttributeField.pronouns`` so retcons and per-beat
    ``character_change`` mutations can update them,
  * ``render_character_full`` so the storyteller's CANON
    ATTRIBUTES block carries an explicit "USE THESE pronouns"
    line when the field is populated.

Empty falls through to the prior behaviour (storyteller derives
from gender / context) so legacy saves and tests stay valid.
"""

from __future__ import annotations

from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import (
    CharacterAttributeField,
    CharacterChange,
    NewCharacterDescriptor,
)
from lucidium.orchestration.prompts.common import render_character_full
from lucidium.orchestration.responses import LlmCharacterPayload


def _char(**over: object) -> Character:
    base: dict[str, object] = dict(
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
        kind=CharacterKind.human,
    )
    base.update(over)
    return Character(**base)  # type: ignore[arg-type]


def test_character_has_pronouns_field_default_empty() -> None:
    """Schema field exists and defaults to empty so legacy
    saves and existing tests don't have to populate it."""
    char = _char()
    assert char.pronouns == ""


def test_character_pronouns_accepts_freeform_value() -> None:
    """Free text — covers neopronouns, setting-specific forms,
    "she/her", "they/them", whatever the player picks."""
    char = _char(pronouns="they/them")
    assert char.pronouns == "they/them"


def test_new_character_descriptor_has_pronouns_field() -> None:
    desc = NewCharacterDescriptor(
        id="iris",
        name="Iris",
        description="archivist",
        age=28,
        outfit="wool coat",
        pose="standing",
        expression="alert",
    )
    assert desc.pronouns == ""
    desc2 = NewCharacterDescriptor(
        id="iris",
        name="Iris",
        description="archivist",
        age=28,
        outfit="wool coat",
        pose="standing",
        expression="alert",
        pronouns="xe/xir",
    )
    assert desc2.pronouns == "xe/xir"


def test_llm_character_payload_accepts_pronouns() -> None:
    """LLM responses carry pronouns when the model populates
    them; legacy responses without the field still parse."""
    payload = LlmCharacterPayload(
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
    )
    assert payload.pronouns == ""
    payload2 = LlmCharacterPayload(
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
        pronouns="she/her",
    )
    assert payload2.pronouns == "she/her"


def test_pronouns_in_character_attribute_field_enum() -> None:
    """Without this entry, retcon updates and beat-level
    ``character_change`` mutations targeting pronouns silently
    fail validation."""
    assert CharacterAttributeField.pronouns in CharacterAttributeField


def test_character_change_can_update_pronouns() -> None:
    """A storyteller beat or retcon can flip a character's
    pronouns mid-narrative — common for trans/coming-out
    arcs and for non-binary characters whose presentation
    shifts across the story."""
    change = CharacterChange(
        character_id="iris",
        field=CharacterAttributeField.pronouns,
        new_value="they/them",
    )
    assert change.field == CharacterAttributeField.pronouns
    assert change.new_value == "they/them"


def test_render_character_full_includes_pronouns_line_when_set() -> None:
    """Storyteller's CANON ATTRIBUTES block surfaces an
    explicit pronouns line so the LLM honours the player's
    choice instead of guessing each turn."""
    char = _char(pronouns="they/them")
    rendered = render_character_full(char)
    assert "they/them" in rendered
    assert "USE THESE" in rendered or "do not substitute" in rendered.lower()


def test_render_character_full_omits_pronouns_line_when_empty() -> None:
    """No pronouns set → no line — falls back to the
    pre-pronouns behaviour where the storyteller picks from
    gender / context. Legacy saves / characters with empty
    pronouns must keep narrating exactly as before."""
    char = _char(pronouns="")
    rendered = render_character_full(char)
    assert "USE THESE" not in rendered


def test_render_character_full_includes_pronouns_for_nonhuman() -> None:
    """Nonhuman pipeline takes a different code path; pronouns
    line still emits when set so a robot / spirit / monster
    that prefers ``it/its`` or ``they/them`` is honoured."""
    char = _char(
        kind=CharacterKind.nonhuman,
        physical_description="bronze automaton",
        pronouns="it/its",
    )
    rendered = render_character_full(char)
    assert "it/its" in rendered
    assert "NONHUMAN" in rendered
