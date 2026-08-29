"""Mid-game character kind transitions.

The cast UI has always exposed a kind toggle (human ↔ nonhuman)
that goes through ``c2s/edit/character`` — that path bypasses
``CharacterAttributeField``'s allow-list and worked. The
LLM-driven paths (retcon and per-beat ``character_change``)
DID gate on the enum, so an LLM that wanted to transform a
character mid-narration was silently dropped. This file pins
the addition of ``kind`` and ``physical_description`` to the
enum so both paths now allow the transition.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lucidium.domain.dialog import (
    CharacterAttributeField,
    CharacterChange,
)


def test_kind_is_a_valid_character_attribute_field() -> None:
    """Pin: ``kind`` is in the enum so retcon updates and
    per-beat character_change mutations targeting it pass
    validation."""
    assert CharacterAttributeField.kind in CharacterAttributeField
    assert CharacterAttributeField.kind.value == "kind"


def test_physical_description_is_a_valid_character_attribute_field() -> None:
    """Counterpart for the nonhuman pipeline's freeform
    description field — needed alongside ``kind`` so the
    LLM can transition a character to nonhuman AND populate
    the field that pipeline reads."""
    assert CharacterAttributeField.physical_description in CharacterAttributeField
    assert CharacterAttributeField.physical_description.value == "physical_description"


def test_character_change_accepts_kind_field() -> None:
    """A storyteller beat can encode a transformation moment as
    ``character_change`` with field=kind. Pre-fix this raised
    pydantic validation; the enum entry now lets it parse."""
    change = CharacterChange(
        character_id="iris",
        field=CharacterAttributeField.kind,
        new_value="nonhuman",
    )
    assert change.field == CharacterAttributeField.kind
    assert change.new_value == "nonhuman"


def test_character_change_accepts_physical_description() -> None:
    """A retcon or beat that adds a physical_description to a
    character that just transitioned to nonhuman parses too."""
    change = CharacterChange(
        character_id="iris",
        field=CharacterAttributeField.physical_description,
        new_value="towering bronze automaton with copper-rivet seams",
    )
    assert change.field == CharacterAttributeField.physical_description


def test_invalid_field_still_rejected() -> None:
    """Defensive: the enum is still an allow-list. Random
    field names not in the enum are rejected so the LLM
    can't write to arbitrary character attributes."""
    with pytest.raises(ValidationError):
        CharacterChange(
            character_id="iris",
            field="invented_attribute",  # type: ignore[arg-type]
            new_value="x",
        )
