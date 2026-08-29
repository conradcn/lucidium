"""Character-descriptor repair prompt builder."""

from __future__ import annotations

from ...domain.character import Character
from ...domain.world import WorldState
from .common import SYSTEM_GAME_DIRECTOR

_REPAIR_SHAPE = """{
  "name": str, "description": str,
  "kind": "human"|"nonhuman",
  "physical_description": str,                          // nonhuman only; "" for human
  "gender": str, "age": int,                            // human; gender "" if nonhuman
  "ethnicity": str, "skin": str, "hair_color": str, "hairstyle": str,
  "eye_color": str, "build": str, "bust": str,         // ALL "" if kind="nonhuman"
  "outfit": str, "pose": str, "expression": str
}"""


def build(
    *, world: WorldState, character: Character, missing_fields: list[str]
) -> list[dict[str, str]]:
    from ...domain.character import CharacterKind

    kind_text = (
        "nonhuman — populate ``physical_description`` (6-15 words "
        "covering silhouette, materials, colours, distinguishing "
        "features) and leave human-anatomy fields (gender, "
        "ethnicity, skin, hair_color, hairstyle, eye_color, build, "
        "bust) as empty strings"
        if character.kind == CharacterKind.nonhuman
        else "human — populate the structured anatomy fields"
    )
    user = (
        f"GAME: {world.game_name} | setting: {world.setting} | genre: {world.genre}\n\n"
        f"Existing partial character:\n"
        f"  name: {character.name}\n"
        f"  description: {character.description}\n"
        f"  kind: {character.kind.value} ({kind_text})\n"
        f"  age: {character.age}\n"
        f"Missing or empty fields: {', '.join(missing_fields)}\n\n"
        "Return a complete character descriptor that fills the missing fields "
        "while preserving everything that's already set. Keep ``kind`` "
        "unchanged. Return JSON only.\n" + _REPAIR_SHAPE
    )
    return [
        {"role": "system", "content": SYSTEM_GAME_DIRECTOR},
        {"role": "user", "content": user},
    ]
