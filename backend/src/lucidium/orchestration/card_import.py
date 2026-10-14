"""Convert between SillyTavern character cards and Lucidium characters.

Import: a card becomes a cast member either directly — when it carries
Lucidium's own ``extensions.lucidium`` record from an earlier export —
or through one LLM call that adapts the card's free text into the
structured appearance fields the portrait pipeline needs (see
``prompts/character_card.py``). The card itself is kept on the
character (``Character.card``) so its personality and example dialogue
keep informing the storyteller, and so exporting it again loses nothing.

Export: any cast member becomes a V2+V3 card. A character that came
from a card exports as that card (renamed if the player renamed them)
plus the Lucidium record; a character the story invented gets a card
composed from its description, appearance, facts and the lines it has
spoken in the story.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..domain.character import (
    DEFAULT_AGE,
    DEFAULT_EXPRESSION,
    DEFAULT_OUTFIT,
    DEFAULT_POSE,
    Character,
    CharacterKind,
    Fact,
    FactConfidence,
)
from ..domain.character_card import LUCIDIUM_EXTENSION_KEY, CharacterCardData
from .prompts.character_card import card_character_adaptation
from .responses import parse_json_object

if TYPE_CHECKING:
    from ..domain.game import Game
    from .session import Session

_log = logging.getLogger(__name__)

LUCIDIUM_EXTENSION_VERSION = 1
CARD_THUMBNAIL_DIRNAME = "_card_thumbs"
DEFAULT_CARDS_DIRNAME = "character_cards"

# Character fields carried in the ``extensions.lucidium`` record. ``name``
# and ``description`` are not here: name is the card's own, and the
# Lucidium description is stored separately so it doesn't overwrite the
# card's (usually much longer) original.
_RECORD_FIELDS: tuple[str, ...] = (
    "kind",
    "physical_description",
    "gender",
    "pronouns",
    "age",
    "ethnicity",
    "skin",
    "hair_color",
    "hairstyle",
    "eye_color",
    "build",
    "bust",
    "outfit",
    "pose",
    "expression",
    "effects",
    "decals",
    "seed",
)
_MAX_IMPORTED_FACTS = 8
_PERSONALITY_PROMPT_CHARS = 600
_EXAMPLE_PROMPT_CHARS = 500

_CHAR_MACRO = re.compile(r"\{\{\s*char\s*\}\}|<BOT>", re.IGNORECASE)
_USER_MACRO = re.compile(r"\{\{\s*user\s*\}\}|<USER>", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def resolve_cards_dir(configured: str, saves_root: Path) -> Path:
    """The card library directory: the configured one, or
    ``<app-data>/character_cards`` next to the saves folder."""
    text = (configured or "").strip()
    if text:
        return Path(text).expanduser().resolve()
    return (saves_root.parent / DEFAULT_CARDS_DIRNAME).resolve()


def expand_macros(text: str, *, char_name: str, user_name: str) -> str:
    """Replace the ``{{char}}`` / ``{{user}}`` placeholders cards use."""
    text = _CHAR_MACRO.sub(lambda _m: char_name, text)
    return _USER_MACRO.sub(lambda _m: user_name, text)


def card_summary(card: CharacterCardData, limit: int = 280) -> str:
    """A one-paragraph preview of the card for lists and prompts."""
    source = card.description or card.personality or card.scenario or card.creator_notes
    flat = _WHITESPACE.sub(
        " ", expand_macros(source, char_name=card.name, user_name="the protagonist")
    ).strip()
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def side_character_line(card: CharacterCardData) -> str:
    """The line the world-init prompt sees for a card-backed side character."""
    return f"{card.name} (from an imported character card): {card_summary(card, 400)}"


def card_prompt_lines(character: Character) -> str:
    """Extra canon lines for the storyteller drawn from the character's
    card: its personality and a sample of how it talks. Empty for
    characters without a card. Bounded so a huge card can't crowd the
    rest of the prompt out."""
    card = character.card
    if card is None:
        return ""

    def _flat(text: str, limit: int) -> str:
        flat = _WHITESPACE.sub(
            " ", expand_macros(text, char_name=character.name, user_name="the protagonist")
        ).strip()
        return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"

    lines = ""
    if card.personality.strip():
        lines += f"  personality (from character card): {_flat(card.personality, _PERSONALITY_PROMPT_CHARS)}\n"
    if card.mes_example.strip():
        sample = card.mes_example.replace("<START>", " ")
        lines += f"  speech sample (from character card): {_flat(sample, _EXAMPLE_PROMPT_CHARS)}\n"
    return lines


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


class LlmCardAdaptation(BaseModel):
    """The adaptation call's reply. Everything defaults so one dropped
    key self-heals instead of failing the import."""

    model_config = ConfigDict(extra="ignore")

    description: str = ""
    kind: str = CharacterKind.human.value
    physical_description: str = ""
    gender: str = ""
    pronouns: str = ""
    age: int = Field(default=DEFAULT_AGE, ge=0)
    ethnicity: str = ""
    skin: str = ""
    hair_color: str = ""
    hairstyle: str = ""
    eye_color: str = ""
    build: str = ""
    bust: str = ""
    outfit: str = DEFAULT_OUTFIT
    pose: str = DEFAULT_POSE
    expression: str = DEFAULT_EXPRESSION
    effects: str = ""
    decals: str = ""
    facts: list[str] = Field(default_factory=list)


def _canon_facts(texts: list[Any]) -> list[Fact]:
    out: list[Fact] = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            out.append(Fact(text=text.strip(), confidence=FactConfidence.canon))
        if len(out) >= _MAX_IMPORTED_FACTS:
            break
    return out


def character_from_extension(
    card: CharacterCardData, *, character_id: str | None = None, seed: int | None = None
) -> Character | None:
    """Rebuild a character exactly from a card Lucidium exported earlier.
    ``None`` when the card has no (valid) Lucidium record."""
    ext = card.extensions.get(LUCIDIUM_EXTENSION_KEY)
    record = ext.get("character") if isinstance(ext, dict) else None
    if not isinstance(record, dict):
        return None
    fields: dict[str, Any] = {f: record[f] for f in _RECORD_FIELDS if f in record}
    if seed is not None:
        fields["seed"] = seed
    fields.setdefault("seed", _random_seed())
    if character_id is not None:
        fields["id"] = character_id
    description = record.get("description")
    try:
        return Character(
            name=card.name,
            description=description
            if isinstance(description, str) and description
            else card_summary(card, 400),
            facts=_canon_facts(record.get("facts") or []),
            card=card,
            **fields,
        )
    except ValidationError:
        _log.warning("card %r has an unusable Lucidium record; adapting it instead", card.name)
        return None


def card_stub(card: CharacterCardData) -> Character:
    """A placeholder cast entry for a card whose adaptation hasn't run
    yet (the New Game flow defers it to Begin, like typed side
    characters)."""
    return Character(
        name=card.name,
        description=card_summary(card, 400),
        outfit="",
        pose="",
        expression="",
        seed=_random_seed(),
        card=card,
    )


def _lorebook_text(card: CharacterCardData) -> str:
    book = card.character_book
    entries = book.get("entries") if isinstance(book, dict) else None
    if not isinstance(entries, list):
        return ""
    lines: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("enabled") is False:
            continue
        content = entry.get("content")
        if isinstance(content, str) and content.strip():
            lines.append(f"- {content.strip()}")
    return "\n".join(lines)


async def adapt_card(
    session: Session,
    card: CharacterCardData,
    *,
    setting: str,
    player_name: str,
    base: Character | None = None,
) -> Character:
    """Turn ``card`` into a full cast member.

    ``base`` (a stub from ``card_stub``) supplies the id and seed so
    anything already keyed on the stub keeps resolving. LLM transport
    errors propagate — the caller decides whether that fails the action.
    A reply that won't parse falls back to the card's own text with
    neutral appearance defaults rather than dropping the character.
    """
    character_id = base.id if base is not None else None
    seed = base.seed if base is not None else None
    exact = character_from_extension(card, character_id=character_id, seed=seed)
    if exact is not None:
        return exact

    user = player_name.strip() or "the protagonist"

    def _x(text: str) -> str:
        return expand_macros(text, char_name=card.name, user_name=user)

    sections = {
        "description": _x(card.description),
        "personality": _x(card.personality),
        "scenario": _x(card.scenario),
        "example dialogue": _x(card.mes_example),
        "lorebook": _x(_lorebook_text(card)),
    }
    raw, _chunks = await session.llm_text(
        card_character_adaptation(name=card.name, sections=sections, setting=setting)
    )
    try:
        adapted = parse_json_object(raw, LlmCardAdaptation)
    except Exception:
        _log.warning(
            "card adaptation for %r did not validate; raw[:300]=%r — using card defaults",
            card.name,
            raw[:300],
        )
        adapted = LlmCardAdaptation()

    try:
        kind = CharacterKind(adapted.kind.strip().lower())
    except ValueError:
        kind = CharacterKind.human
    extra: dict[str, Any] = {}
    if character_id is not None:
        extra["id"] = character_id
    return Character(
        name=card.name,
        description=adapted.description.strip() or card_summary(card, 400),
        kind=kind,
        physical_description=adapted.physical_description,
        gender=adapted.gender,
        pronouns=adapted.pronouns,
        age=adapted.age,
        ethnicity=adapted.ethnicity,
        skin=adapted.skin,
        hair_color=adapted.hair_color,
        hairstyle=adapted.hairstyle,
        eye_color=adapted.eye_color,
        build=adapted.build,
        bust=adapted.bust,
        outfit=adapted.outfit or DEFAULT_OUTFIT,
        pose=adapted.pose or DEFAULT_POSE,
        expression=adapted.expression or DEFAULT_EXPRESSION,
        effects=adapted.effects,
        decals=adapted.decals,
        facts=_canon_facts(adapted.facts),
        seed=seed if seed is not None else _random_seed(),
        card=card,
        **extra,
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def character_record(character: Character) -> dict[str, Any]:
    """The ``extensions.lucidium`` payload for ``character``."""
    record: dict[str, Any] = {f: getattr(character, f) for f in _RECORD_FIELDS}
    record["kind"] = character.kind.value
    record["description"] = character.description
    record["facts"] = [f.text for f in character.facts]
    return {"version": LUCIDIUM_EXTENSION_VERSION, "character": record}


def _appearance(character: Character) -> str:
    if character.kind == CharacterKind.nonhuman:
        parts = [
            ("Form", character.physical_description),
            ("Age", str(character.age)),
            ("Adornment", character.outfit),
            ("Markings", character.decals),
        ]
    else:
        hair = " ".join(p for p in (character.hair_color, character.hairstyle) if p)
        parts = [
            ("Gender", character.gender),
            ("Pronouns", character.pronouns),
            ("Age", str(character.age)),
            ("Ethnicity", character.ethnicity),
            ("Skin", character.skin),
            ("Build", character.build),
            ("Hair", hair),
            ("Eyes", character.eye_color),
            ("Outfit", character.outfit),
            ("Markings", character.decals),
        ]
    return "\n".join(f"{label}: {value}" for label, value in parts if value.strip())


def _spoken_lines(character: Character, game: Game | None, limit: int = 4) -> list[str]:
    if game is None:
        return []
    lines: list[str] = []
    nodes = game.dialog_tree.nodes
    for node_id in game.dialog_tree.committed_path:
        node = nodes.get(node_id)
        if node is None or node.speaker_id != character.id:
            continue
        text = (node.text or "").strip()
        if text:
            lines.append(text)
        if len(lines) >= limit:
            break
    return lines


def card_from_character(character: Character, game: Game | None) -> CharacterCardData:
    """Build the card to export for ``character``."""
    record = {LUCIDIUM_EXTENSION_KEY: character_record(character)}
    if character.card is not None:
        original = character.card
        return original.model_copy(
            update={
                "name": character.name,
                "extensions": {**original.extensions, **record},
                "source_file": "",
            }
        )

    description = character.description.strip()
    appearance = _appearance(character)
    if appearance:
        description += f"\n\n[Appearance]\n{appearance}"
    if character.facts:
        description += "\n\n[Known facts]\n" + "\n".join(f"- {f.text}" for f in character.facts)
    world = game.world if game is not None else None
    spoken = _spoken_lines(character, game)
    scenario = ""
    tags = ["Lucidium"]
    notes = "Exported from Lucidium."
    if world is not None:
        scenario = ". ".join(
            p
            for p in (
                f"Setting: {world.setting}" if world.setting else "",
                f"Genre: {world.genre}" if world.genre else "",
            )
            if p
        )
        if world.genre.strip():
            tags.append(world.genre.strip())
        if world.game_name.strip():
            notes = f'Exported from the Lucidium story "{world.game_name.strip()}".'
    return CharacterCardData(
        name=character.name,
        description=description,
        scenario=scenario,
        first_mes=spoken[0] if spoken else "",
        mes_example="\n".join(f"<START>\n{{{{char}}}}: {line}" for line in spoken[1:]),
        creator_notes=notes,
        tags=tags,
        character_version="1",
        extensions=record,
        spec="chara_card_v3",
    )


def portrait_bytes(character: Character) -> bytes | None:
    """The character's newest portrait, if it is on disk."""
    for image in character.images:
        try:
            return Path(image.path).read_bytes()
        except OSError:
            continue
    return None


def _random_seed() -> int:
    from .session import random_seed

    return random_seed()
