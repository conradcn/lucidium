"""SillyTavern-compatible character card data.

A character card is the de-facto interchange format for roleplay
characters: a PNG (the avatar) carrying the character definition as
base64 JSON in a ``tEXt`` chunk, or a bare ``.json`` file, or a
``.charx`` zip. Three spec generations exist in the wild:

  * V1 — flat object: ``name`` / ``description`` / ``personality`` /
    ``scenario`` / ``first_mes`` / ``mes_example``.
  * V2 (``chara_card_v2``) — ``{"spec", "spec_version", "data": {...}}``
    adding creator notes, system prompt, alternate greetings, tags,
    an embedded lorebook (``character_book``) and ``extensions``.
  * V3 (``chara_card_v3``) — V2 plus assets, group-only greetings,
    multilingual creator notes and timestamps.

``CharacterCardData`` is the normalised superset Lucidium stores on a
``Character`` that came from (or is going to) a card. It keeps every
field the specs define so an imported card exports back out without
losing anything; fields Lucidium doesn't model (V3 assets, timestamps,
unknown keys an editor added) ride along in ``extra_fields``.

Parsing and writing the on-disk formats lives in
``persistence/character_cards.py``; this module is just the shape.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Key under ``extensions`` where Lucidium stashes its own structured
# character record when it exports a card. Other frontends preserve
# unknown extension keys (the V2 spec requires it), so a card that
# round-trips through SillyTavern comes back with its exact appearance
# and Lucidium can skip the LLM adaptation pass on re-import.
LUCIDIUM_EXTENSION_KEY = "lucidium"


class CharacterCardData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    personality: str = ""
    scenario: str = ""
    first_mes: str = ""
    mes_example: str = ""
    creator_notes: str = ""
    system_prompt: str = ""
    post_history_instructions: str = ""
    alternate_greetings: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    creator: str = ""
    character_version: str = ""
    nickname: str = ""
    extensions: dict[str, Any] = Field(default_factory=dict)
    # The embedded lorebook, kept verbatim — its entry schema is large
    # and frontends extend it freely, so modelling it would only lose
    # data on export.
    character_book: dict[str, Any] | None = None
    # V3-only (and unknown) ``data`` keys, preserved for export.
    extra_fields: dict[str, Any] = Field(default_factory=dict)
    # Which spec the card was read from (``chara_card_v1`` / ``_v2`` /
    # ``_v3``). Informational; exports always write V2 + V3.
    spec: str = "chara_card_v2"
    # File name (not path) inside the card library the card was
    # imported from. Empty for characters exported from a story.
    source_file: str = ""
