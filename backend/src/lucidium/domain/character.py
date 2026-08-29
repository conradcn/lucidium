"""Character entity, facts, portraits, and age-band rules."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .ids import new_id


class FactConfidence(StrEnum):
    canon = "canon"
    inferred = "inferred"


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    text: str
    confidence: FactConfidence = FactConfidence.inferred
    source_node_id: str | None = None


class CharacterImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    path: str
    prompt_hash: str
    # Hash of every portrait input EXCEPT the expression (appearance,
    # outfit, pose, lighting, seed, kind, ...). When a later render has
    # the same ``identity_hash`` but a different ``prompt_hash``, the
    # expression is the ONLY thing that changed — the orchestrator can
    # then refresh just the face via a localized inpaint instead of
    # re-rendering the whole portrait. Empty on images saved before this
    # field existed (and on non-embedded backends); the fast path simply
    # doesn't trigger for those and a full render happens as before.
    identity_hash: str = ""
    attributes_snapshot: dict[str, str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CharacterKind(StrEnum):
    """Branches the character generation pipeline.

    ``human`` (default) flows through the existing structured
    anatomy fields — gender, ethnicity, hair, eye colour, build,
    etc. — that map cleanly onto SDXL's portrait prior. ``nonhuman``
    skips most of those (a dragon doesn't have ``hair_color``,
    a ghost doesn't have ``bust``) and uses the freeform
    ``physical_description`` field instead. This lets the
    storyteller introduce robots, monsters, spirits, animals,
    eldritch beings, and the like without forcing them into a
    person-shaped schema and producing nonsense fields.
    """

    human = "human"
    nonhuman = "nonhuman"


# Intelligent fallback values for the character-descriptor fields the
# storyteller LLM occasionally omits (age, outfit, pose, expression).
# Applied as field defaults on the LLM-facing descriptor models
# (``NewCharacterDescriptor``, ``LlmCharacterPayload``) and on
# ``Character`` itself, so a single missing key yields a sensible value
# instead of a hard ``ValidationError``. Before this, a dropped field
# failed the whole ``world_init`` parse and bounced the call back to the
# LLM to retry — which could fail several times in a row and stall new
# game init indefinitely. These defaults only apply when the key is
# ABSENT; an explicit empty string the model sends is preserved. The
# neutral values keep narration/portraits sane for a just-introduced
# character until the storyteller fleshes them out on a later turn.
DEFAULT_AGE = 30
DEFAULT_OUTFIT = "practical everyday clothing"
DEFAULT_POSE = "standing"
DEFAULT_EXPRESSION = "neutral"


class Character(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    is_player: bool = False
    name: str
    description: str
    # Branch the rendering / prompt pipeline. ``human`` keeps the
    # structured anatomy fields below as canon; ``nonhuman`` reads
    # ``physical_description`` instead and treats the human-only
    # fields as best-effort metadata (often empty). Defaults to
    # ``human`` so existing saves and tests stay valid without
    # migration.
    kind: CharacterKind = CharacterKind.human
    # Freeform anatomy / form description used for nonhuman
    # characters. Carries everything the structured human fields
    # encode (silhouette, materials, colours, distinguishing
    # features) plus any species-specific traits the human schema
    # can't express ("six-legged", "translucent", "smoke-bodied",
    # "cast iron and brass plating", "bioluminescent fronds along
    # the dorsal ridge"). Empty for human characters; the
    # storyteller is told to populate it richly when ``kind`` is
    # ``nonhuman``.
    physical_description: str = ""
    # Human-anatomy fields. Required-feeling for human characters
    # but allowed empty so the storyteller can leave them blank on
    # nonhuman characters where they don't apply (a smoke spirit
    # has no ``ethnicity``; a brass automaton has no ``bust``).
    # Renderer / prompt builder filters empty entries out so an
    # empty ``hair_color`` doesn't leave a stray ``" hair"`` tag
    # in the SDXL prompt.
    gender: str = ""
    # Narrative pronouns the storyteller addresses this character
    # with. Free text so non-binary / neopronoun / setting-specific
    # forms work ("she/her", "they/them", "xe/xir", or whatever
    # the player wants). Empty means "the storyteller derives
    # pronouns from gender or context" — same pre-pronoun
    # behaviour, so legacy saves keep narrating correctly.
    # Threaded into the storyteller / summarizer prompts when
    # populated so the LLM honours the player's choice instead
    # of guessing from gender each turn.
    pronouns: str = ""
    # Age in years. No upper cap — sci-fi / fantasy / mythic
    # premises legitimately call for ages in the thousands or
    # higher (immortals, archaeons, ghosts), and any cap the
    # storyteller picks above will land on a different ``too
    # large`` failure mode. ``ge=0`` stays — negative ages are a
    # parse bug, not a creative choice. Defaults to ``DEFAULT_AGE``
    # when omitted so a descriptor that drops the field materialises
    # a plausible adult instead of failing validation.
    age: int = Field(default=DEFAULT_AGE, ge=0)
    ethnicity: str = ""
    skin: str = ""
    hair_color: str = ""
    hairstyle: str = ""
    eye_color: str = ""
    build: str = ""
    bust: str = ""
    # Wardrobe / staging fields. Default to neutral fallbacks so a
    # descriptor that omits them (the storyteller occasionally drops
    # one on a just-introduced NPC) materialises cleanly instead of
    # failing validation and bouncing init back to the LLM.
    outfit: str = DEFAULT_OUTFIT
    pose: str = DEFAULT_POSE
    expression: str = DEFAULT_EXPRESSION
    # Visible physical effects that aren't part of the wardrobe or
    # baseline appearance — cuts, bruises, soot, blood, soaked
    # clothing, dishevelment, restraints, etc. Kept SEPARATE from
    # ``outfit`` so injuries don't bleed into the LLM's wardrobe
    # description and don't have to be re-typed in every outfit
    # change. Default empty string means "no special effects". 6
    # words max, same trim cap as outfit / pose / expression.
    effects: str = ""
    # Permanent skin markings: tattoos, scars, birthmarks, brands,
    # freckle patterns, ritual paint, cybernetic decals, etc. Kept
    # SEPARATE from ``effects`` (which is transient — cuts, soot,
    # dishevelment) because these persist across every outfit / pose /
    # expression change and shouldn't be re-typed. Surfaced and editable
    # in the Cast tab so the player can dial in a character's markings
    # and rerender. Default empty means "no distinctive markings".
    decals: str = ""
    facts: list[Fact] = Field(default_factory=list)
    images: list[CharacterImage] = Field(default_factory=list)
    seed: int = Field(ge=0, le=(1 << 64) - 1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Set when the character is gone from the live story — either
    # the LLM marked them as having died, or the player manually
    # dismissed them via the characters panel. Removed characters
    # are filtered out of on_stage and out of every prompt the
    # storyteller / summarizer sees, so they don't get re-summoned
    # in subsequent turns. The Session's undo stack snapshots Game
    # before each advance, so undoing past the death/dismissal
    # restores the prior ``removed=False`` state automatically.
    removed: bool = False
    # ``"dead"`` if the LLM narrated their death; ``"dismissed"``
    # if the player removed them from the characters panel; empty
    # string when ``removed`` is False.
    removed_reason: str = ""


_AGE_WORDS: tuple[str, ...] = (
    "eighteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
)


def age_band(age: int) -> str:
    """Map an exact age to a coarse age band used in image prompts.

    Lucidium intentionally keeps the player-supplied integer in storage
    (so edits round-trip exactly) and only blurs to a band at the moment
    a prompt is constructed (FR-005).

    Floors at 18: no image prompt sent to stable diffusion ever
    describes a minor. The character's stored ``age`` is left
    untouched — narration, dialogue, and player edits round-trip
    the original integer. Only this prompt-side helper raises it,
    so ages 0–19 render as ``"eighteen"`` in SDXL prompts (the
    older ``"teenage"`` word was dropped because SDXL's training
    distribution couples ``"teenage"`` with under-18 visual
    features even when the rest of the prompt names an adult age).
    The ``"child"`` band is unreachable — neither ``"child"`` nor
    ``"teenage"`` is present in ``_AGE_WORDS`` at all.
    """
    if age < 0:
        msg = f"age must be non-negative, got {age}"
        raise ValueError(msg)
    age = max(age, 18)
    if age >= 100:
        return _AGE_WORDS[-1]
    decade_index = age // 10 - 1  # 20 -> 1 -> "twenty"
    return _AGE_WORDS[decade_index]
