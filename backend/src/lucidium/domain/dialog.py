"""Dialog tree: nodes, options, character changes, premise hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .character import (
    DEFAULT_AGE,
    DEFAULT_EXPRESSION,
    DEFAULT_OUTFIT,
    DEFAULT_POSE,
    CharacterKind,
)
from .ids import new_id


class DialogNodeState(StrEnum):
    speculative = "speculative"
    pending_text = "pending_text"
    ready = "ready"
    committed = "committed"
    invalidated = "invalidated"


class CharacterAttributeField(StrEnum):
    description = "description"
    outfit = "outfit"
    pose = "pose"
    expression = "expression"
    effects = "effects"
    # Permanent skin markings (tattoos, scars, birthmarks). In the
    # enum so a cast-tab edit is stamped onto the beat's
    # character_changes and survives replay / retcon / summarizer
    # re-derivation — without it, edits applied live but a later
    # ``_apply_node`` pass could revert them. Mirrors ``effects``.
    decals = "decals"
    name = "name"
    hair_color = "hair_color"
    hairstyle = "hairstyle"
    eye_color = "eye_color"
    skin = "skin"
    build = "build"
    bust = "bust"
    ethnicity = "ethnicity"
    gender = "gender"
    # Mid-game human ↔ nonhuman transitions. Both fields are
    # editable from the cast UI directly via
    # ``c2s/edit/character`` (which doesn't gate on this enum),
    # but they ALSO need to flow through the LLM-driven paths:
    #   * retcon — "this character was always a spirit" rewrites
    #     past narration AND flips their kind in one pass.
    #   * character_change on a beat — a transformation moment
    #     can be encoded as a per-beat mutation that the
    #     storyteller emits, e.g. a person revealing themselves
    #     as something else mid-scene. Without these enum
    #     entries, both paths silently dropped attempts to
    #     mutate kind / physical_description.
    kind = "kind"
    physical_description = "physical_description"
    # Pronouns are an attribute the LLM can update mid-narration
    # (a character revealing they prefer different pronouns) or
    # via retcon ("this character has been they/them all along").
    # Cast-tab edits go through ``c2s/edit/character`` which
    # bypasses this enum, so the field is editable from the UI
    # regardless of this entry — the entry exists to enable the
    # narrative + retcon paths.
    pronouns = "pronouns"


class CharacterChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_id: str
    field: CharacterAttributeField
    new_value: str


class NewCharacterDescriptor(BaseModel):
    """Full descriptor for a character introduced by a dialog beat.

    The LLM emits this when a beat references a character ID that
    doesn't yet exist on the stage. The handler turns each descriptor
    into a full ``Character`` (assigning a seed) before applying the
    beat, so ``entering_character_ids`` always resolves cleanly.

    The schema mirrors ``Character`` — ``kind`` branches the
    rendering pipeline (``human`` → structured anatomy fields;
    ``nonhuman`` → freeform ``physical_description`` with empty
    anatomy fields). All human-anatomy fields default to empty
    string so the LLM can omit them on nonhuman entries; the
    portrait builder filters empties out so we don't emit stray
    ``" hair"`` or ``"thirty-year-old "`` tags.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str
    # Mirrors ``Character.kind`` — see character.py for the
    # rendering-pipeline split.
    kind: CharacterKind = CharacterKind.human
    # Mirrors ``Character.physical_description`` — used only when
    # ``kind="nonhuman"`` (freeform 6-15 word anatomy). Empty for
    # human characters.
    physical_description: str = ""
    gender: str = ""
    # Mirrors ``Character.pronouns`` — free-text pronouns the
    # storyteller uses to address this NPC. Empty falls back to
    # the storyteller deriving pronouns from gender / context.
    pronouns: str = ""
    age: int = Field(default=DEFAULT_AGE, ge=0)
    ethnicity: str = ""
    skin: str = ""
    hair_color: str = ""
    hairstyle: str = ""
    eye_color: str = ""
    build: str = ""
    bust: str = ""
    # Wardrobe / staging. Default to neutral fallbacks so a beat that
    # introduces a character but omits one of these still parses — a
    # single dropped field used to fail the whole world_init /
    # dialog payload and force an LLM retry. See character.py.
    outfit: str = DEFAULT_OUTFIT
    pose: str = DEFAULT_POSE
    expression: str = DEFAULT_EXPRESSION
    # Visible physical effects (cuts, bruises, soot, soaked clothes,
    # restraints). Stays separate from outfit so wardrobe stays
    # stable when an injury appears.
    effects: str = ""
    # Permanent skin markings (tattoos, scars, birthmarks, brands).
    # Mirrors ``Character.decals`` — persists across outfit / pose
    # changes. Empty means no distinctive markings.
    decals: str = ""


class DialogOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    text: str


class GenerationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    prompt_hash: str | None = None
    seed_parameters: dict[str, str | int | float] = Field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0


class DialogNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_id)
    parent_id: str | None = None
    chosen_option_id: str | None = None
    speaker_id: str | None = None
    text: str | None = None
    options: list[DialogOption] = Field(default_factory=list)
    entering_character_ids: list[str] = Field(default_factory=list)
    leaving_character_ids: list[str] = Field(default_factory=list)
    # New characters this beat introduces. Their ids appear in
    # ``entering_character_ids``; the handler creates Character entries
    # from these descriptors before computing the on-stage roster.
    new_characters: list[NewCharacterDescriptor] = Field(default_factory=list)
    location_id: str | None = None
    location_prompt: str | None = None
    # Short lighting descriptor for THIS beat's location (e.g.
    # "candlelit gloom", "harsh midday sun"). Carried on the beat
    # because lighting often changes within a scene — moving from
    # the foyer to the kitchen, dawn breaking, the lamps going out.
    # Empty string falls through to the parent environment's stored
    # ``lighting`` value at prompt-build time.
    location_lighting: str = ""
    character_changes: list[CharacterChange] = Field(default_factory=list)
    state: DialogNodeState = DialogNodeState.speculative
    premise_hash: str
    generation_metadata: GenerationMetadata = Field(default_factory=GenerationMetadata)


class DialogTree(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: dict[str, DialogNode] = Field(default_factory=dict)
    root_id: str | None = None
    committed_path: list[str] = Field(default_factory=list)


def world_snapshot_vector(
    *,
    on_stage_character_ids: Iterable[str],
    character_attributes: Mapping[str, Mapping[str, str]],
    world_fields: Mapping[str, str],
) -> list[object]:
    """Project the parts of state that influence dialog generation.

    Stable, deterministic, and small — this is the input the premise hash
    canonicalizes. Order is enforced (sorted by id / key) so the hash is
    stable across runs.
    """
    sorted_ids = sorted(on_stage_character_ids)
    char_block = [[cid, sorted(character_attributes.get(cid, {}).items())] for cid in sorted_ids]
    world_block = sorted(world_fields.items())
    return [sorted_ids, char_block, world_block]


def premise_hash(
    *,
    parent_id: str | None,
    chosen_option_id: str | None,
    snapshot_vector: list[object],
) -> str:
    payload = {
        "parent_id": parent_id,
        "chosen_option_id": chosen_option_id,
        "snapshot": snapshot_vector,
    }
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
