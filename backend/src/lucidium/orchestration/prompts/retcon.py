"""Retcon prompts: rewrite committed history globally to match player instructions.

Retcon runs in BATCHES rather than one big call. The original
single-call shape worked on frontier models with 8K+ output windows
but truncated mid-JSON on the smaller / cheaper models the engine
actually ships with — leaving the tail of history un-rewritten.
Batching keeps each call's response small, parses cleanly, and lets
the orchestrator fan the beat batches out in parallel.

Three call shapes:

  * ``build_beat_batch`` — rewrite one slice of committed beats.
    Keeps the JSON output bounded by ``RETCON_BATCH_SIZE`` so even
    a 1024-token cap holds.
  * ``build_character_updates`` — emit ``character_updates`` only.
    Single small call regardless of history length.
  * ``build_world_updates`` — emit ``summarizer_assessment`` /
    ``overall_plot_direction`` replacements only when the retcon
    invalidates them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ...domain.character import Character
from ...domain.dialog import DialogNode
from ...domain.world import WorldState
from .common import render_character_brief, render_character_full, system_prompt

# Batch size for ``build_beat_batch``. 4 beats × ~300 chars of
# rewritten text each ≈ ~600-800 output tokens including JSON
# scaffolding — fits well under the default 1024-token cap that
# non-frontier models often impose.
RETCON_BATCH_SIZE: int = 4


_BEAT_RESPONSE_SHAPE = """{
  "rewritten_beats": [
    {"node_id": str, "text": str, "speaker_id": str|null}
  ]
}"""

_CHARACTER_RESPONSE_SHAPE = """{
  "character_updates": [
    {"character_id": str, "field": str, "new_value": str}
  ]
}"""

_WORLD_RESPONSE_SHAPE = """{
  "world_updates": {
    "summarizer_assessment": str,
    "overall_plot_direction": str
  }
}"""

_BEAT_RULES = (
    "RETCON RULES — BEAT BATCH: the player has issued a GLOBAL "
    "retcon. You are rewriting ONE batch of committed beats so they "
    "fit the new reality. Walk every beat in the BATCH BEATS block "
    "below and decide whether it needs rewriting. "
    "COMPLETENESS IS MANDATORY WITHIN THIS BATCH: when the retcon "
    "instruction touches tone, setting, weather, lighting, "
    "time-of-day, character wardrobe, character appearance, or any "
    "other quality this batch's beats reference, you MUST rewrite "
    "EVERY beat in this batch that mentions or implies that "
    "quality. If unsure whether a beat needs rewriting, err on the "
    "side of rewriting it. Beats that are GENUINELY untouched "
    "(the instruction has no bearing on what that beat contains) "
    "may be omitted from the rewrite list. Preserve the original "
    "prose voice and second-person perspective in rewritten beats. "
    "DO NOT include beats from outside this batch — only the ids "
    "listed in BATCH BEATS are valid. Return strictly-valid JSON "
    "matching the response shape — no prose, no Markdown fences."
)

_CHARACTER_RULES = (
    "RETCON RULES — CHARACTER UPDATES: the player has issued a "
    "global retcon. Walk every character in the CHARACTERS block "
    "and emit a ``character_updates`` entry for each MUTABLE "
    "attribute that needs to change to fit the new reality. "
    "Attributes that are already consistent should NOT be returned. "
    "Return strictly-valid JSON matching the response shape — no "
    "prose, no Markdown fences."
)

_WORLD_RULES = (
    "RETCON RULES — WORLD UPDATES: the player has issued a global "
    "retcon. If it materially shifts what ``summarizer_assessment`` "
    "or ``overall_plot_direction`` should now say, emit replacement "
    "values. If the existing values still hold, return an empty "
    "``world_updates`` object. Return strictly-valid JSON matching "
    "the response shape — no prose, no Markdown fences."
)

_CHARACTER_FIELDS = (
    "CHARACTER ATTRIBUTE FIELDS (use these EXACT names): "
    "description, outfit, pose, expression, name, hair_color, "
    "hairstyle, eye_color, skin, build, bust, ethnicity, gender."
)


def chunk_beats(
    committed_history: Sequence[DialogNode],
    *,
    batch_size: int = RETCON_BATCH_SIZE,
) -> list[list[DialogNode]]:
    """Slice committed beats into ``batch_size``-sized batches in
    committed order. Last batch may be shorter; empty input yields
    an empty list (the handler skips the LLM round-trip entirely)."""
    if not committed_history:
        return []
    return [
        list(committed_history[i : i + batch_size])
        for i in range(0, len(committed_history), batch_size)
    ]


def _format_batch_block(batch: Sequence[DialogNode]) -> str:
    lines: list[str] = []
    for node in batch:
        speaker = node.speaker_id or "narrator"
        text = node.text or ""
        lines.append(f"  [{node.id}] {speaker}: {text}")
    return "\n".join(lines) or "  (empty batch)"


def _format_brief_history(committed_history: Sequence[DialogNode]) -> str:
    """Render the FULL committed history as one-line briefs (no node
    ids — the LLM cannot rewrite these from outside the batch). Used
    as context inside batch prompts so the model knows the arc the
    rewrites should land inside, but can't accidentally emit ids
    that aren't in scope."""
    if not committed_history:
        return "  (no committed history)"
    lines: list[str] = []
    for i, node in enumerate(committed_history, start=1):
        speaker = node.speaker_id or "narrator"
        text = (node.text or "").replace("\n", " ").strip()
        if len(text) > 200:
            text = text[:200].rstrip() + "…"
        lines.append(f"  {i}. {speaker}: {text}")
    return "\n".join(lines)


def build_beat_batch(
    *,
    world: WorldState,
    instructions: str,
    committed_history: Sequence[DialogNode],
    batch: Sequence[DialogNode],
    characters: dict[str, Character],
) -> list[dict[str, str]]:
    """Build the prompt for ONE batch of beat rewrites.

    ``committed_history`` is the full committed path so the LLM has
    arc context — but it can ONLY emit rewrites for ids in
    ``batch``. The full-history block is rendered in one-line brief
    form (no ids) to make this constraint physically enforceable.
    """
    char_block = (
        "\n".join(render_character_brief(ch) for ch in characters.values()) or "(no characters)"
    )

    user = (
        f"GAME: {world.game_name}\n"
        f"SETTING: {world.setting} | GENRE: {world.genre} | "
        f"VISUAL STYLE: {world.visual_style}\n\n"
        f"PLAYER RETCON INSTRUCTIONS:\n  {instructions.strip()}\n\n"
        f"CHARACTERS (for reference only — do NOT emit "
        f"character_updates here):\n{char_block}\n\n"
        f"FULL ARC (one-line summary of every committed beat — for "
        f"context, NOT for rewriting):\n"
        f"{_format_brief_history(committed_history)}\n\n"
        f"BATCH BEATS (these are the ONLY ids you may emit "
        f"rewrites for — anything else is dropped):\n"
        f"{_format_batch_block(batch)}\n\n"
        f"{_BEAT_RULES}\n\n"
        f"Return JSON in this shape:\n{_BEAT_RESPONSE_SHAPE}"
    )
    return [
        {"role": "system", "content": system_prompt(mature_content=False)},
        {"role": "user", "content": user},
    ]


def build_character_updates(
    *,
    world: WorldState,
    instructions: str,
    characters: dict[str, Character],
) -> list[dict[str, str]]:
    """Build the prompt for the character_updates pass. Output is a
    short JSON object, well under any reasonable max_tokens cap."""
    char_block = (
        "\n\n".join(render_character_full(ch) for ch in characters.values()) or "(no characters)"
    )

    user = (
        f"GAME: {world.game_name}\n"
        f"SETTING: {world.setting} | GENRE: {world.genre} | "
        f"VISUAL STYLE: {world.visual_style}\n\n"
        f"PLAYER RETCON INSTRUCTIONS:\n  {instructions.strip()}\n\n"
        f"CHARACTERS (CANON ATTRIBUTES):\n{char_block}\n\n"
        f"{_CHARACTER_RULES}\n\n"
        f"{_CHARACTER_FIELDS}\n\n"
        f"Return JSON in this shape:\n{_CHARACTER_RESPONSE_SHAPE}"
    )
    return [
        {"role": "system", "content": system_prompt(mature_content=False)},
        {"role": "user", "content": user},
    ]


def build_world_updates(
    *,
    world: WorldState,
    instructions: str,
) -> list[dict[str, str]]:
    """Build the prompt for the world_updates pass. Tiny output — at
    most two short string replacements."""
    user = (
        f"GAME: {world.game_name}\n"
        f"SETTING: {world.setting} | GENRE: {world.genre} | "
        f"VISUAL STYLE: {world.visual_style}\n\n"
        f"CURRENT summarizer_assessment:\n  {world.summarizer_assessment or '(empty)'}\n\n"
        f"CURRENT overall_plot_direction:\n  {world.overall_plot_direction or '(empty)'}\n\n"
        f"PLAYER RETCON INSTRUCTIONS:\n  {instructions.strip()}\n\n"
        f"{_WORLD_RULES}\n\n"
        f"Return JSON in this shape:\n{_WORLD_RESPONSE_SHAPE}"
    )
    return [
        {"role": "system", "content": system_prompt(mature_content=False)},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Backward-compat: the old single-shot ``build`` is preserved for
# callers (and tests) that still want the all-in-one prompt. The
# handler now uses the batched builders above; this stub is kept
# only for any out-of-tree consumers and may be removed later.
# ---------------------------------------------------------------------------

_LEGACY_RESPONSE_SHAPE = """{
  "rewritten_beats": [
    {"node_id": str, "text": str, "speaker_id": str|null}
  ],
  "character_updates": [
    {"character_id": str, "field": str, "new_value": str}
  ],
  "world_updates": {
    "summarizer_assessment": str,
    "overall_plot_direction": str
  }
}"""


def build(
    *,
    world: WorldState,
    instructions: str,
    committed_history: Iterable[DialogNode],
    characters: dict[str, Character],
) -> list[dict[str, str]]:
    """Legacy single-call retcon prompt. Prefer the batched builders.

    Retained so nothing that depended on the old one-call shape
    silently breaks; the handler no longer reaches it.
    """
    char_block = (
        "\n\n".join(render_character_full(ch) for ch in characters.values()) or "(no characters)"
    )
    history = list(committed_history)
    history_block = _format_batch_block(history)

    user = (
        f"GAME: {world.game_name}\n"
        f"SETTING: {world.setting} | GENRE: {world.genre} | "
        f"VISUAL STYLE: {world.visual_style}\n\n"
        f"PLAYER RETCON INSTRUCTIONS:\n  {instructions.strip()}\n\n"
        f"CHARACTERS (CANON ATTRIBUTES):\n{char_block}\n\n"
        f"COMMITTED HISTORY:\n{history_block}\n\n"
        f"{_BEAT_RULES}\n\n{_CHARACTER_RULES}\n\n{_WORLD_RULES}\n\n"
        f"{_CHARACTER_FIELDS}\n\n"
        f"Return JSON in this shape:\n{_LEGACY_RESPONSE_SHAPE}"
    )
    return [
        {"role": "system", "content": system_prompt(mature_content=False)},
        {"role": "user", "content": user},
    ]
