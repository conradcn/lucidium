"""Facts-consolidation prompt: condense a character's accumulated
facts list when it has grown past the soft threshold.

The summarizer adds facts every turn but rarely prunes; on a long
run a single character's list balloons past 20+ entries that often
overlap or duplicate. This pass is a separate cleanup job — fired
when ``facts_need_consolidation`` flags a character — that asks
the LLM to merge near-duplicates and drop low-signal entries while
preserving canonical truths. The output replaces the character's
``facts`` list; the calling handler reapplies it onto the live
``Game``.
"""

from __future__ import annotations

from .common import system_prompt

_RESPONSE_SHAPE = """{
  "facts_by_character": {
    "<character_id>": [
      {"text": str, "confidence": "canon"|"inferred"},
      ...
    ],
    ...
  }
}"""

_RULES = (
    "FACTS-CONSOLIDATION RULES: each character below has accumulated "
    "many facts. Your job: produce a tighter list per character that "
    "preserves the same information without the bloat. "
    "(1) MERGE near-duplicates — 'has a sister', 'mentioned a "
    "younger sister' → one entry like 'has a younger sister'. "
    "(2) DROP redundant entries — anything implied by another "
    "fact, anything restating the character's outfit / pose / "
    "expression (those live on the character object itself; the "
    "facts list shouldn't echo them). "
    "(3) PRESERVE canon facts that are concrete and load-bearing — "
    "names mentioned, family relationships, oaths, secrets, fears, "
    "promises made. The storyteller reads these to keep continuity. "
    "(4) DROP ephemeral observations — 'looked tired this morning', "
    "'was annoyed by the dog', 'mentioned the weather'. These "
    "describe one beat, not the character. "
    "(5) ``confidence``: 'canon' for facts the player or another "
    "character stated explicitly; 'inferred' for everything else. "
    "(6) Aim for AT MOST 12 facts per character. The hard cap is "
    "higher; this is a soft target so the next few summarizer "
    "additions don't immediately re-trigger consolidation. "
    "(7) Keep entries SHORT — under ~80 chars, lowercase first "
    "letter unless it's a proper noun, no trailing punctuation. "
    "Return strictly-valid JSON in the response shape — no prose, "
    "no markdown fences."
)


def build(
    *,
    facts_by_character: dict[str, list[tuple[str, str]]],
    character_briefs: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Compose the consolidation prompt.

    ``facts_by_character`` maps id → list of ``(text, confidence)``
    pairs (the existing facts to be condensed). ``character_briefs``
    is an optional map from id → "name — description" so the LLM
    has context for which character is which when the id alone
    isn't self-explanatory.
    """
    blocks: list[str] = []
    for cid, facts in facts_by_character.items():
        brief = ""
        if character_briefs and cid in character_briefs:
            brief = f" ({character_briefs[cid]})"
        rendered = "\n".join(f"    - [{conf}] {text}" for text, conf in facts) or "    (no facts)"
        blocks.append(f"  {cid}{brief}:\n{rendered}")
    user = (
        "Condense the per-character facts below. Each character has "
        "more than the consolidation threshold; merge overlapping "
        "entries, drop the noise, keep what matters for continuity.\n\n"
        f"{chr(10).join(blocks)}\n\n"
        f"{_RULES}\n\n"
        f"Return JSON in this shape:\n{_RESPONSE_SHAPE}"
    )
    return [
        {"role": "system", "content": system_prompt(mature_content=False)},
        {"role": "user", "content": user},
    ]
