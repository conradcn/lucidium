"""Profile-consolidation prompt: shrink an oversized cross-save
``UserProfile`` and strip storyline-specific entries."""

from __future__ import annotations

from .common import system_prompt

_RESPONSE_SHAPE = """{
  "likes":    [str, ...],
  "dislikes": [str, ...],
  "notes":    [str, ...]
}"""

_RULES = (
    "PROFILE CONSOLIDATION RULES: this profile is CROSS-SAVE — it "
    "describes the player's TASTE, not the current story. Your job: "
    "(1) MERGE entries that are saying the same thing in different "
    "words ('slow burns', 'unhurried pacing', 'lets scenes breathe' "
    "→ one entry like 'pacing: slow burns'). "
    "(2) DROP anything tied to a specific storyline — character "
    "names, plot beats, save-specific stakes, settings, locations, "
    "antagonists. If an entry only makes sense inside ONE story, "
    "delete it. Examples to drop: 'likes Mira', 'wary of the "
    "harbour cult', 'enjoyed the tavern ambush'. Examples to keep: "
    "'likes morally grey allies', 'wary of cult-style antagonists', "
    "'enjoys ambush set-pieces'. "
    "(3) DROP procedural-habit minutiae about HOW the player does "
    "things — 'reads notes twice', 'always asks follow-up "
    "questions', 'opens doors before windows', 'examines room "
    "before talking'. These are play-style noise that don't tell "
    "the next save what KIND of story this player wants. Keep "
    "entries that describe what KINDS OF SCENES or CHARACTERS the "
    "player gravitates to. "
    "(4) DROP generic preferences that almost any player would "
    "share — 'likes interesting characters', 'enjoys mystery', "
    "'wants engaging story'. Only keep tags that DIFFERENTIATE this "
    "player from the median. "
    "(5) Keep entries SHORT — under ~60 chars, lowercase, no "
    "trailing punctuation. "
    "(6) Return AT MOST 8 entries per bucket. Pick the most "
    "informative; the engine fills the bucket back up over time. "
    "Return strictly-valid JSON in the response shape — no prose, "
    "no Markdown fence."
)


def build(
    *,
    likes: list[str],
    dislikes: list[str],
    notes: list[str],
) -> list[dict[str, str]]:
    def block(label: str, items: list[str]) -> str:
        if not items:
            return f"  {label}: (empty)"
        return f"  {label}:\n" + "\n".join(f"    - {tag}" for tag in items)

    user = (
        "Consolidate the cross-save player profile below. The "
        "buckets have grown long enough that they're crowding the "
        "main storyteller prompt; merge near-duplicates and drop "
        "anything that's actually about a SPECIFIC storyline rather "
        "than a TASTE.\n\n"
        f"{block('LIKES', likes)}\n"
        f"{block('DISLIKES', dislikes)}\n"
        f"{block('NOTES', notes)}\n\n"
        f"{_RULES}\n\n"
        f"Return JSON in this shape:\n{_RESPONSE_SHAPE}"
    )
    return [
        {"role": "system", "content": system_prompt(mature_content=False)},
        {"role": "user", "content": user},
    ]
