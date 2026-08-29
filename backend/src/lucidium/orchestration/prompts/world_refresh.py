"""World-state refresh prompt builder."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ...domain.dialog import DialogNode
from ...domain.environment import Environment
from ...domain.world import WorldState
from .common import SYSTEM_GAME_DIRECTOR, clamp_history

_RESPONSE_SHAPE = """{
  "summarizer_assessment": str,
  "direction_signal": "stay_focused" | "reintroduce_thread" | "none",
  "new_facts_by_character": {character_id: [{"id": str, "text": str, "confidence": "canon"|"inferred", "source_node_id": str|null}]},
  "pruned_fact_ids": [str],
  "current_stage_id": str | null,
  "revised_outline": [{"id": str, "title": str, "goal": str, "summary": str}] | null,
  "user_profile_additions": {"likes": [str], "dislikes": [str], "notes": [str]},
  "characters_to_offstage": [str]
}"""


def build(
    *,
    world: WorldState,
    history: Iterable[DialogNode],
    player_typed_quotes: list[str],
    user_likes: list[str] | None = None,
    user_dislikes: list[str] | None = None,
    user_notes: list[str] | None = None,
    environments: Mapping[str, Environment] | None = None,
    player_choices: list[str] | None = None,
    on_stage_ids: list[str] | None = None,
    player_character_id: str | None = None,
    character_names: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    rendered_history = clamp_history(
        history,
        char_budget=world.prompt_history_clamp_chars,
        environments=environments,
    )
    quotes_block = "\n".join(f"- {q}" for q in player_typed_quotes) or "(none)"
    if world.plot_outline:
        outline_block = "\n".join(
            f"  [{s.id}] {s.title} — goal: {s.goal}\n    {s.summary}" for s in world.plot_outline
        )
    else:
        outline_block = "  (no structured outline; legacy save)"
    current_stage_id = world.current_stage_id or "(unset)"
    profile_lines: list[str] = []
    for label, items in (
        ("likes", user_likes or []),
        ("dislikes", user_dislikes or []),
        ("notes", user_notes or []),
    ):
        if items:
            profile_lines.append(f"  {label}: " + "; ".join(items))
    profile_block = "\n".join(profile_lines) if profile_lines else "  (empty)"
    prior_assessment = (world.summarizer_assessment or "").strip() or "(empty)"
    choices_block = "\n".join(f"  - {c}" for c in (player_choices or [])) or "  (none yet)"
    # On-stage roster — used for job 5 (idle-character cleanup).
    # The player character is excluded from the listing so the
    # LLM doesn't accidentally propose removing the lens.
    if on_stage_ids:
        names = character_names or {}
        roster_lines: list[str] = []
        for cid in on_stage_ids:
            if cid == player_character_id:
                continue
            label = names.get(cid, cid)
            roster_lines.append(f"  - {cid} ({label})")
        on_stage_block = "\n".join(roster_lines) or "  (only the player is on stage)"
    else:
        on_stage_block = "  (no characters on stage)"
    user = (
        f"GAME: {world.game_name}\n"
        f"PLOT OUTLINE (the long-form plan — full text shown only to you, "
        f"the summarizer; the storyteller sees only the active stage):\n"
        f"{outline_block}\n"
        f"CURRENTLY-ACTIVE STAGE: {current_stage_id}\n"
        f"PRIOR SUMMARIZER ASSESSMENT (your previous synthesis — REWRITE this "
        f"to incorporate the new history, do NOT append. KEEP IT BOUNDED: cap "
        f"at ~400 words. Drop early-game details that no longer steer current "
        f"decisions; the assessment is a LIVING synopsis, not an append-only "
        f"log. If a thread closed, condense it to one sentence and move on. "
        f"If a character left and isn't returning, reduce them to a single "
        f"reference. Past structural beats roll up; current emotional / "
        f"informational state stays detailed):\n  {prior_assessment}\n\n"
        f"PLAYER PROFILE (cross-save, you maintain this — only ADD to it "
        f"when you have new evidence in the history below):\n"
        f"{profile_block}\n"
        f"PLAYER-TYPED INPUT (weight FAR HIGHER than provided choices when "
        f"inferring profile — these are the player's own words, not a menu "
        f"item the engine wrote):\n{quotes_block}\n\n"
        f"PLAYER CHOICES (the COMPLETE list of moments where the player "
        f"acted — every menu pick they selected, every line they typed. "
        f"This is the ONLY input you may use to infer profile tags in job "
        f"4 below. Engine-authored narrator beats / NPC dialogue / scene "
        f"setup do NOT count as player taste evidence — those are the "
        f"engine's choices, not the player's. ``picked:`` lines are menu "
        f"selections; ``typed:`` lines are free-text the player wrote "
        f"themselves. typed signals weight far higher than picked):\n"
        f"{choices_block}\n\n"
        f"CURRENTLY ON STAGE (besides the player — visible portraits "
        f"the storyteller can address):\n{on_stage_block}\n\n"
        f"RECENT HISTORY (use this for jobs 1-3 + 5 — facts, stage "
        f"selection, drift check, idle-cleanup. NOT for profile inference):\n"
        f"{rendered_history or '(empty)'}\n\n"
        "Refresh the world state. Five jobs:\n"
        "  1. Per-character facts: propose new canon/inferred facts and list "
        "any ids that have become redundant.\n"
        "  2. Stage selection: set ``current_stage_id`` to the id of the stage "
        "the actual story is in RIGHT NOW. Usually the same as the input — "
        "advance to the NEXT stage only when its goal is plainly met. Stay-put "
        "is the safe default.\n"
        "  3. Drift check: if the actual story has wandered far enough that the "
        "outline no longer fits, emit ``revised_outline`` with a replacement "
        "plan. Otherwise leave it ``null``. Revising the outline is expensive "
        "— only do it when the existing plan is genuinely broken.\n"
        "  4. User-profile inferences (CROSS-SAVE — these PERSIST forever, "
        "ACROSS DIFFERENT GAMES). PURPOSE: the storyteller reads this "
        "profile on every turn to decide what KIND of scene and "
        "character to author next, so it can surface things this "
        "specific player will find INTERESTING.\n"
        "     SOURCING — STRICT: every tag you propose MUST be supported "
        "by an entry in the PLAYER CHOICES block above. Things the "
        "ENGINE chose (settings, weather, NPCs introduced, dialogue "
        "the storyteller wrote, plot events the player didn't select) "
        "are NOT player-taste evidence — the player didn't choose them, "
        "they happened. Likewise, the existence of a HAUNTED HOUSE in "
        "the scene is NOT evidence the player likes haunted houses; "
        "evidence would be ``picked: investigate the haunted house`` "
        "WHEN ANOTHER OPTION WAS AVAILABLE. If you can't point to a "
        "specific PLAYER CHOICES entry that supports a tag, do not add "
        "the tag. A good tag is one that a different storyteller, on "
        "a different save, would actually use to steer toward content "
        "the player enjoys. Tag what the PLAYER finds INTERESTING, "
        "not what they did. Two priorities, in order:\n"
        "       (i)  WHAT KINDS OF SCENES they want to spend time in — "
        "quiet investigation, charged confrontations, tender domestic "
        "interludes, action set-pieces, court intrigue, eerie liminal "
        "spaces, etc.\n"
        "       (ii) WHAT KINDS OF CHARACTERS they engage with — morally "
        "grey mentors, antagonists they can argue with, soft-spoken "
        "competents, charismatic rogues, hostile witnesses, etc.\n"
        "     DO NOT log minutiae about HOW the player does things — "
        "'always reads room descriptions before talking', 'tends to ask "
        "follow-up questions', 'opens doors before windows', 'reads notes "
        "twice'. Procedural habits are noise; they don't tell the next "
        "save what kind of STORY this player wants. Generic preference "
        "tags ('likes interesting characters', 'enjoys mystery') are also "
        "noise — every player likes those in the abstract; only log what "
        "DIFFERENTIATES this player.\n"
        "     EVIDENCE WEIGHTING: a single PLAYER-TYPED quote is worth "
        "many menu picks — when the player writes their own line, what "
        "they CHOSE TO SAY is direct signal about their taste. Menu picks "
        "from engine-provided options reveal less because every choice "
        "was on offer. Wait for at least one strong typed-input signal "
        "or 3+ consistent menu picks before adding any tag.\n"
        "     FORMAT: tags are 2-8 words, lowercase, label-style "
        "(``likes: slow-burn investigation``, "
        "``likes: morally grey mentors``, "
        "``dislikes: tonal whiplash``, "
        "``notes: gravitates to scenes of solitude``). Never repeat a "
        "tag already in the profile above. Most refresh calls should "
        "add nothing.\n"
        "  5. Departed-character cleanup: walk through EACH id in "
        "the CURRENTLY ON STAGE roster above and ask: 'is this "
        "character still in the same place as the player at the "
        "very end of the recent history?' If yes → leave them on "
        "stage. If no → add them to ``characters_to_offstage``. "
        "Multiple ids in one call is NORMAL whenever the player "
        "has moved scenes — every character who was at the prior "
        "location and didn't come along should be listed.\n"
        "     Two narrow triggers, both require textual evidence "
        "in the recent history:\n"
        "       (a) the character has explicitly left — they said "
        "goodbye, walked out, hung up, dismissed themselves, were "
        "sent away, etc. Narration shows them exiting the location.\n"
        "       (b) the player has left them behind — the player "
        "moved to a different location / scene / time and the "
        "character did not come along. EVERY character who was at "
        "the prior location and is not narrated as accompanying "
        "the player into the new location belongs in this list. "
        "Don't stop at the first one you spot — check every "
        "on-stage id against the FINAL location of the history.\n"
        "     This job is NOT about idle pruning. A character who "
        "is simply quiet for several beats but is still in the "
        "same room with the player STAYS ON STAGE. Do not remove "
        "characters because they have not spoken recently — only "
        "because the narrative has explicitly moved past them. "
        "When the player hasn't changed scene and no one has "
        "explicitly exited, empty list is the right answer. "
        "NEVER emit the player character's id (the player is the "
        "lens, not a stage actor).\n"
        "Return JSON only.\n" + _RESPONSE_SHAPE
    )
    return [
        {"role": "system", "content": SYSTEM_GAME_DIRECTOR},
        {"role": "user", "content": user},
    ]
