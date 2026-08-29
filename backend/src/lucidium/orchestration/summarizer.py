"""Summarizer: maintains character facts and world steering signals."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..domain.character import Fact, FactConfidence
from ..domain.dialog import DialogNode
from ..domain.settings import UserProfile
from ..domain.world import DirectionSignal, PlotStage
from .responses import LlmSummaryResult


def collect_player_typed_quotes(history: Iterable[DialogNode], *, limit: int = 5) -> list[str]:
    """Pull the most recent player-typed quotes from history.

    Player-typed beats are recognizable because they have ``speaker_id is None``
    and ``chosen_option_id is None`` while sitting under a parent — i.e., the
    only way they can exist is via a free-text submission. (Engine-emitted
    narrator beats always carry generation_metadata or aren't free-text.)
    """
    typed: list[str] = []
    for node in reversed(list(history)):
        if (
            node.speaker_id is None
            and node.chosen_option_id is None
            and node.parent_id is not None
            and node.text
            and not node.generation_metadata.model
        ):
            typed.append(node.text)
            if len(typed) >= limit:
                break
    return list(reversed(typed))


def collect_player_choices(
    history: Iterable[DialogNode],
    *,
    limit: int = 30,
) -> list[str]:
    """Pull every PLAYER-DRIVEN moment from the committed history,
    in order, as readable strings. Two kinds of moments count:

      * Menu picks — the player chose option text X from an
        engine-provided list. Identified by ``chosen_option_id``
        being set on a node; the option text is fetched from the
        parent's ``options`` list.
      * Typed inputs — the player typed their own free-text. Same
        recognition rule as ``collect_player_typed_quotes``.

    Engine-authored narrator beats and NPC dialogue are EXCLUDED.
    The summarizer's profile-inference step uses this filtered
    list instead of the whole history so it doesn't infer a
    "likes haunted houses" tag from the engine deciding the
    setting was a haunted house — only the player's own choices
    or typed lines actually reveal taste.

    Returns the most-recent ``limit`` choices in chronological
    order. Each element is one of:
        ``picked: <option text>``
        ``typed: <free text>``
    Lets the LLM see WHICH kind of signal each line is — typed
    quotes weight far higher than menu picks because every menu
    pick was on offer, while typed text is the player's own
    initiative.
    """
    nodes = list(history)
    by_id = {n.id: n for n in nodes}
    out: list[str] = []
    for node in nodes:
        if node.chosen_option_id and node.parent_id:
            parent = by_id.get(node.parent_id)
            if parent is None:
                continue
            for opt in parent.options:
                if opt.id == node.chosen_option_id:
                    text = (opt.text or "").strip()
                    if text:
                        out.append(f"picked: {text}")
                    break
            continue
        if (
            node.speaker_id is None
            and node.chosen_option_id is None
            and node.parent_id is not None
            and node.text
            and not node.generation_metadata.model
        ):
            text = node.text.strip()
            if text:
                out.append(f"typed: {text}")
    if len(out) > limit:
        out = out[-limit:]
    return out


@dataclass
class SummaryApplication:
    """Aggregate of every state mutation the summarizer wants to
    apply. Returned as a single struct so the caller does atomic
    updates against both ``WorldState`` and the cross-save
    ``Settings.user_profile``."""

    character_facts: dict[str, list[Fact]]
    summarizer_assessment: str
    direction_signal: DirectionSignal
    # New stage pointer if the LLM advanced (None means "no change").
    current_stage_id: str | None = None
    # Replacement outline if drift was detected (None means "keep
    # the existing outline").
    revised_outline: list[PlotStage] | None = None
    # Cross-save profile after merging the summary's additions. The
    # caller writes this back to ``Settings`` and persists. Will
    # equal the input profile (no mutation) when the LLM proposed
    # nothing new.
    user_profile: UserProfile | None = None
    # Character ids the summarizer wants removed from ``Game.on_stage``.
    # Validated against the live on-stage list so a hallucinated id
    # or a player-character id is silently dropped before the caller
    # touches game state. Empty in most calls.
    characters_to_offstage: list[str] = field(default_factory=list)


PROFILE_BUCKET_CAP = 10
# Soft threshold that triggers the facts consolidator. Lower than
# the hard ``FACTS_PER_CHARACTER_CAP`` so the cleanup job runs
# BEFORE the cap kicks in and silently drops oldest entries —
# consolidation lets the LLM merge near-duplicates and surface the
# most informative, leaving the hard cap as a defensive backstop.
FACTS_CONSOLIDATION_THRESHOLD = 20
# Hard cap on facts per character. The summarizer adds new facts
# every turn but rarely prunes; without a cap, the storyteller
# prompt's per-character render grows linearly with turn count,
# which is the dominant driver of late-session prompt bloat /
# slower turns. When ``apply_summary`` walks past the cap, the
# OLDEST facts are dropped (canon facts come first in the list,
# so the survivors are typically the most recently learned). A
# regression test asserts this cap holds across many turns.
FACTS_PER_CHARACTER_CAP = 30
# Hard cap on the summarizer-assessment string length. Free-text
# field that the LLM happily fills with paragraphs; clamped here
# so the storyteller prompt's "SUMMARIZER ASSESSMENT" line stays
# bounded. Past the cap, the START of the string is trimmed (the
# tail is the most recent reasoning).
SUMMARIZER_ASSESSMENT_CAP = 1200


def _merge_profile_additions(
    *, current: UserProfile, additions: dict[str, list[str]]
) -> UserProfile:
    """Append summarizer additions to the profile.

    Player-owned buckets (``likes`` / ``dislikes`` / ``notes``) are
    sacrosanct — they pass through untouched. Additions go into the
    parallel ``summarizer_*`` buckets, deduplicated case-insensitively
    against BOTH the player-owned bucket AND the existing summarizer
    bucket so the summarizer doesn't echo something the player just
    typed by hand.

    Strength bookkeeping: each tag carries a 0–``TRAIT_STRENGTH_CAP``
    float in the matching ``summarizer_*_scores`` dict. First
    sighting starts at ``INITIAL_TRAIT_STRENGTH`` (0.1); each
    subsequent re-inference adds ``TRAIT_STRENGTH_INCREMENT``
    (0.1), capped at the strength cap (2.0). A tag must reach
    ``TRAIT_SURFACE_THRESHOLD`` (1.0) before it's visible to
    storyteller / summarizer prompts or to the Settings UI —
    so a brand-new inference (0.1) is invisible until it's
    been independently re-inferred ~9 more times.
    Player-bucket / dismissed-list tags flow through unchanged
    and are never scored.
    """
    from ..domain.settings import (
        INITIAL_TRAIT_STRENGTH,
        TRAIT_STRENGTH_CAP,
        TRAIT_STRENGTH_INCREMENT,
    )

    def merged(
        player: list[str],
        summarizer: list[str],
        dismissed: list[str],
        scores: dict[str, float],
        add: list[str],
    ) -> tuple[list[str], dict[str, float]]:
        out = list(summarizer)
        new_scores = {k: float(v) for k, v in scores.items()}
        existing = {tag.strip().casefold() for tag in summarizer if tag.strip()}
        # ``dismissed`` joins the dedup set so anything the player
        # explicitly removed from this bucket via the Settings UI
        # never re-appears, even if the LLM re-detects the pattern.
        # Player-bucket entries also block summarizer additions —
        # the LLM doesn't echo a tag the player typed by hand.
        rejected = {tag.strip().casefold() for tag in (*player, *dismissed) if tag.strip()}
        for raw in add:
            tag = (raw or "").strip()
            if not tag:
                continue
            key = tag.casefold()
            if key in rejected:
                continue
            if key in existing:
                # Same tag inferred again — bump strength,
                # don't append a duplicate.
                current_strength = new_scores.get(key, INITIAL_TRAIT_STRENGTH)
                new_scores[key] = min(
                    current_strength + TRAIT_STRENGTH_INCREMENT,
                    TRAIT_STRENGTH_CAP,
                )
                continue
            # New tag — append to list, seed strength at the
            # initial floor (well below surface threshold). The
            # tag is invisible to LLMs and the UI until enough
            # subsequent inferences accumulate.
            existing.add(key)
            out.append(tag)
            new_scores[key] = INITIAL_TRAIT_STRENGTH
        # Drop scores for tags no longer in the list (defensive —
        # the merge above never removes, but a future edit path
        # might delete a tag, and a stale score would confuse
        # the strength lookup).
        live_keys = {tag.strip().casefold() for tag in out if tag.strip()}
        new_scores = {k: v for k, v in new_scores.items() if k in live_keys}
        return out, new_scores

    likes_list, likes_scores = merged(
        current.likes,
        current.summarizer_likes,
        current.dismissed_likes,
        current.summarizer_likes_scores,
        additions.get("likes", []),
    )
    dislikes_list, dislikes_scores = merged(
        current.dislikes,
        current.summarizer_dislikes,
        current.dismissed_dislikes,
        current.summarizer_dislikes_scores,
        additions.get("dislikes", []),
    )
    notes_list, notes_scores = merged(
        current.notes,
        current.summarizer_notes,
        current.dismissed_notes,
        current.summarizer_notes_scores,
        additions.get("notes", []),
    )

    return UserProfile(
        likes=list(current.likes),
        dislikes=list(current.dislikes),
        notes=list(current.notes),
        summarizer_likes=likes_list,
        summarizer_dislikes=dislikes_list,
        summarizer_notes=notes_list,
        summarizer_likes_scores=likes_scores,
        summarizer_dislikes_scores=dislikes_scores,
        summarizer_notes_scores=notes_scores,
        dismissed_likes=list(current.dismissed_likes),
        dismissed_dislikes=list(current.dismissed_dislikes),
        dismissed_notes=list(current.dismissed_notes),
    )


def facts_need_consolidation(
    characters: dict[str, list[Fact]],
) -> list[str]:
    """Return character ids whose fact list has grown past the
    consolidation threshold. Caller fires the LLM-driven cleanup
    pass for whichever ids the helper returns; an empty list
    short-circuits the round-trip."""
    return [cid for cid, facts in characters.items() if len(facts) >= FACTS_CONSOLIDATION_THRESHOLD]


def consolidate_facts_for_character(
    *,
    existing: list[Fact],
    consolidated_texts: list[tuple[str, str]],
) -> list[Fact]:
    """Apply the LLM consolidator's output to one character's facts.

    ``consolidated_texts`` is a list of ``(text, confidence)`` pairs
    in the order the LLM proposed. The output preserves Fact ids
    where possible (matched case-insensitively against the existing
    list) so a fact whose text the consolidator didn't change keeps
    its stable id; anything new gets a fresh id from ``Fact()``.
    Caps at ``FACTS_PER_CHARACTER_CAP - 5`` so the next few
    summarizer additions don't immediately re-trigger
    consolidation.
    """
    headroom = max(1, FACTS_PER_CHARACTER_CAP - 5)
    by_text = {f.text.strip().casefold(): f for f in existing}
    out: list[Fact] = []
    seen: set[str] = set()
    for raw_text, raw_conf in consolidated_texts:
        text = (raw_text or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        confidence = FactConfidence.canon if raw_conf == "canon" else FactConfidence.inferred
        existing_fact = by_text.get(key)
        if existing_fact is not None:
            out.append(existing_fact.model_copy(update={"text": text, "confidence": confidence}))
        else:
            out.append(Fact(text=text, confidence=confidence))
        if len(out) >= headroom:
            break
    return out


def profile_needs_consolidation(profile: UserProfile) -> bool:
    """True when any SUMMARIZER bucket has hit the cap. Player-owned
    entries don't count toward the cap so a player who hand-types
    twenty preferences via Settings never trips consolidation; the
    cap fires only when the summarizer's own inferred buckets grow
    past ``PROFILE_BUCKET_CAP``."""
    return (
        len(profile.summarizer_likes) >= PROFILE_BUCKET_CAP
        or len(profile.summarizer_dislikes) >= PROFILE_BUCKET_CAP
        or len(profile.summarizer_notes) >= PROFILE_BUCKET_CAP
    )


def consolidate_profile(*, profile: UserProfile, consolidated: dict[str, list[str]]) -> UserProfile:
    """Replace the SUMMARIZER buckets with the consolidator's output.

    Player-owned ``likes`` / ``dislikes`` / ``notes`` pass through
    untouched — the consolidator never sees them, so they can't be
    rewritten or dropped. Caps each summarizer bucket at
    ``PROFILE_BUCKET_CAP - 2`` so a single consolidation pass leaves
    headroom for the next few summarizer additions before
    consolidation has to run again.

    Strength preservation across consolidation: when the LLM
    rewrites a bucket — usually to merge near-duplicate tags
    into a smaller set — we have to decide what each output
    tag's strength is. Two cases:

      * **Exact carry-over.** An output tag whose case-folded
        form equals an input tag inherits that input's
        strength.
      * **Merge accumulation.** An output tag that doesn't
        exact-match any input — typical when the LLM consolidates
        "likes mysteries" + "likes character-driven stories" into
        "likes character-driven mysteries" — sums the strengths
        of every input tag whose case-folded form is a substring
        of the output (or vice-versa). This is the "batched
        together by the summarizer" mechanic the user asked for:
        multiple weak observations about a similar trait converge
        on a single trait that crosses the surface threshold
        sooner than any of them would alone.

    Strengths sum but stay clamped at ``TRAIT_STRENGTH_CAP``;
    pure-rename outputs (substring merges that find no inputs)
    fall back to ``INITIAL_TRAIT_STRENGTH`` so consolidator
    hallucinations can't bypass the gate.
    """
    from ..domain.settings import (
        INITIAL_TRAIT_STRENGTH,
        TRAIT_STRENGTH_CAP,
    )

    headroom = max(1, PROFILE_BUCKET_CAP - 2)

    def trimmed_with_strengths(
        original: list[str],
        original_scores: dict[str, float],
        rewritten: list[str],
    ) -> tuple[list[str], dict[str, float]]:
        seen: set[str] = set()
        out: list[str] = []
        new_scores: dict[str, float] = {}
        original_keys = [tag.strip().casefold() for tag in original]
        original_score_for_key: dict[str, float] = {
            key: float(original_scores.get(key, INITIAL_TRAIT_STRENGTH))
            for key in original_keys
            if key
        }
        for raw in rewritten:
            tag = (raw or "").strip()
            if not tag:
                continue
            key = tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(tag)
            # Strength resolution. Preference order:
            #   1. Exact match against an input tag — carry strength forward.
            #   2. Substring overlap with one or more input tags — sum
            #      their strengths (capped). Captures the "merged
            #      near-duplicates into one consolidated tag" case.
            #   3. No match — initial floor (treat as a new inference).
            if key in original_score_for_key:
                new_scores[key] = original_score_for_key[key]
            else:
                contributors = [
                    original_score_for_key[ok]
                    for ok in original_score_for_key
                    if ok and (ok in key or key in ok)
                ]
                if contributors:
                    new_scores[key] = min(
                        sum(contributors),
                        TRAIT_STRENGTH_CAP,
                    )
                else:
                    new_scores[key] = INITIAL_TRAIT_STRENGTH
            if len(out) >= headroom:
                break
        return out, new_scores

    likes_list, likes_scores = trimmed_with_strengths(
        list(profile.summarizer_likes),
        profile.summarizer_likes_scores,
        consolidated.get("likes", list(profile.summarizer_likes)),
    )
    dislikes_list, dislikes_scores = trimmed_with_strengths(
        list(profile.summarizer_dislikes),
        profile.summarizer_dislikes_scores,
        consolidated.get("dislikes", list(profile.summarizer_dislikes)),
    )
    notes_list, notes_scores = trimmed_with_strengths(
        list(profile.summarizer_notes),
        profile.summarizer_notes_scores,
        consolidated.get("notes", list(profile.summarizer_notes)),
    )

    return UserProfile(
        likes=list(profile.likes),
        dislikes=list(profile.dislikes),
        notes=list(profile.notes),
        summarizer_likes=likes_list,
        summarizer_dislikes=dislikes_list,
        summarizer_notes=notes_list,
        summarizer_likes_scores=likes_scores,
        summarizer_dislikes_scores=dislikes_scores,
        summarizer_notes_scores=notes_scores,
        dismissed_likes=list(profile.dismissed_likes),
        dismissed_dislikes=list(profile.dismissed_dislikes),
        dismissed_notes=list(profile.dismissed_notes),
    )


def apply_summary(
    *,
    summary: LlmSummaryResult,
    character_facts: dict[str, list[Fact]],
    current_stage_id: str | None,
    plot_outline: list[PlotStage],
    user_profile: UserProfile | None = None,
    on_stage: list[str] | None = None,
    player_character_id: str | None = None,
) -> SummaryApplication:
    """Resolve the LLM's summary into a concrete set of mutations.

    * Facts are pruned + extended per the summary.
    * ``current_stage_id`` is updated only if the summary's choice
      is a real id in the current (or revised) outline; otherwise
      the existing pointer stays put. This protects against the
      LLM hallucinating a stage that doesn't exist.
    * ``revised_outline`` is propagated only when the LLM emitted
      one — the default null means "no drift, leave the plan."
    * ``user_profile_additions`` are merged into ``user_profile``
      with case-insensitive dedupe. The caller is responsible for
      writing the returned profile back to Settings + persisting.
    """
    pruned_ids = set(summary.pruned_fact_ids)
    updated_facts: dict[str, list[Fact]] = {}
    for cid, facts in character_facts.items():
        kept = [f for f in facts if f.id not in pruned_ids]
        new_facts = summary.new_facts_by_character.get(cid, [])
        merged = kept + list(new_facts)
        updated_facts[cid] = _cap_facts(merged)
    for cid, facts in summary.new_facts_by_character.items():
        if cid not in updated_facts:
            updated_facts[cid] = _cap_facts(list(facts))

    direction = DirectionSignal.none
    if summary.direction_signal == "stay_focused":
        direction = DirectionSignal.stay_focused
    elif summary.direction_signal == "reintroduce_thread":
        direction = DirectionSignal.reintroduce_thread

    # If a revised outline was supplied, the new stage id must
    # resolve against THAT outline; otherwise against the existing
    # one. Hallucinated ids fall back to the prior pointer.
    effective_outline = (
        summary.revised_outline if summary.revised_outline is not None else plot_outline
    )
    valid_ids = {s.id for s in effective_outline}
    next_stage_id = current_stage_id
    if summary.current_stage_id and summary.current_stage_id in valid_ids:
        next_stage_id = summary.current_stage_id

    merged_profile: UserProfile | None = None
    if user_profile is not None:
        merged_profile = _merge_profile_additions(
            current=user_profile,
            additions={
                "likes": summary.user_profile_additions.likes,
                "dislikes": summary.user_profile_additions.dislikes,
                "notes": summary.user_profile_additions.notes,
            },
        )

    assessment = summary.summarizer_assessment.strip()
    if len(assessment) > SUMMARIZER_ASSESSMENT_CAP:
        # Trim from the START — the tail is the most recent reasoning
        # the summarizer added this turn. Prefix a "..." so a downstream
        # prompt doesn't read the trim as a fluent sentence boundary.
        assessment = "..." + assessment[-(SUMMARIZER_ASSESSMENT_CAP - 3) :]

    # Validate offstage ids: keep only those that are actually
    # on stage right now AND aren't the player character. The
    # player is the lens — never silently exited, even if the
    # summarizer hallucinates them as offstage-able. A
    # hallucinated id (one not in on_stage) is a no-op that
    # would cause the caller to do an unnecessary game.model_copy,
    # so filter here.
    offstage_ids: list[str] = []
    if summary.characters_to_offstage and on_stage is not None:
        on_stage_set = set(on_stage)
        seen: set[str] = set()
        for cid in summary.characters_to_offstage:
            if cid == player_character_id:
                continue
            if cid not in on_stage_set:
                continue
            if cid in seen:
                continue
            seen.add(cid)
            offstage_ids.append(cid)

    return SummaryApplication(
        character_facts=updated_facts,
        summarizer_assessment=assessment,
        direction_signal=direction,
        current_stage_id=next_stage_id,
        revised_outline=summary.revised_outline,
        user_profile=merged_profile,
        characters_to_offstage=offstage_ids,
    )


def _cap_facts(facts: list[Fact]) -> list[Fact]:
    """Trim the facts list to ``FACTS_PER_CHARACTER_CAP`` entries.

    Canon facts are preserved before inferred ones (so the canon
    spine survives even when a long-running session would otherwise
    drop them); within each group, the OLDEST entries are trimmed
    first because the most recently learned facts are usually the
    most relevant to the current scene.
    """
    if len(facts) <= FACTS_PER_CHARACTER_CAP:
        return facts
    canon = [f for f in facts if f.confidence.value == "canon"]
    inferred = [f for f in facts if f.confidence.value != "canon"]
    if len(canon) >= FACTS_PER_CHARACTER_CAP:
        return canon[-FACTS_PER_CHARACTER_CAP:]
    take_inferred = FACTS_PER_CHARACTER_CAP - len(canon)
    return canon + inferred[-take_inferred:]
