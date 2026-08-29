"""Regression: turn latency was growing over a long session because
character.facts and summarizer_assessment had no upper bound, so each
turn shipped a strictly larger storyteller prompt to the LLM.

This module locks in the caps:

* ``apply_summary`` trims any character's facts list to
  ``FACTS_PER_CHARACTER_CAP`` (canon facts preserved before inferred).
* ``apply_summary`` clamps ``summarizer_assessment`` to
  ``SUMMARIZER_ASSESSMENT_CAP`` characters.
* The cross-save user profile already has a per-bucket cap with
  consolidation; this test asserts the cap stays in place after many
  apply_summary calls.

A second test simulates a long arc (50 summarizer passes that each
add new facts and append to the assessment) and asserts the
storyteller prompt size at the END of the arc is within a constant
factor of the size at turn 1. If a future change uncaps any of these
fields, the prompt grows linearly with turns and the assertion
trips.
"""

from __future__ import annotations

from lucidium.domain.character import Character, Fact, FactConfidence
from lucidium.domain.dialog import DialogNode, DialogNodeState
from lucidium.domain.settings import UserProfile
from lucidium.domain.world import WorldState
from lucidium.orchestration import summarizer
from lucidium.orchestration.prompts import text_gen
from lucidium.orchestration.responses import (
    LlmSummaryResult,
    LlmUserProfileAdditions,
)


def _fact(text: str, *, confidence: str = "inferred") -> Fact:
    return Fact(text=text, confidence=FactConfidence(confidence))


def _summary(
    *,
    new_facts: list[Fact],
    assessment: str = "small assessment",
    likes: list[str] | None = None,
) -> LlmSummaryResult:
    return LlmSummaryResult(
        summarizer_assessment=assessment,
        new_facts_by_character={"c1": new_facts},
        user_profile_additions=LlmUserProfileAdditions(
            likes=likes or [],
        ),
    )


def test_facts_cap_holds_after_many_summarizer_passes() -> None:
    facts: dict[str, list[Fact]] = {"c1": []}
    profile = UserProfile()
    for turn in range(60):
        summary = _summary(
            new_facts=[_fact(f"learned thing #{turn}")],
            assessment=f"turn {turn} assessment",
        )
        applied = summarizer.apply_summary(
            summary=summary,
            character_facts=facts,
            current_stage_id=None,
            plot_outline=[],
            user_profile=profile,
        )
        facts = applied.character_facts
        if applied.user_profile is not None:
            profile = applied.user_profile

    assert len(facts["c1"]) <= summarizer.FACTS_PER_CHARACTER_CAP
    # The latest fact MUST survive — recency wins on inferred-only
    # trims because we drop from the front.
    assert any("#59" in f.text for f in facts["c1"])


def test_canon_facts_outlive_inferred_when_capped() -> None:
    canon = [_fact(f"canon fact {i}", confidence="canon") for i in range(5)]
    inferred = [_fact(f"inferred fact {i}") for i in range(50)]
    capped = summarizer._cap_facts(canon + inferred)

    assert len(capped) == summarizer.FACTS_PER_CHARACTER_CAP
    canon_survivors = [f for f in capped if f.confidence == FactConfidence.canon]
    assert len(canon_survivors) == 5, "canon facts must not be evicted"


def test_assessment_clamped_when_oversized() -> None:
    huge = "x" * (summarizer.SUMMARIZER_ASSESSMENT_CAP + 500)
    summary = _summary(new_facts=[], assessment=huge)
    applied = summarizer.apply_summary(
        summary=summary,
        character_facts={},
        current_stage_id=None,
        plot_outline=[],
        user_profile=None,
    )
    assert len(applied.summarizer_assessment) <= summarizer.SUMMARIZER_ASSESSMENT_CAP


def test_storyteller_prompt_growth_is_bounded_over_long_arc() -> None:
    """End-to-end: build the storyteller prompt at turn 1 and turn 50
    after 50 simulated summarizer passes. The prompt at turn 50 must
    not exceed the turn-1 prompt by more than a generous constant
    factor (4×). A regression that uncaps facts / assessment would
    grow the prompt linearly with turn count and trip this gate.
    """
    facts: dict[str, list[Fact]] = {"c1": []}
    profile = UserProfile()
    assessment = "opening assessment"

    def make_character(facts_list: list[Fact]) -> Character:
        return Character(
            id="c1",
            is_player=False,
            name="Mira",
            description="archive keeper",
            gender="female",
            age=34,
            ethnicity="local",
            skin="pale",
            hair_color="black",
            hairstyle="bobbed",
            eye_color="grey",
            build="slim",
            bust="small",
            outfit="ink-stained smock",
            pose="standing",
            expression="watchful",
            facts=facts_list,
            seed=7,
        )

    def make_world(assessment_text: str) -> WorldState:
        return WorldState(
            game_name="Test",
            setting="harbor",
            genre="mystery",
            visual_style="ink",
            summarizer_assessment=assessment_text,
        )

    def make_history(turn: int) -> list[DialogNode]:
        return [
            DialogNode(
                id=f"n{i}",
                parent_id=None if i == 0 else f"n{i - 1}",
                text=f"beat {i}: a small line of dialog.",
                premise_hash="h",
                state=DialogNodeState.committed,
            )
            for i in range(min(turn, 6))
        ]

    def render_prompt(turn: int) -> str:
        c = make_character(facts["c1"])
        prompt = text_gen.build(
            world=make_world(assessment),
            history=make_history(turn),
            on_stage={"c1": c},
            off_stage={},
            chosen_option_text=None,
            user_likes=list(profile.likes),
            user_dislikes=list(profile.dislikes),
            user_notes=list(profile.notes),
        )
        return "".join(m.get("content", "") for m in prompt)

    turn1_size = len(render_prompt(turn=1))

    for turn in range(50):
        summary = _summary(
            new_facts=[_fact(f"new fact at turn {turn}")],
            assessment=f"running assessment with growing detail at turn {turn} " * 3,
            likes=[f"likes thing {turn}"] if turn % 5 == 0 else [],
        )
        applied = summarizer.apply_summary(
            summary=summary,
            character_facts=facts,
            current_stage_id=None,
            plot_outline=[],
            user_profile=profile,
        )
        facts = applied.character_facts
        assessment = applied.summarizer_assessment
        if applied.user_profile is not None:
            profile = applied.user_profile

    turn50_size = len(render_prompt(turn=50))

    assert turn50_size <= turn1_size * 4, (
        f"storyteller prompt grew unbounded — turn1={turn1_size}, "
        f"turn50={turn50_size}, ratio={turn50_size / turn1_size:.2f}. "
        "Check that facts / summarizer_assessment / user_profile caps "
        "are still applied in apply_summary."
    )
