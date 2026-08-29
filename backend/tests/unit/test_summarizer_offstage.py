"""The summarizer can remove idle characters from ``Game.on_stage``.

Pin: ``LlmSummaryResult.characters_to_offstage`` flows through
``apply_summary`` into ``SummaryApplication.characters_to_offstage``,
filtered against the live on-stage roster (hallucinated ids dropped)
and against the player character (the player is the lens, never
silently exited). The ``world_refresh`` prompt surfaces the on-stage
roster and asks for the offstage list as job 5.
"""

from __future__ import annotations

from lucidium.domain.world import WorldState
from lucidium.orchestration import summarizer
from lucidium.orchestration.prompts import world_refresh
from lucidium.orchestration.responses import LlmSummaryResult


def _world() -> WorldState:
    return WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink",
    )


def _summary(**overrides: object) -> LlmSummaryResult:
    base: dict[str, object] = {
        "summarizer_assessment": "ok",
        "direction_signal": "none",
    }
    base.update(overrides)
    return LlmSummaryResult(**base)  # type: ignore[arg-type]


def test_summary_passes_offstage_through_when_id_is_currently_on_stage() -> None:
    application = summarizer.apply_summary(
        summary=_summary(characters_to_offstage=["c1", "c2"]),
        character_facts={"c1": [], "c2": [], "c3": []},
        current_stage_id=None,
        plot_outline=[],
        on_stage=["c1", "c2", "c3"],
        player_character_id="p",
    )
    assert application.characters_to_offstage == ["c1", "c2"]


def test_summary_drops_offstage_id_not_currently_on_stage() -> None:
    application = summarizer.apply_summary(
        summary=_summary(characters_to_offstage=["ghost-character"]),
        character_facts={},
        current_stage_id=None,
        plot_outline=[],
        on_stage=["c1"],
        player_character_id=None,
    )
    assert application.characters_to_offstage == []


def test_summary_drops_player_character_from_offstage_list() -> None:
    application = summarizer.apply_summary(
        summary=_summary(characters_to_offstage=["p", "c1"]),
        character_facts={},
        current_stage_id=None,
        plot_outline=[],
        on_stage=["p", "c1"],
        player_character_id="p",
    )
    assert application.characters_to_offstage == ["c1"]


def test_summary_dedupes_offstage_ids() -> None:
    application = summarizer.apply_summary(
        summary=_summary(characters_to_offstage=["c1", "c1", "c2"]),
        character_facts={},
        current_stage_id=None,
        plot_outline=[],
        on_stage=["c1", "c2"],
        player_character_id=None,
    )
    assert application.characters_to_offstage == ["c1", "c2"]


def test_summary_default_offstage_is_empty_list() -> None:
    application = summarizer.apply_summary(
        summary=_summary(),
        character_facts={},
        current_stage_id=None,
        plot_outline=[],
        on_stage=["c1", "c2"],
        player_character_id=None,
    )
    assert application.characters_to_offstage == []


def test_world_refresh_prompt_lists_on_stage_roster_and_idle_cleanup_job() -> None:
    msgs = world_refresh.build(
        world=_world(),
        history=[],
        player_typed_quotes=[],
        on_stage_ids=["hale", "mira"],
        player_character_id="player",
        character_names={"hale": "Hale", "mira": "Mira", "player": "You"},
    )
    blob = "\n".join(m.get("content", "") for m in msgs)
    # Roster present, ids + names rendered.
    assert "CURRENTLY ON STAGE" in blob
    assert "hale (Hale)" in blob
    assert "mira (Mira)" in blob
    # Player NEVER appears on the roster — they're the lens.
    assert "(You)" not in blob
    # Job 5 instruction is present.
    assert "characters_to_offstage" in blob
    assert "Departed-character cleanup" in blob
    # Soft trigger — only when the scene has explicitly moved on.
    assert "explicitly moved past them" in blob
    assert "NOT about idle pruning" in blob
    # Per-character walk: prompt walks the roster and emphasizes
    # that multiple ids per call is normal when the player moves
    # scenes (caught a B2-miss regression where only the most
    # recently departed character was offstaged).
    assert "EACH id" in blob
    assert "Multiple ids" in blob


def test_world_refresh_prompt_handles_empty_on_stage_gracefully() -> None:
    msgs = world_refresh.build(
        world=_world(),
        history=[],
        player_typed_quotes=[],
        on_stage_ids=[],
        player_character_id=None,
    )
    blob = "\n".join(m.get("content", "") for m in msgs)
    assert "CURRENTLY ON STAGE" in blob
    assert "(no characters on stage)" in blob


def test_world_refresh_prompt_handles_player_only_stage() -> None:
    """When the only on-stage entry is the player, we filter them
    out and render the 'only the player is on stage' fallback so
    the LLM doesn't see a confusing empty roster after the
    player-id filter."""
    msgs = world_refresh.build(
        world=_world(),
        history=[],
        player_typed_quotes=[],
        on_stage_ids=["player"],
        player_character_id="player",
        character_names={"player": "You"},
    )
    blob = "\n".join(m.get("content", "") for m in msgs)
    assert "only the player is on stage" in blob


def test_response_shape_includes_characters_to_offstage_field() -> None:
    """Pin the JSON schema the LLM is asked to return — without
    the field on the schema doc, GPT-class models often silently
    omit it from the response."""
    msgs = world_refresh.build(
        world=_world(),
        history=[],
        player_typed_quotes=[],
        on_stage_ids=["c1"],
        player_character_id=None,
    )
    blob = "\n".join(m.get("content", "") for m in msgs)
    assert '"characters_to_offstage"' in blob


def test_llm_summary_result_parses_with_offstage_field() -> None:
    """Pydantic schema accepts the new field with a default of []."""
    no_field = LlmSummaryResult(
        summarizer_assessment="ok",
        direction_signal="none",
    )
    assert no_field.characters_to_offstage == []
    with_field = LlmSummaryResult(
        summarizer_assessment="ok",
        direction_signal="none",
        characters_to_offstage=["c1", "c2"],
    )
    assert with_field.characters_to_offstage == ["c1", "c2"]
