"""The naming-taboo rule must reach every prompt that asks the LLM to
invent a character name — otherwise we keep getting Lyra and Elara,
which immediately tag the output as AI-written for any reader who has
seen LLM fiction. Pin: text_gen + every interview-side name-emitting
prompt embeds the taboo list and at least the canonical offenders."""

from __future__ import annotations

from lucidium.domain.world import WorldState
from lucidium.orchestration.prompts import text_gen
from lucidium.orchestration.prompts.common import FORBIDDEN_NAMES, NAMING_TABOO_RULE
from lucidium.orchestration.prompts.interview import (
    name_options,
    side_character_expansion,
    surprise_me_scenario,
    world_init,
)


def _world() -> WorldState:
    return WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink",
    )


def _full_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(m.get("content", "") for m in messages)


def test_naming_taboo_lists_canonical_offenders() -> None:
    for name in ("Lyra", "Elara", "Aria", "Kael", "Seraphina"):
        assert name in FORBIDDEN_NAMES


def test_naming_taboo_rule_constant_is_self_contained() -> None:
    assert "NAMING TABOO" in NAMING_TABOO_RULE
    for name in ("Lyra", "Elara", "Kael"):
        assert name in NAMING_TABOO_RULE


def test_text_gen_embeds_naming_taboo_with_canonical_offenders() -> None:
    msgs = text_gen.build(
        world=_world(),
        history=[],
        on_stage={},
        off_stage={},
        chosen_option_text=None,
    )
    blob = _full_text(msgs)
    assert "NAMING TABOO" in blob
    for name in ("Lyra", "Elara"):
        assert name in blob


def test_name_options_embeds_naming_taboo() -> None:
    blob = _full_text(name_options(setting="harbor", character_description="constable"))
    assert "NAMING TABOO" in blob
    assert "Lyra" in blob
    assert "Elara" in blob


def test_side_character_expansion_embeds_naming_taboo() -> None:
    blob = _full_text(
        side_character_expansion(
            one_line="weathered keeper of the lighthouse",
            setting="harbor",
        )
    )
    assert "NAMING TABOO" in blob
    assert "Lyra" in blob


def test_world_init_embeds_naming_taboo() -> None:
    blob = _full_text(
        world_init(
            setting="harbor",
            genre="mystery",
            visual_style="ink",
            character_description="harbour constable",
            name="Hale",
            side_characters=["the keeper"],
        )
    )
    assert "NAMING TABOO" in blob
    assert "Lyra" in blob


def test_surprise_me_scenario_embeds_naming_taboo() -> None:
    blob = _full_text(
        surprise_me_scenario(
            visual_style="ink",
            likes=[],
            dislikes=[],
            notes=[],
        )
    )
    assert "NAMING TABOO" in blob
    assert "Elara" in blob
