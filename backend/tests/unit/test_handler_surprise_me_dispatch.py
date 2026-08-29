"""``c2s/new_game/surprise_me`` through dispatch.

Surprise Me skips the five-step interview: one small LLM call authors a
scenario, the handler fills every ``InterviewState`` slot the interview
would have filled, and then hands off to ``new_game_confirm_handler`` for
the (expensive, separately-tested) world_init + opening-chain build.

The hand-off is stubbed here. What these tests pin is the part unique to
this handler and untested anywhere else: that the scenario lands in the
interview state, that the visual style is inherited from the most recent
save (with a defined fallback when there is none), that the player's
cross-save profile actually reaches the prompt, and that a corrupt save
degrades instead of blocking a new game.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lucidium.api import handlers
from lucidium.api.errors import ProviderValidationError
from lucidium.api.handlers import HandlerContext
from lucidium.api.messages import MessageType, S2CStatePatch
from lucidium.domain.settings import Settings, UserProfile
from lucidium.orchestration.prompts import interview as interview_prompts
from lucidium.persistence import save_store

from .handler_harness import (
    ScriptedLlm,
    dispatch,
    make_game,
    make_registry,
    make_session,
    types_of,
)

SCENARIO = {
    "setting": "a salvage rig above a dead reactor",
    "genre": "industrial thriller",
    "character_description": "a decommissioning inspector with a forged badge",
    "name": "Vale Ostrander",
}


@pytest.fixture
def stub_confirm(monkeypatch: pytest.MonkeyPatch) -> list[HandlerContext]:
    """Replace the world_init hand-off with a sentinel reply.

    ``new_game_confirm_handler`` is exercised by the interview
    integration tests; re-running it here would make every case below a
    multi-LLM-call world build for no added coverage.
    """
    seen: list[HandlerContext] = []

    async def _fake_confirm(_payload: Any, ctx: HandlerContext):
        seen.append(ctx)

        async def gen():
            yield (MessageType.s2c_state_patch, S2CStatePatch(ops=[]))

        return gen()

    monkeypatch.setattr(handlers, "new_game_confirm_handler", _fake_confirm)
    return seen


def _session_with_scenario(tmp_app_data: Path, *, settings: Settings | None = None):
    llm = ScriptedLlm([json.dumps(SCENARIO)])
    return make_session(tmp_app_data, settings=settings, llm_client=llm), llm


# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_surprise_me_fills_the_interview_state_and_hands_off(
    tmp_app_data: Path, stub_confirm: list[HandlerContext]
) -> None:
    session, _llm = _session_with_scenario(tmp_app_data)

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_new_game_surprise_me,
        {},
    )

    # It delegated exactly once, with the same context.
    assert len(stub_confirm) == 1
    assert stub_confirm[0].session is session
    assert types_of(messages) == [MessageType.s2c_state_patch]

    interview = session.interview
    assert interview.setting == SCENARIO["setting"]
    assert interview.genre == SCENARIO["genre"]
    assert interview.character_description == SCENARIO["character_description"]
    assert interview.character_name == SCENARIO["name"]
    # Surprise Me never authors side characters up front.
    assert interview.side_character_descriptions == []
    assert interview.side_characters == []


@pytest.mark.asyncio
async def test_surprise_me_falls_back_to_the_first_visual_style_with_no_saves(
    tmp_app_data: Path, stub_confirm: list[HandlerContext]
) -> None:
    session, _llm = _session_with_scenario(tmp_app_data)

    await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_new_game_surprise_me,
        {},
    )

    assert session.interview.visual_style == interview_prompts.VISUAL_STYLES_HYPERREALISTIC[0]


@pytest.mark.asyncio
async def test_surprise_me_reuses_the_most_recent_saves_visual_style(
    tmp_app_data: Path, stub_confirm: list[HandlerContext]
) -> None:
    session, _llm = _session_with_scenario(tmp_app_data)
    prior, *_ = make_game()
    prior = prior.model_copy(
        update={"world": prior.world.model_copy(update={"visual_style": "wet plate collodion"})}
    )
    save_store.commit_save(prior, Settings(), name="Prior", root=session.saves_root)

    await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_new_game_surprise_me,
        {},
    )

    assert session.interview.visual_style == "wet plate collodion"


@pytest.mark.asyncio
async def test_surprise_me_survives_a_corrupt_most_recent_save(
    tmp_app_data: Path, stub_confirm: list[HandlerContext]
) -> None:
    """A save that lists but won't parse must not block a new game — the
    handler falls back to the default style and carries on."""
    session, _llm = _session_with_scenario(tmp_app_data)
    prior, *_ = make_game()
    save_store.commit_save(prior, Settings(), name="Prior", root=session.saves_root)
    (session.saves_root / prior.id / "game.json").write_text("{ not json", encoding="utf-8")

    await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_new_game_surprise_me,
        {},
    )

    assert session.interview.visual_style == interview_prompts.VISUAL_STYLES_HYPERREALISTIC[0]
    assert session.interview.setting == SCENARIO["setting"]


@pytest.mark.asyncio
async def test_surprise_me_feeds_the_surfaced_profile_into_the_prompt(
    tmp_app_data: Path, stub_confirm: list[HandlerContext]
) -> None:
    """The scenario is meant to be tailored to the cross-save profile.
    Without this, the profile plumbing could rot silently — the handler
    would still 'work', just generically."""
    settings = Settings(
        user_profile=UserProfile(likes=["derelict machinery"], dislikes=["courtroom drama"])
    )
    session, llm = _session_with_scenario(tmp_app_data, settings=settings)

    await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_new_game_surprise_me,
        {},
    )

    assert llm.prompts, "the scenario LLM call never happened"
    prompt_text = json.dumps(llm.prompts[0])
    assert "derelict machinery" in prompt_text
    assert "courtroom drama" in prompt_text


@pytest.mark.asyncio
async def test_surprise_me_rejects_an_unparseable_scenario_before_delegating(
    tmp_app_data: Path, stub_confirm: list[HandlerContext]
) -> None:
    """A malformed scenario must not fan out into a world build with a
    half-filled interview state."""
    llm = ScriptedLlm(["not json at all"] * 8)
    session = make_session(tmp_app_data, llm_client=llm)

    with pytest.raises(ProviderValidationError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_new_game_surprise_me,
            {},
        )

    assert stub_confirm == []
    assert session.interview.setting == ""
