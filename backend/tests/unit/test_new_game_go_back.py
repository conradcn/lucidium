"""``c2s/new_game/go_back`` — abandon the new-game interview and
return to the main menu.

Pins:
  * the session's ``interview`` field is reset to a fresh
    ``InterviewState`` (no leaked partial answers into the next
    new-game session);
  * EVERY in-flight prefetch the interview spawned is cancelled
    (world_init, char_desc, name_options, preview bg, preview guide,
    pc_portrait) — leaving any of them running would burn credits
    on inputs the player has already discarded;
  * the handler emits NO state-patch (the renderer optimistically
    transitions to the main menu before the handler returns).
"""

from __future__ import annotations

import asyncio

import pytest

from lucidium.api.handlers import (
    HandlerContext,
    new_game_go_back_handler,
)
from lucidium.api.messages import C2SNewGameGoBack


class _StubSession:
    def __init__(self) -> None:
        from lucidium.domain.settings import Settings
        from lucidium.orchestration.session import InterviewState

        self.interview = InterviewState()
        self.settings = Settings()
        self._world_init_task: asyncio.Task | None = None
        self._char_desc_task: asyncio.Task | None = None
        self._name_options_task: asyncio.Task | None = None
        self._preview_bg_task: asyncio.Task | None = None
        self._preview_guide_task: asyncio.Task | None = None
        self._pc_portrait_task: asyncio.Task | None = None


async def _drain(gen) -> list:
    out = []
    async for msg in gen:
        out.append(msg)
    return out


@pytest.mark.asyncio
async def test_go_back_resets_interview_to_fresh_state() -> None:
    from lucidium.orchestration.session import InterviewState

    session = _StubSession()
    session.interview.setting = "stone harbor"
    session.interview.genre = "Mystery"
    session.interview.visual_style = "ink wash"
    session.interview.character_description = "wry archivist"
    session.interview.character_name = "Iris"
    session.interview.pronouns = "she/her"
    ctx = HandlerContext(session=session)

    result = await new_game_go_back_handler(C2SNewGameGoBack(), ctx)
    messages = await _drain(result)

    # No state-patch on the wire: the renderer has already
    # navigated to the main menu and a stray patch would race the
    # next New Game click's start handler.
    assert messages == []

    # Interview state has been wiped clean.
    fresh = InterviewState()
    assert session.interview == fresh, (
        f"interview state was not reset; still holds {session.interview!r}"
    )


@pytest.mark.asyncio
async def test_go_back_cancels_every_interview_prefetch() -> None:
    """Every in-flight task the interview spawned must be cancelled.
    Without this, abandoning a half-filled review screen would leave
    a 30 s world_init LLM call (the most expensive task in the
    interview) running for a session the player has walked away
    from, and a returning ``c2s/new_game/start`` would race the
    stale prefetch instead of getting a fresh one."""
    session = _StubSession()
    ctx = HandlerContext(session=session)

    started = {
        name: asyncio.Event()
        for name in (
            "_world_init_task",
            "_char_desc_task",
            "_name_options_task",
            "_preview_bg_task",
            "_preview_guide_task",
            "_pc_portrait_task",
        )
    }
    cancelled = {name: asyncio.Event() for name in started}

    def make_slow(name: str):
        async def _slow() -> str:
            started[name].set()
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                cancelled[name].set()
                raise
            return "never"

        return _slow

    for name in started:
        task = asyncio.create_task(make_slow(name)())
        setattr(session, name, task)

    for name, event in started.items():
        try:
            await asyncio.wait_for(event.wait(), timeout=1.0)
        except TimeoutError:
            pytest.fail(f"prefetch {name} never started")

    result = await new_game_go_back_handler(C2SNewGameGoBack(), ctx)
    await _drain(result)

    for name, event in cancelled.items():
        try:
            await asyncio.wait_for(event.wait(), timeout=1.0)
        except TimeoutError:
            pytest.fail(f"prefetch {name} was not cancelled by go_back")
        assert getattr(session, name) is None, (
            f"{name} was cancelled but the session slot was not cleared; "
            "a stale Task reference would re-fire on the next interview"
        )
