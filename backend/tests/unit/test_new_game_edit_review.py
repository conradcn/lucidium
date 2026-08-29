"""``c2s/new_game/edit_review`` — edit a previously-answered
interview field from the Review step.

Pin: the handler updates the named interview field in-place,
echoes a state/patch, and cancels + re-fires the in-flight
world_init prefetch so the next ``new_game/confirm`` consumes
a prefetch that matches what's on screen.

Without the prefetch swap, an edit on Review would fall through
silently — world_init would complete with the prior values,
the confirm handler would consume that stale result, and the
player's edit would have no effect on the actual game.
"""

from __future__ import annotations

import asyncio

import pytest

from lucidium.api.errors import SchemaError
from lucidium.api.handlers import (
    HandlerContext,
    new_game_edit_review_handler,
)
from lucidium.api.messages import (
    C2SNewGameEditReview,
    MessageType,
)


class _StubSession:
    def __init__(self) -> None:
        from lucidium.domain.settings import Settings
        from lucidium.orchestration.session import InterviewState

        self.interview = InterviewState()
        self._world_init_task = None
        # The prefetch path reads ``session.settings.mature_content``
        # and ``session.settings.music.enabled``. Default Settings
        # gives sensible values for both.
        self.settings = Settings()

    def install_game(self, _g: object) -> None:
        pass

    async def commit(self) -> None:
        self.commit_blocking()

    def commit_blocking(self) -> None:
        pass

    async def llm_text(self, _prompt, **_kwargs):
        # Stand-in so tests where the handler RE-FIRES the
        # prefetch (because character_name is set) don't blow up
        # on a missing method when the freshly-spawned task tries
        # to call into the real LLM. The prefetch returns the raw
        # text it gets; the test never awaits the new task, so
        # the empty body is fine.
        return "{}", []


async def _drain(gen) -> list:
    out = []
    async for msg in gen:
        out.append(msg)
    return out


@pytest.mark.asyncio
async def test_edit_review_updates_named_field() -> None:
    session = _StubSession()
    session.interview.setting = "old harbor"
    ctx = HandlerContext(session=session)

    result = await new_game_edit_review_handler(
        C2SNewGameEditReview(field="setting", value="storm-shadowed harbor"),
        ctx,
    )
    messages = await _drain(result)

    assert session.interview.setting == "storm-shadowed harbor"
    # Echo state/patch with the new value.
    assert len(messages) == 1
    msg_type, payload = messages[0]
    assert msg_type == MessageType.s2c_state_patch
    assert payload.ops[0].path == "/interview/setting"
    assert payload.ops[0].value == "storm-shadowed harbor"


@pytest.mark.asyncio
async def test_edit_review_rejects_unknown_field() -> None:
    """Allow-list guard. ``side_characters`` and
    ``preview_*_path`` aren't editable from this handler — a
    stray patch trying to write them raises SchemaError."""
    session = _StubSession()
    ctx = HandlerContext(session=session)

    with pytest.raises(SchemaError):
        await new_game_edit_review_handler(
            C2SNewGameEditReview(
                field="side_characters",  # not in the allow-list
                value="haxx",
            ),
            ctx,
        )
    with pytest.raises(SchemaError):
        await new_game_edit_review_handler(
            C2SNewGameEditReview(
                field="preview_background_path",
                value="/etc/passwd",
            ),
            ctx,
        )


@pytest.mark.asyncio
async def test_edit_review_cancels_inflight_world_init_prefetch() -> None:
    """The prefetch is launched at the end of the Name step
    against the values that EXISTED THEN. Editing on Review
    must cancel that task so the next prefetch (or confirm)
    consumes fresh values."""
    session = _StubSession()
    session.interview.setting = "old"
    session.interview.character_name = "Iris"

    cancelled: list[bool] = []

    async def _stale_prefetch() -> None:
        try:
            await asyncio.sleep(60)  # never completes
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    session._world_init_task = asyncio.create_task(_stale_prefetch())
    # Yield once so the task body actually starts running and hits
    # its sleep — without this, ``cancel()`` lands on a task that
    # never executed and our except-block (which records the
    # cancellation) never runs.
    await asyncio.sleep(0)
    ctx = HandlerContext(session=session)

    # We need to AVOID actually running _prefetch_world_init
    # here because it makes a real LLM call. Stub it via the
    # session's llm_text path: the prefetch helper calls
    # ``session.llm_text(...)`` which would error here. The
    # quickest way to verify "cancel-then-re-fire" is to patch
    # _prefetch_world_init to a no-op for this test.
    import lucidium.api.handlers as h

    real_prefetch = h._prefetch_world_init

    async def _noop_prefetch(*_a, **_kw):
        return ""

    h._prefetch_world_init = _noop_prefetch
    try:
        old_task = session._world_init_task
        await _drain(
            await new_game_edit_review_handler(
                C2SNewGameEditReview(field="setting", value="new"),
                ctx,
            )
        )
        # Wait for the old task to actually process its
        # cancellation (cancel() is a flag set, not a sync
        # operation).
        try:
            await old_task
        except asyncio.CancelledError:
            pass
        assert cancelled == [True], "old prefetch task should have been cancelled"
        assert session._world_init_task is not None, "fresh prefetch should have been queued"
        assert session._world_init_task is not old_task, (
            "fresh prefetch should be a different task instance"
        )
    finally:
        h._prefetch_world_init = real_prefetch


@pytest.mark.asyncio
async def test_edit_review_skips_prefetch_re_fire_when_name_not_yet_answered() -> None:
    """During the interview itself (player editing Setting after
    answering Genre but before Name), there's no prefetch to
    refresh. The handler must skip the prefetch fork instead of
    firing one against an empty character_name."""
    session = _StubSession()
    session.interview.setting = "old"
    session.interview.character_name = ""  # Name step not yet answered
    ctx = HandlerContext(session=session)

    await _drain(
        await new_game_edit_review_handler(
            C2SNewGameEditReview(field="setting", value="new"),
            ctx,
        )
    )
    # No prefetch fired — the world_init task is still None.
    assert session._world_init_task is None


@pytest.mark.asyncio
async def test_edit_review_accepts_all_five_review_fields() -> None:
    """Pin the allow-list explicitly so a future refactor
    doesn't accidentally drop any of the five fields the
    Review screen exposes."""
    fields_to_test = [
        ("setting", "x"),
        ("visual_style", "y"),
        ("genre", "z"),
        ("character_description", "w"),
        ("character_name", "v"),
    ]
    for field, value in fields_to_test:
        session = _StubSession()
        ctx = HandlerContext(session=session)
        await _drain(
            await new_game_edit_review_handler(
                C2SNewGameEditReview(field=field, value=value),
                ctx,
            )
        )
        assert getattr(session.interview, field) == value
