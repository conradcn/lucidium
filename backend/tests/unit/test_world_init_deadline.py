"""New-game ``world_init`` acquisition deadline.

``_acquire_world_init`` resolves the world_init LLM response for a new
game — reusing the Name-step prefetch when possible, falling back to a
retry-wrapped call otherwise. The confirm handler runs the whole thing
under ``asyncio.wait_for(..., WORLD_INIT_DEADLINE_S)`` so a wedged
provider can't freeze "Begin" for many minutes (the retry layers
underneath — validation × llm_text × complete — otherwise stack out to
tens of minutes against the 300 s httpx read timeout).

These tests pin:
  * the prefetch-reuse happy path,
  * the no-prefetch fallback path,
  * cancellation-safety when the outer deadline fires — the in-flight
    prefetch task must be cancelled (not leaked) and the cancellation
    must propagate as a TimeoutError, NOT be swallowed into a silent
    fresh call that would blow the budget again.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from lucidium.api.handlers import _acquire_world_init
from lucidium.config import WORLD_INIT_DEADLINE_S

_VALID_WORLD_INIT = json.dumps(
    {
        "game_name": "Test World",
        "opening_node": {"beats": [{"text": "The story opens."}]},
    }
)

_PROMPT = [{"role": "user", "content": "world_init"}]


class _FakeSession:
    """Minimal stand-in exposing only what ``_acquire_world_init`` and
    its fallback (``call_llm_json_with_retry`` → ``llm_text``) touch."""

    def __init__(self, *, llm_reply: str = _VALID_WORLD_INIT) -> None:
        self._world_init_task: asyncio.Task | None = None
        self._llm_reply = llm_reply
        self.llm_calls = 0

    async def llm_text(self, prompt, *, max_tokens=None):
        self.llm_calls += 1
        return self._llm_reply, []


def test_deadline_default_is_one_read_window_plus_margin() -> None:
    # Sanity-guard the constant: it must clear one 300 s httpx read
    # window so a single legit attempt never gets clipped, but stay
    # far below the multi-minute retry pile-up it exists to cap.
    assert 300.0 < WORLD_INIT_DEADLINE_S <= 600.0


@pytest.mark.asyncio
async def test_acquire_reuses_valid_prefetch() -> None:
    """A prefetch that resolved to valid JSON is reused as-is — no
    fresh LLM call — and the task handle is cleared."""
    session = _FakeSession()

    async def _prefetch() -> str:
        return _VALID_WORLD_INIT

    session._world_init_task = asyncio.ensure_future(_prefetch())

    init = await _acquire_world_init(session, _PROMPT)

    assert init.game_name == "Test World"
    assert session.llm_calls == 0  # prefetch reused, no fallback call
    assert session._world_init_task is None


@pytest.mark.asyncio
async def test_acquire_falls_back_when_no_prefetch() -> None:
    """No prefetch (cancelled / resumed save) → a fresh retry-wrapped
    call fires and its result is parsed."""
    session = _FakeSession()
    assert session._world_init_task is None

    init = await _acquire_world_init(session, _PROMPT)

    assert init.game_name == "Test World"
    assert session.llm_calls == 1


@pytest.mark.asyncio
async def test_deadline_cancels_prefetch_and_raises_timeout() -> None:
    """When the outer deadline fires mid-prefetch, the acquisition
    cancels the in-flight prefetch task and propagates cancellation as
    a TimeoutError — it must NOT swallow it into a silent fresh call."""
    session = _FakeSession()

    async def _hang() -> str:
        await asyncio.Event().wait()  # never resolves
        return _VALID_WORLD_INIT

    task = asyncio.ensure_future(_hang())
    session._world_init_task = task

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_acquire_world_init(session, _PROMPT), timeout=0.05)

    # The prefetch was cancelled (not leaked) and the handle cleared.
    assert session._world_init_task is None
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    # No fallback call was fired — the deadline aborted cleanly.
    assert session.llm_calls == 0


@pytest.mark.asyncio
async def test_independently_cancelled_prefetch_falls_back() -> None:
    """A prefetch that was cancelled elsewhere (e.g. a race) — as
    opposed to the deadline cancelling us — must fall back to a fresh
    call rather than surfacing the cancellation."""
    session = _FakeSession()

    async def _hang() -> str:
        await asyncio.Event().wait()
        return _VALID_WORLD_INIT

    task = asyncio.ensure_future(_hang())
    await asyncio.sleep(0)  # let it start
    task.cancel()
    session._world_init_task = task

    init = await _acquire_world_init(session, _PROMPT)

    assert init.game_name == "Test World"
    assert session.llm_calls == 1
    assert session._world_init_task is None
