"""The real play handlers serialise on the session's play lock.

The bug: ``c2s/play/free_text`` submitted while ``c2s/play/advance``
was awaiting speculation produced a "skip backwards" symptom in the
renderer. Both handlers would call ``install_game`` concurrently;
whichever finished last won, which could be the OLDER premise's
chain — so the dialog appeared to skip back to a beat the player
had already moved past.

Fix: a per-session ``_play_lock`` (``asyncio.Lock``) guards
``play_advance`` / ``play_free_text`` / ``play_undo``. The second
handler in flight blocks until the first releases.

These tests drive the ACTUAL handler coroutines concurrently against a
fake session, with the three lock-protected bodies replaced by
instrumented stand-ins that record when each one enters and leaves.
Removing the lock from any one of the three handlers makes
``max_inside`` reach 2 and the first test fail.
"""

from __future__ import annotations

import asyncio

import pytest

from lucidium.api import handlers
from lucidium.api.handlers import (
    HandlerContext,
    play_advance_handler,
    play_free_text_handler,
    play_undo_handler,
)
from lucidium.api.messages import C2SPlayAdvance, C2SPlayFreeText
from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import DialogNode, DialogNodeState, DialogTree
from lucidium.domain.game import Game
from lucidium.domain.settings import Settings
from lucidium.domain.world import WorldState
from lucidium.orchestration.session import Session

# How long each instrumented body holds the lock. Long enough that a
# handler which failed to take the lock would demonstrably overlap the
# one that did, short enough to keep the suite fast.
_HOLD_S = 0.05


# ---------------------------------------------------------------------------
# fake session + game
# ---------------------------------------------------------------------------


class _Latency:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def record(self, _phase: str, **fields) -> None:
        self.records.append(fields)


class _RenderScheduler:
    def __init__(self) -> None:
        self.turns = 0

    def on_turn(self) -> None:
        self.turns += 1


class _FakeSession:
    """The slice of ``Session`` the three play handlers touch.

    Notably it uses the REAL ``Session._ensure_play_lock`` — the lock
    under test is the production one, lazily bound to the running loop.
    """

    _ensure_play_lock = Session._ensure_play_lock

    def __init__(self, game: Game) -> None:
        self.game = game
        self.settings = Settings()
        self._play_lock: asyncio.Lock | None = None
        self.undo_stack: list[Game] = [game.model_copy(deep=True)]
        self._foreground_task = None
        self._foreground_stream_task = None
        self._speculative_tasks: dict[str, asyncio.Task] = {}
        self.latency = _Latency()
        self.render_scheduler = _RenderScheduler()
        self.installs: list[Game] = []

    def install_game(self, game: Game) -> None:
        self.installs.append(game)
        self.game = game

    def invalidate_speculation_from(self, _root_id: str) -> list[str]:
        return []

    async def commit(self) -> None:
        return None


def _world() -> WorldState:
    return WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink wash",
    )


def _player() -> Character:
    return Character(
        name="Iris",
        description="archivist",
        gender="female",
        age=28,
        ethnicity="local",
        skin="pale",
        hair_color="auburn",
        hairstyle="braid",
        eye_color="grey",
        build="slight",
        bust="moderate",
        outfit="wool coat",
        pose="standing",
        expression="alert",
        seed=1,
        is_player=True,
        kind=CharacterKind.human,
    )


def _game() -> Game:
    """A two-beat tree: the player sits on ``n1``, and ``n2`` is a
    pre-generated child ``advance`` can walk into without an LLM call
    (its in-tree HIT path, which does all its work under the lock)."""
    parent = DialogNode(
        id="n1",
        text="the lamp gutters",
        state=DialogNodeState.committed,
        premise_hash="a" * 64,
    )
    child = DialogNode(
        id="n2",
        text="the door opens",
        parent_id="n1",
        chosen_option_id=None,
        state=DialogNodeState.committed,
        premise_hash="b" * 64,
    )
    tree = DialogTree(
        nodes={parent.id: parent, child.id: child},
        root_id=parent.id,
        committed_path=[parent.id],
    )
    pc = _player()
    return Game(
        world=_world(),
        characters={pc.id: pc},
        dialog_tree=tree,
        current_node_id=parent.id,
    )


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------


class _Tracker:
    """Records overlap across the three instrumented handler bodies."""

    def __init__(self) -> None:
        self.inside = 0
        self.max_inside = 0
        self.enter_order: list[str] = []
        self.exit_order: list[str] = []

    async def section(self, label: str) -> None:
        """Occupy the caller's lock-protected body for ``_HOLD_S``."""
        self.inside += 1
        self.max_inside = max(self.max_inside, self.inside)
        self.enter_order.append(label)
        try:
            await asyncio.sleep(_HOLD_S)
        finally:
            self.exit_order.append(label)
            self.inside -= 1


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch) -> _Tracker:
    """Replace each handler's lock-protected body with a tracked sleep.

    ``advance`` and ``free_text`` are instrumented at the point where
    they mutate the tree; ``undo`` is instrumented through the session's
    ``commit``, which it awaits while holding the lock. The lock
    acquisition itself is left entirely to production code.
    """
    tr = _Tracker()

    async def _walk_existing_child(_ctx, _node):
        await tr.section("advance")
        yield ("advance-beat", None)

    async def _generate_and_walk_chain(_ctx, **_kwargs):
        await tr.section("free_text")
        yield ("free-text-beat", None)

    # Pin advance onto its in-tree HIT path. Whether ``n2`` is still a
    # valid child depends on whether free_text's invalidation got there
    # first, and this file is about the lock, not about obsolescence.
    monkeypatch.setattr(
        handlers.obsolescence,
        "first_valid_child",
        lambda _game, _parent_id, *, chosen_option_id=None: "n2",
    )
    monkeypatch.setattr(handlers, "_walk_existing_child", _walk_existing_child)
    monkeypatch.setattr(handlers, "_generate_and_walk_chain", _generate_and_walk_chain)
    monkeypatch.setattr(handlers, "_ensure_speculative_branches", lambda _s: None)
    monkeypatch.setattr(handlers, "_full_state_message", lambda _s: None)
    return tr


async def _drain(result) -> list:
    return [message async for message in result]


async def _run_advance(session: _FakeSession) -> list:
    ctx = HandlerContext(session=session)  # type: ignore[arg-type]
    return await _drain(await play_advance_handler(C2SPlayAdvance(option_id=None), ctx))


async def _run_free_text(session: _FakeSession) -> list:
    ctx = HandlerContext(session=session)  # type: ignore[arg-type]
    return await _drain(await play_free_text_handler(C2SPlayFreeText(text="I look away"), ctx))


async def _run_undo(session: _FakeSession, tracker: _Tracker) -> list:
    ctx = HandlerContext(session=session)  # type: ignore[arg-type]
    # ``undo``'s only await inside the lock is ``session.commit()``.
    session.commit = lambda: tracker.section("undo")  # type: ignore[assignment]
    return await _drain(await play_undo_handler(None, ctx))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_play_handlers_do_not_overlap(tracker: _Tracker) -> None:
    """advance / free_text / undo dispatched together on one session run
    strictly one at a time. This is the regression guard: drop the lock
    from ANY of the three and two bodies are inside at once."""
    session = _FakeSession(_game())

    await asyncio.gather(
        _run_advance(session),
        _run_free_text(session),
        _run_undo(session, tracker),
    )

    assert tracker.max_inside == 1, (
        f"play handlers overlapped (enter order {tracker.enter_order}) — "
        "the play lock did not serialise them"
    )
    assert sorted(tracker.enter_order) == ["advance", "free_text", "undo"]
    # Each handler ran its body to completion before the next started.
    assert tracker.enter_order == tracker.exit_order


@pytest.mark.asyncio
async def test_advance_and_free_text_serialise_pairwise(tracker: _Tracker) -> None:
    """The exact pair from the bug report: a free-text submitted while
    an advance is mid-flight waits for it instead of racing it into
    ``install_game``."""
    session = _FakeSession(_game())

    advance = asyncio.create_task(_run_advance(session))
    # Let advance reach its lock-protected body before free_text is
    # dispatched — the interleaving the player actually produced.
    await asyncio.sleep(_HOLD_S / 5)
    free_text = asyncio.create_task(_run_free_text(session))
    await asyncio.gather(advance, free_text)

    assert tracker.max_inside == 1
    assert tracker.enter_order == ["advance", "free_text"]
    assert tracker.exit_order == ["advance", "free_text"]


@pytest.mark.asyncio
async def test_repeated_advances_queue_rather_than_interleave(
    tracker: _Tracker,
) -> None:
    """A player mashing Continue produces several advances in flight at
    once; they queue on the lock instead of stacking tree mutations."""
    session = _FakeSession(_game())

    await asyncio.gather(*(_run_advance(session) for _ in range(4)))

    assert tracker.max_inside == 1
    assert tracker.enter_order == ["advance"] * 4


@pytest.mark.asyncio
async def test_play_lock_is_per_session_not_shared(tracker: _Tracker) -> None:
    """Two sessions must not serialise against each other. Asserted by
    BUILDING both locks and holding one: work on the other session runs
    to completion while the first is held, and the two lock objects are
    distinct."""
    held = _FakeSession(_game())
    other = _FakeSession(_game())

    held_lock = held._ensure_play_lock()
    other_lock = other._ensure_play_lock()
    assert held_lock is not other_lock
    # Lazily built, then stable: a handler must not mint a fresh lock
    # per call, or nothing would ever contend.
    assert held._ensure_play_lock() is held_lock

    await held_lock.acquire()
    try:
        assert held_lock.locked()
        # A play handler on the OTHER session completes while the first
        # session's lock is held — it never touches ``held_lock``.
        await asyncio.wait_for(_run_advance(other), timeout=2.0)
    finally:
        held_lock.release()

    assert tracker.enter_order == ["advance"]
    assert not other_lock.locked()


@pytest.mark.asyncio
async def test_play_lock_is_lazily_built() -> None:
    """Constructed at first use, not at ``Session.__init__`` — a real
    ``Session`` is built outside an event loop in dozens of tests, and
    an eagerly-created ``asyncio.Lock`` would bind to the wrong loop."""
    session = Session()
    assert session._play_lock is None
    lock = session._ensure_play_lock()
    assert isinstance(lock, asyncio.Lock)
    assert session._play_lock is lock
