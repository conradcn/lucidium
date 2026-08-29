"""When an under-18 character is described with a nudity outfit,
the engine MUST NOT render them.

Two layers of defence:

  1. ``age_band`` floors any age below 18 at 18 inside the SDXL
     prompt builder. Pinned in ``test_age_band.py``.

  2. ``check_minor_nudity_and_correct`` scans the live game state
     before any portrait prompt is built. When a human character
     is under 18 AND their outfit substring matches the nudity
     pattern (``_nudity_prefix`` would emit ``"nude"`` or
     ``"mostly nude"``), the function:

       - bumps the stored ``age`` to 18,
       - ``install_game``s the revised game,
       - emits ``s2c/notice`` so the renderer pops a modal,
       - fires a fire-and-forget retcon to rewrite past
         narration that referenced the prior age.

This file pins layer 2.
"""

from __future__ import annotations

import asyncio

import pytest

from lucidium.api.messages import MessageType, NoticeKind, S2CNotice
from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import DialogTree
from lucidium.domain.game import Game
from lucidium.domain.world import WorldState
from lucidium.orchestration.assets import check_minor_nudity_and_correct


class _StubSession:
    """Minimal session shape for the safety scan: ``game``,
    ``install_game``, ``commit``, and ``emit``. The retcon path
    is fire-and-forget — it pulls a running event loop and skips
    cleanly when there isn't one (we test in synchronous mode)."""

    def __init__(self, *, game: Game) -> None:
        self.game = game
        self.commits = 0
        self.emitted: list[tuple[MessageType, S2CNotice]] = []
        self.emit = self._emit

    def install_game(self, g: Game) -> None:
        self.game = g

    async def commit(self) -> None:
        self.commit_blocking()

    def commit_blocking(self) -> None:
        self.commits += 1

    def _emit(self, mt: MessageType, payload: object) -> None:
        # The safety code only emits S2CNotice via this stub; cast
        # is fine.
        self.emitted.append((mt, payload))  # type: ignore[arg-type]


def _world() -> WorldState:
    return WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink wash",
    )


def _make_char(
    *,
    name: str,
    age: int,
    outfit: str,
    kind: CharacterKind = CharacterKind.human,
) -> Character:
    return Character(
        name=name,
        description=f"the {name}",
        gender="female",
        age=age,
        ethnicity="local",
        skin="pale",
        hair_color="black",
        hairstyle="short",
        eye_color="brown",
        build="slight",
        bust="moderate",
        outfit=outfit,
        pose="standing",
        expression="neutral",
        seed=1,
        kind=kind,
    )


def _make_session(*characters: Character) -> _StubSession:
    chars = {c.id: c for c in characters}
    game = Game(world=_world(), characters=chars, dialog_tree=DialogTree())
    return _StubSession(game=game)


def test_minor_with_nudity_outfit_gets_age_bumped() -> None:
    minor = _make_char(name="Iris", age=14, outfit="nude")
    session = _make_session(minor)

    corrected = check_minor_nudity_and_correct(session)

    assert corrected == [minor.id]
    assert session.game.characters[minor.id].age == 18
    # commit fires so the bump persists.
    assert session.commits == 1


def test_minor_with_clothed_outfit_is_left_alone() -> None:
    minor = _make_char(name="Iris", age=14, outfit="wool coat")
    session = _make_session(minor)

    corrected = check_minor_nudity_and_correct(session)

    assert corrected == []
    assert session.game.characters[minor.id].age == 14
    assert session.commits == 0
    assert session.emitted == []


def test_adult_with_nudity_outfit_is_left_alone() -> None:
    adult = _make_char(name="Mara", age=28, outfit="nude")
    session = _make_session(adult)

    corrected = check_minor_nudity_and_correct(session)

    assert corrected == []
    assert session.game.characters[adult.id].age == 28


def test_minor_with_mostly_nude_outfit_is_corrected() -> None:
    """Lingerie / towel / bikini also trigger the SDXL ``mostly
    nude`` prefix and should also flip the age guard."""
    minor = _make_char(name="Iris", age=15, outfit="lingerie")
    session = _make_session(minor)

    corrected = check_minor_nudity_and_correct(session)

    assert corrected == [minor.id]
    assert session.game.characters[minor.id].age == 18


def test_safety_correction_emits_s2c_notice() -> None:
    minor = _make_char(name="Iris", age=14, outfit="naked")
    session = _make_session(minor)

    check_minor_nudity_and_correct(session)

    assert len(session.emitted) == 1
    msg_type, payload = session.emitted[0]
    assert msg_type == MessageType.s2c_notice
    assert isinstance(payload, S2CNotice)
    assert payload.kind == NoticeKind.warning
    assert "Iris" in payload.body
    # Player-facing copy explains what happened and that the
    # narration is being rewritten.
    assert "18" in payload.body
    assert "rewrite" in payload.body.lower() or "narration" in payload.body.lower()


def test_safety_correction_is_idempotent() -> None:
    minor = _make_char(name="Iris", age=14, outfit="nude")
    session = _make_session(minor)

    first = check_minor_nudity_and_correct(session)
    assert first == [minor.id]
    second = check_minor_nudity_and_correct(session)
    # Already age=18 → no further correction.
    assert second == []
    # Only one notice / one commit; the second call is a no-op.
    assert len(session.emitted) == 1
    assert session.commits == 1


def test_nonhuman_minor_with_nudity_is_not_treated_as_human_minor() -> None:
    """Nonhuman characters route through a different prompt path
    (freeform physical_description on the environment SDXL
    pipeline) and the human-anatomy age semantics don't apply.
    The structured-anatomy nudity guard is a HUMAN-pipeline guard;
    leave nonhumans alone here."""
    nonhuman = _make_char(
        name="Pip",
        age=4,
        outfit="nude",
        kind=CharacterKind.nonhuman,
    )
    session = _make_session(nonhuman)

    corrected = check_minor_nudity_and_correct(session)

    assert corrected == []
    assert session.game.characters[nonhuman.id].age == 4


def test_multiple_minors_are_all_corrected_in_one_pass() -> None:
    a = _make_char(name="Iris", age=12, outfit="nude")
    b = _make_char(name="Cal", age=15, outfit="lingerie")
    safe = _make_char(name="Mara", age=28, outfit="wool coat")
    session = _make_session(a, b, safe)

    corrected = check_minor_nudity_and_correct(session)

    assert set(corrected) == {a.id, b.id}
    assert session.game.characters[a.id].age == 18
    assert session.game.characters[b.id].age == 18
    assert session.game.characters[safe.id].age == 28
    assert len(session.emitted) == 2


@pytest.mark.asyncio
async def test_safety_correction_inside_running_loop_schedules_retcon() -> None:
    """When called from inside a running event loop, the function
    schedules the retcon as a background task. We don't await the
    retcon here (it would touch the LLM client); we just confirm
    the immediate state mutation + notice still fire and the
    function returns synchronously without blocking."""
    minor = _make_char(name="Iris", age=14, outfit="nude")
    session = _make_session(minor)

    corrected = check_minor_nudity_and_correct(session)
    # Give the event loop one tick so any synchronous portion of
    # the spawned task can run; the actual LLM call would error
    # without a real client and is logged-and-swallowed.
    await asyncio.sleep(0)

    assert corrected == [minor.id]
    assert session.game.characters[minor.id].age == 18
    assert len(session.emitted) == 1
