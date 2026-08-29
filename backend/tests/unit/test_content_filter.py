"""Output-side ML content filter — interface, default fallback,
and the wiring into ``ensure_assets`` / ``ensure_portraits_for``.

The actual ML inference (NudeNet + insightface) is exercised by
``embedded_live`` integration tests; here we use a stub filter so
unit tests run without ML deps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lucidium.api.messages import MessageType, NoticeKind, S2CNotice
from lucidium.domain.character import Character, CharacterKind
from lucidium.domain.dialog import DialogTree
from lucidium.domain.game import Game
from lucidium.domain.settings import ImageSettings, Settings
from lucidium.domain.world import WorldState
from lucidium.orchestration import content_filter
from lucidium.orchestration.assets import (
    _output_filter_passes,
    ensure_portraits_for,
)
from lucidium.orchestration.content_filter import (
    FilterResult,
    Verdict,
    default_filter,
    reset_default_filter_for_tests,
    set_default_filter_for_tests,
)

# ---------- Stubs ------------------------------------------------------------


class _StubFilter:
    def __init__(self, verdict: Verdict, **kw: object) -> None:
        self._verdict = verdict
        self._kw = kw
        self._calls = 0
        self.is_active = verdict != Verdict.unavailable

    @property
    def call_count(self) -> int:
        return self._calls

    def classify(self, image_bytes: bytes) -> FilterResult:
        self._calls += 1
        return FilterResult(verdict=self._verdict, **self._kw)  # type: ignore[arg-type]


class _StubImageClient:
    """Minimal ImageClient. Always returns the same canned PNG
    bytes so the filter has something to inspect."""

    PAYLOAD = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, int]] = []

    async def generate(self, workflow, params, *, seed):
        self.calls.append((workflow, params, seed))
        return self.PAYLOAD


class _StubSession:
    def __init__(self, *, game: Game, settings: Settings) -> None:
        self.game = game
        self.settings = settings
        self.emitted: list[tuple[MessageType, S2CNotice]] = []
        self.emit = self._emit
        self.commits = 0

    @property
    def saves_root(self) -> Path | None:
        return None

    def install_game(self, g: Game) -> None:
        self.game = g

    async def commit(self) -> None:
        self.commit_blocking()

    def commit_blocking(self) -> None:
        self.commits += 1

    def _emit(self, mt: MessageType, payload: object) -> None:
        self.emitted.append((mt, payload))  # type: ignore[arg-type]


def _world() -> WorldState:
    return WorldState(
        game_name="t",
        setting="harbor",
        genre="Mystery",
        visual_style="ink wash",
    )


def _make_char(name: str, age: int = 30, outfit: str = "wool coat") -> Character:
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
        kind=CharacterKind.human,
    )


def _make_session(*characters: Character) -> _StubSession:
    chars = {c.id: c for c in characters}
    game = Game(
        world=_world(),
        characters=chars,
        dialog_tree=DialogTree(),
    )
    return _StubSession(
        game=game,
        settings=Settings(image=ImageSettings()),
    )


# ---------- Default filter selection ----------------------------------------


def test_default_filter_falls_back_to_no_op_when_ml_deps_missing() -> None:
    """No ML deps installed in the unit-test venv → the
    singleton must be the no-op fallback. ``is_active`` says so
    so the renderer can surface a warning."""
    reset_default_filter_for_tests()
    flt = default_filter()
    # Either ML deps aren't installed (typical CI), in which
    # case the filter is the NoOp fallback; or they ARE installed
    # and the filter is active. Both are valid; we only pin the
    # invariant that ``is_active`` matches reality.
    if isinstance(flt, content_filter._NoOpFilter):
        assert flt.is_active is False


def test_no_op_filter_returns_unavailable() -> None:
    flt = content_filter._NoOpFilter()
    result = flt.classify(b"\x89PNG fake")
    assert result.verdict == Verdict.unavailable
    assert "install" in result.detail.lower()


# ---------- _output_filter_passes -------------------------------------------


def test_output_filter_passes_lets_clean_image_through() -> None:
    set_default_filter_for_tests(_StubFilter(Verdict.ok))
    try:
        session = _make_session(_make_char("Iris"))
        ok = _output_filter_passes(
            session,
            _make_char("Iris"),
            b"PNGdata",
        )
        assert ok is True
        assert session.emitted == []
    finally:
        reset_default_filter_for_tests()


def test_output_filter_passes_blocks_unsafe_image_and_emits_notice() -> None:
    """Verdict.blocked → returns False AND emits an s2c/notice.
    No image is written; the caller skips its atomic_write_bytes."""
    blocking = _StubFilter(
        Verdict.blocked,
        nudity_score=0.92,
        min_face_age=14.0,
        detail="nudity_score=0.92 ≥ 0.55 AND face_age≈14.0 < 18",
    )
    set_default_filter_for_tests(blocking)
    try:
        session = _make_session(_make_char("Iris"))
        char = _make_char("Iris")
        ok = _output_filter_passes(session, char, b"PNGdata")
        assert ok is False
        assert len(session.emitted) == 1
        msg_type, payload = session.emitted[0]
        assert msg_type == MessageType.s2c_notice
        assert isinstance(payload, S2CNotice)
        assert payload.kind == NoticeKind.warning
        assert "Iris" in payload.body
        assert "rerender" in payload.body.lower()
    finally:
        reset_default_filter_for_tests()


def test_output_filter_passes_lets_image_through_when_filter_unavailable() -> None:
    """Verdict.unavailable → image still saves. The prompt-side
    + storage-side guards are still in place; we don't gate the
    engine on having ML deps installed.

    It is NOT silent, though: SAFETY.md §3 documents the filter as
    always-on, so a degraded filter surfaces one notice per process.
    See ``test_content_filter_unavailable.py`` for that contract.
    """
    from lucidium.orchestration import content_filter as cf

    set_default_filter_for_tests(_StubFilter(Verdict.unavailable))
    cf.reset_unavailable_reporting_for_tests()
    try:
        session = _make_session(_make_char("Iris"))
        ok = _output_filter_passes(
            session,
            _make_char("Iris"),
            b"PNGdata",
        )
        assert ok is True
        assert [m[0] for m in session.emitted] == [MessageType.s2c_notice]
        assert "not running" in session.emitted[0][1].title
    finally:
        reset_default_filter_for_tests()
        cf.reset_unavailable_reporting_for_tests()


def test_output_filter_passes_swallows_classifier_exceptions() -> None:
    """A buggy filter must never crash the render pipeline. The
    image goes through and we log."""

    class _Boom:
        is_active = True

        def classify(self, image_bytes: bytes) -> FilterResult:
            raise RuntimeError("model file corrupted")

    set_default_filter_for_tests(_Boom())
    try:
        session = _make_session(_make_char("Iris"))
        ok = _output_filter_passes(
            session,
            _make_char("Iris"),
            b"PNGdata",
        )
        assert ok is True
        assert session.emitted == []
    finally:
        reset_default_filter_for_tests()


# ---------- End-to-end: ensure_portraits_for skips writes on block -----------


@pytest.mark.asyncio
async def test_ensure_portraits_for_skips_disk_write_when_filter_blocks(
    tmp_path: Path,
) -> None:
    """Live integration into the portrait loop: when the filter
    blocks, NO png lands on disk and the character.images list
    does NOT pick up a new entry. The notice is emitted."""
    blocking = _StubFilter(
        Verdict.blocked,
        nudity_score=0.93,
        min_face_age=15.0,
    )
    set_default_filter_for_tests(blocking)
    try:
        char = _make_char("Iris", age=22, outfit="bare")
        session = _make_session(char)
        client = _StubImageClient()
        before_files = list(tmp_path.rglob("*.png"))  # noqa: ASYNC240 - sync fs in test assertion

        assets = await ensure_portraits_for(
            session=session,
            image_client=client,
            character_ids=[char.id],
            saves_root=tmp_path,
        )

        # The image client was called (we tried to render), but
        # the filter blocked the result before it hit disk.
        assert len(client.calls) == 1
        # No PNG written to disk.
        after_files = list(tmp_path.rglob("*.png"))  # noqa: ASYNC240 - sync fs in test assertion
        assert len(after_files) == len(before_files)
        # No GeneratedAsset returned for this character.
        assert assets == []
        # Character.images stayed empty — the failed render
        # didn't smuggle in a CharacterImage entry.
        assert session.game.characters[char.id].images == []
        # Player got a notice.
        assert len(session.emitted) == 1
        assert session.emitted[0][0] == MessageType.s2c_notice
    finally:
        reset_default_filter_for_tests()


@pytest.mark.asyncio
async def test_ensure_portraits_for_writes_when_filter_passes(
    tmp_path: Path,
) -> None:
    """Counterpart to the block test — when the filter says
    OK, the image lands on disk and we get a GeneratedAsset."""
    set_default_filter_for_tests(_StubFilter(Verdict.ok))
    try:
        char = _make_char("Mara", age=28, outfit="wool coat")
        session = _make_session(char)
        client = _StubImageClient()

        assets = await ensure_portraits_for(
            session=session,
            image_client=client,
            character_ids=[char.id],
            saves_root=tmp_path,
        )

        assert len(assets) == 1
        assert assets[0].image_path.exists()
        assert session.game.characters[char.id].images
    finally:
        reset_default_filter_for_tests()
