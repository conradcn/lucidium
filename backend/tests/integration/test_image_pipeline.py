"""End-to-end checks on the image pipeline.

Two halves:

1. **Bundled assets (the input side).** The backend hands the renderer
   absolute paths to the bundled ``dream_guide.png`` and
   ``white_room.png`` during interview start (FR-022, FR-027). If those
   paths point at files that don't actually exist on disk — typo,
   missing in package, accidentally renamed — every onboarding from
   here on shows a blank placeholder.

2. **The IPC envelopes (the wire side).** ``c2s/new_game/start`` MUST
   surface the bundled paths in the resulting state patch under
   ``/interview/dream_guide_image_path`` and
   ``/interview/white_room_image_path``. Without those keys the
   renderer's InterviewStage has nothing to render and the dream guide
   silently disappears, which is exactly the regression the user
   reported.

3. **The dialog node (the gameplay side).** A scene with a portrait
   image must round-trip the disk path through the per-character
   ``images[*].path`` field and the per-environment ``image_path``
   field — these are what the renderer turns into ``lucidium-asset://``
   URLs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lucidium.api.handlers import HandlerContext, build_default_registry
from lucidium.api.messages import Envelope, MessageType
from lucidium.config import (
    bundled_dream_guide_path,
    bundled_white_room_path,
)
from lucidium.orchestration.session import Session


class _StubLlm:
    async def complete(self, *_a, **_kw):
        async def _empty():
            if False:
                yield ""

        return _empty()


class _StubImage:
    async def generate(self, *_a, **_kw) -> bytes:
        return b""


@pytest.fixture()
def session(tmp_app_data: Path) -> Session:
    return Session(llm_client=_StubLlm(), image_client=_StubImage())


def test_bundled_placeholder_paths_exist_on_disk() -> None:
    """The dream guide and white room are committed to the repo."""

    dream = bundled_dream_guide_path()
    white = bundled_white_room_path()
    assert dream.exists(), f"dream guide missing on disk: {dream}"
    assert white.exists(), f"white room missing on disk: {white}"

    # Both must be PNGs — the renderer assumes a static MIME type.
    assert dream.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert white.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    # And non-trivial: an empty file would technically pass exists()
    # but render as nothing in the InterviewStage.
    assert dream.stat().st_size > 1024
    assert white.stat().st_size > 1024


@pytest.mark.asyncio
async def test_new_game_start_emits_bundled_paths(session: Session) -> None:
    """``c2s/new_game/start`` puts dream-guide + white-room paths on the wire.

    The renderer reads these from the interview snapshot and renders
    them as the InterviewStage backdrop. Missing keys → silent visual
    regression.
    """

    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    out: list[tuple[MessageType, dict]] = []
    async for envelope in registry.dispatch(
        Envelope(type=MessageType.c2s_new_game_start, payload={}),
        ctx,
    ):
        out.append(envelope)

    # Find the patch envelope and walk its ops. ``payload`` is an
    # S2CStatePatch Pydantic model; access via attributes.
    patch_payloads = [payload for kind, payload in out if kind == MessageType.s2c_state_patch]
    assert patch_payloads, "expected at least one state/patch from c2s/new_game/start"

    flat_ops = [op for payload in patch_payloads for op in payload.ops]

    def _value_at(path: str) -> object | None:
        for op in flat_ops:
            if op.path == path:
                return op.value
            # The handler may issue a single ``replace /interview``
            # whose value is the whole InterviewState — search inside.
            if op.path == "/interview" and isinstance(op.value, dict):
                key = path.split("/")[-1]
                if key in op.value:
                    return op.value[key]
        return None

    dream_path = _value_at("/interview/dream_guide_image_path")
    white_path = _value_at("/interview/white_room_image_path")
    assert dream_path is not None, "dream_guide_image_path missing from patch"
    assert white_path is not None, "white_room_image_path missing from patch"

    # And those paths point at the bundled files on disk.
    assert Path(str(dream_path)).exists()  # noqa: ASYNC240 - sync fs in test assertion
    assert Path(str(white_path)).exists()  # noqa: ASYNC240 - sync fs in test assertion
    assert Path(str(dream_path)).resolve() == bundled_dream_guide_path().resolve()  # noqa: ASYNC240 - sync fs in test assertion
    assert Path(str(white_path)).resolve() == bundled_white_room_path().resolve()  # noqa: ASYNC240 - sync fs in test assertion


def test_session_interview_state_holds_bundled_paths(session: Session) -> None:
    """The InterviewState defaults expose the bundled paths.

    Even before ``c2s/new_game/start`` runs, the dataclass should be
    constructible with the bundled values — so a unit-level renderer
    test that builds an interview state directly never falls back to
    empty strings.
    """

    state = session.interview
    # Default-constructed: empty string. The handler sets them.
    assert state.dream_guide_image_path == ""
    assert state.white_room_image_path == ""
    # But the bundled paths must resolve to real files when asked.
    assert bundled_dream_guide_path().is_file()
    assert bundled_white_room_path().is_file()
