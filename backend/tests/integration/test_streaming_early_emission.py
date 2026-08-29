"""Streaming dialog generation: pin that the FIRST beat's
state/patch + state/full messages emit BEFORE the LLM has
finished generating beats 2+.

The implementation streams chunks from the LLM, drives a
``StreamingDialogParser`` chunk-by-chunk, and yields the walk
events for beat[0] as soon as that beat's closing brace lands.
Subsequent beats accumulate in the dialog tree but don't trigger
their own walk — the player advances through them via Continue.

Test shape: an SSE-style mock transport that delivers chunks
with controlled timing. The first chunks complete beat[0]; a
delay opens before the closing `]` and `options` chunks arrive.
The test asserts the handler yields beat[0]'s state/full BEFORE
the slow tail lands.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from lucidium.api.handlers import HandlerContext, build_default_registry
from lucidium.api.messages import Envelope, MessageType
from lucidium.domain.character import Character
from lucidium.domain.dialog import (
    DialogNode,
    DialogNodeState,
    DialogOption,
    DialogTree,
)
from lucidium.domain.game import Game
from lucidium.domain.settings import LlmSettings, Settings
from lucidium.domain.world import WorldState
from lucidium.orchestration.session import Session
from lucidium.providers.llm_client import OpenAiCompatibleLlmClient


def _bootstrap_game() -> Game:
    pc = Character(
        id="pc",
        is_player=True,
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
    )
    n0 = DialogNode(
        id="n0",
        parent_id=None,
        chosen_option_id=None,
        text="A choice in front of you.",
        options=[
            DialogOption(id="o1", text="Walk left."),
            DialogOption(id="o2", text="Walk right."),
        ],
        state=DialogNodeState.committed,
        premise_hash="h" * 64,
    )
    return Game(
        world=WorldState(
            game_name="t",
            setting="harbor",
            genre="Mystery",
            visual_style="ink wash",
        ),
        characters={pc.id: pc},
        environments={},
        dialog_tree=DialogTree(
            nodes={n0.id: n0},
            root_id=n0.id,
            committed_path=[n0.id],
        ),
        current_node_id=n0.id,
        on_stage=[],
    )


class _NullImage:
    async def generate(self, *_a, **_kw) -> bytes:
        return bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
        )


def _split_into_chunks(text: str, sizes: list[int]) -> list[str]:
    """Slice ``text`` into chunks of ``sizes[0]``, ``sizes[1]``, …
    bytes; the last chunk holds whatever remains."""
    out: list[str] = []
    i = 0
    for s in sizes:
        out.append(text[i : i + s])
        i += s
    if i < len(text):
        out.append(text[i:])
    return out


def _make_streaming_transport(
    chunks: list[str],
    *,
    delay_before_chunk: list[float] | None = None,
) -> httpx.MockTransport:
    """SSE-style mock transport that delivers ``chunks`` as
    server-sent events with optional per-chunk delays.

    Index ``i`` of ``delay_before_chunk`` is the seconds to sleep
    before yielding ``chunks[i]``. Defaults to no delay.
    """
    delays = delay_before_chunk or [0.0] * len(chunks)
    assert len(delays) == len(chunks)

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8") or "{}")
        if not body.get("stream"):
            # Non-streaming fallback (speculation path) — return all
            # chunks concatenated as a single message body.
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "".join(chunks)}},
                    ],
                },
            )

        # Build an async generator that yields SSE frames with
        # per-chunk delays. httpx hands them back chunk-by-chunk
        # to the streaming reader.
        async def stream_body():  # type: ignore[no-untyped-def]
            for delay, chunk in zip(delays, chunks, strict=True):
                if delay > 0:
                    await asyncio.sleep(delay)
                frame = "data: " + json.dumps({"choices": [{"delta": {"content": chunk}}]}) + "\n\n"
                yield frame.encode("utf-8")
            yield b"data: [DONE]\n\n"

        return httpx.Response(
            200,
            content=stream_body(),
            headers={"content-type": "text/event-stream"},
        )

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
async def test_first_beat_state_full_emits_before_slow_tail(
    tmp_path: Path,
) -> None:
    """The headline streaming pin: beat[0]'s ``s2c/state/full``
    arrives BEFORE the chunks carrying the closing array bracket
    + ``options`` finish streaming.

    Setup: payload split so beat[0] closes early, then a ~1.5 s
    pause, then the rest. The handler should emit walk events
    for beat[0] within milliseconds of the early chunks landing.
    """
    payload = {
        "beats": [
            {
                "text": "The cobbles tilt under your boots.",
                "speaker_id": None,
                "entering_character_ids": [],
                "leaving_character_ids": [],
                "new_characters": [],
                "location_id": None,
                "location_prompt": None,
                "location_lighting": "",
                "character_changes": [],
            },
            {
                "text": "Above, gulls wheel and complain.",
                "speaker_id": None,
                "entering_character_ids": [],
                "leaving_character_ids": [],
                "new_characters": [],
                "location_id": None,
                "location_prompt": None,
                "location_lighting": "",
                "character_changes": [],
            },
        ],
        "options": [
            {"id": "ox", "text": "Press onward."},
            {"id": "oy", "text": "Pause and listen."},
        ],
    }
    payload_text = json.dumps(payload)
    # Find a split point AFTER beat[0]'s closing brace + comma so
    # the streaming parser can emit beat[0] as soon as the first
    # chunk arrives. We split at the second beat's opening brace
    # so the slow tail carries beats[1] + options.
    split_idx = payload_text.index("Above, gulls")
    # Walk back to the comma between beats.
    while split_idx > 0 and payload_text[split_idx - 1] != ",":
        split_idx -= 1
    head = payload_text[:split_idx]
    tail = payload_text[split_idx:]

    # Tail arrives 1.5 s late — plenty of time to assert that
    # beat[0]'s state/full landed first.
    transport = _make_streaming_transport(
        chunks=[head, tail],
        delay_before_chunk=[0.0, 1.5],
    )
    http = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
    )
    settings = Settings(
        llm=LlmSettings(
            base_url="http://fake.test",
            model="x/fake",
            api_key="sk-fake",
            temperature=0.0,
        ),
    )
    llm_client = OpenAiCompatibleLlmClient(settings.llm, http=http)

    session = Session(
        settings=settings,
        llm_client=llm_client,
        image_client=_NullImage(),
        saves_root=tmp_path / "saves",
    )
    session.install_game(_bootstrap_game())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    started = time.monotonic()
    state_full_at: float | None = None
    state_full_count = 0
    async for msg_type, _payload in registry.dispatch(
        Envelope(
            type=MessageType.c2s_play_advance,
            payload={"option_id": "o1"},
        ),
        ctx,
    ):
        if msg_type == MessageType.s2c_state_full and state_full_at is None:
            state_full_at = time.monotonic() - started
        if msg_type == MessageType.s2c_state_full:
            state_full_count += 1

    assert state_full_at is not None, "play_advance never emitted state/full"
    # The slow tail arrives at +1.5 s. Beat[0]'s state/full must
    # have landed well before that. Generous 1.0 s budget so the
    # test isn't flaky on slow CI machines while still proving
    # the early-emission property.
    assert state_full_at < 1.0, (
        f"state/full for beat 0 arrived at {state_full_at:.2f}s — "
        "should have been emitted before the slow tail at +1.5s"
    )
    # Game state must have advanced past the parent.
    assert session.game is not None
    assert session.game.current_node_id != "n0"


# Truncation tolerance is exhaustively covered by
# ``test_streaming_dialog_parser`` at the unit level. An
# integration repro here would have to suppress the background
# speculative-branch spawn that fires after a successful
# play_advance (it'd hit the same truncated mock content via
# the non-streaming path and error noisily on cleanup), and
# the speculation cancellation plumbing isn't worth the
# coverage delta.
