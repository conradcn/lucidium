"""Pin that EVERY beat in a streaming LLM chain is displayed as soon
as its closing ``}`` lands in the stream — not just beat 1.

The earlier streaming test (``test_streaming_early_emission``) only
checked the FIRST advance call. This test exercises the symptom the
user reported: while the foreground drain is still parsing later beats,
each subsequent Continue should walk into beat N as soon as beat N's
closing brace arrives, NOT wait until the full LLM response is done.

Setup: an SSE mock that delivers a 3-beat chain across multiple chunks
with controlled gaps between beats. The test paces Continue clicks so
each click lands BEFORE the matching beat has closed in the stream;
then asserts that each beat's display happens close to (not long after)
the moment the beat's closing brace lands."""

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
        # Smallest valid PNG.
        return bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
        )


def _three_beat_payload() -> dict:
    return {
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
            {
                "text": "A bell tolls in the distance.",
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


def _split_after_each_beat(text: str) -> list[str]:
    """Split the serialised JSON into chunks where each beat's
    closing ``}`` ends a chunk. Each beat object is at depth=2 (outer
    object brace is depth=1, beat objects inside beats[] are depth=2).
    """
    chunks: list[str] = []
    depth = 0
    in_string = False
    escape = False
    chunk_start = 0
    inside_beats_array = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            if inside_beats_array and depth == 2:
                # Beat object closed.
                chunks.append(text[chunk_start : i + 1])
                chunk_start = i + 1
            depth -= 1
        elif ch == "[" and depth == 1 and not inside_beats_array:
            inside_beats_array = True
        elif ch == "]" and inside_beats_array and depth == 1:
            inside_beats_array = False
            # Everything after the beats array (the closing ``]`` plus
            # the trailing ``"options": [...]}``) is a single tail
            # chunk — splitting the options array would add more chunks
            # than the test needs.
            chunks.append(text[chunk_start:])
            return chunks
    if chunk_start < len(text):
        chunks.append(text[chunk_start:])
    return chunks


def _make_streaming_transport(
    chunks: list[str],
    *,
    delay_before_chunk: list[float] | None = None,
) -> httpx.MockTransport:
    delays = delay_before_chunk or [0.0] * len(chunks)
    assert len(delays) == len(chunks)

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8") or "{}")
        if not body.get("stream"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "".join(chunks)}},
                    ],
                },
            )

        async def stream_body():
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
async def test_each_continue_displays_when_its_beat_closes(
    tmp_path: Path,
) -> None:
    """The headline pin: across a 3-beat streaming chain, EACH
    Continue (the initial advance + the two follow-ups) is displayed
    within a tight budget of the moment its beat's closing ``}``
    landed in the stream — not after the full response finishes.
    """
    payload_text = json.dumps(_three_beat_payload())
    chunks = _split_after_each_beat(payload_text)
    # Expected structure: chunk[0] carries beat[0] + closing brace,
    # chunk[1] beat[1], chunk[2] beat[2], chunk[3] the trailing
    # ``],"options":[...]}`` tail.
    assert len(chunks) == 4, f"unexpected split: {[c[:40] for c in chunks]}"

    BEAT_GAP_S = 0.6
    # Big tail delay so we can detect "waited for full response".
    TAIL_GAP_S = 2.0
    delays = [0.0, BEAT_GAP_S, BEAT_GAP_S, TAIL_GAP_S]

    transport = _make_streaming_transport(
        chunks=chunks,
        delay_before_chunk=delays,
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
    session.latency.enable()
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    # Track per-dispatch timing: when click went in, when first
    # s2c/state/patch came out, what node we landed on.
    dispatch_results: list[dict] = []

    async def _do_advance(*, option_id: str | None) -> None:
        click_at = time.monotonic()
        patch_at: float | None = None
        node_landed: str | None = None
        payload = {"option_id": option_id}
        async for msg_type, msg_payload in registry.dispatch(
            Envelope(
                type=MessageType.c2s_play_advance,
                payload=payload,
            ),
            ctx,
        ):
            if msg_type == MessageType.s2c_state_patch and patch_at is None:
                patch_at = time.monotonic()
                # The patch carries the new current_node_id.
                for op in msg_payload.ops:
                    if op.path == "/current_node_id":
                        node_landed = op.value
                        break
        dispatch_results.append(
            {
                "click_at": click_at,
                "patch_at": patch_at,
                "node_landed": node_landed,
            }
        )

    # --- dispatch sequence ---------------------------------------------
    # 1) First Continue triggers LLM streaming. Beat 1 completes at
    #    ~0.0 s (first chunk arrives immediately).
    test_started = time.monotonic()
    await _do_advance(option_id="o1")
    # 2) Click Continue #2 immediately. Beat 2 won't close until
    #    ~BEAT_GAP_S after test start. Drain task should install it
    #    when the closing brace arrives and play_advance walks to it.
    await _do_advance(option_id=None)
    # 3) Click Continue #3 immediately. Beat 3 won't close until
    #    ~2 * BEAT_GAP_S. Same expectation.
    await _do_advance(option_id=None)

    # --- assertions ----------------------------------------------------
    for i, info in enumerate(dispatch_results):
        assert info["patch_at"] is not None, f"dispatch #{i + 1}: never emitted s2c/state/patch"
        assert info["node_landed"] is not None, (
            f"dispatch #{i + 1}: state/patch carried no current_node_id"
        )

    # Beat-ready timestamps from the latency audit identify the moment
    # the parser closed each beat.
    beat_ready_events = [e for e in session.latency.events if e.name == "beat_ready"]
    assert len(beat_ready_events) >= 3, (
        f"expected ≥3 beat_ready events, got {len(beat_ready_events)}: "
        f"{[(e.name, e.extra) for e in beat_ready_events]}"
    )

    # Per dispatch: the display (state/patch) must happen within a
    # small budget of the matching beat's beat_ready. The budget is
    # generous (250 ms) to keep this test stable on CI but tight
    # enough that "wait for full response" (≥ TAIL_GAP_S = 2 s) would
    # clearly violate it.
    BUDGET_MS = 250.0
    for i, info in enumerate(dispatch_results):
        # Find beat_ready for the landed node.
        match = next(
            (e for e in beat_ready_events if e.node_id == info["node_landed"]),
            None,
        )
        assert match is not None, (
            f"dispatch #{i + 1}: no beat_ready for landed node {info['node_landed']}"
        )
        delay_ms = (info["patch_at"] - match.ts) * 1000
        # Allow a small negative window (sub-ms) if patch fired first
        # in tests where beat_ready records inside an install path.
        assert delay_ms < BUDGET_MS, (
            f"dispatch #{i + 1}: state/patch landed "
            f"{delay_ms:.0f} ms AFTER its beat_ready — should display "
            f"within {BUDGET_MS:.0f} ms of beat close, not wait for "
            f"the full response"
        )

    # And final dispatch must NOT have waited for the full stream tail.
    # The tail arrives at ~test_started + 2*BEAT_GAP_S + TAIL_GAP_S
    # = +3.2 s. Last patch must land before that.
    last = dispatch_results[-1]
    elapsed_to_last_patch = last["patch_at"] - test_started
    tail_arrival = 2 * BEAT_GAP_S + TAIL_GAP_S
    assert elapsed_to_last_patch < tail_arrival - 0.3, (
        f"final dispatch state/patch landed at +{elapsed_to_last_patch:.2f} s "
        f"— close to the tail at +{tail_arrival:.2f} s (looks like it "
        f"waited for the full response)"
    )

    # Cleanup any in-flight drain task so the test exits cleanly.
    task = getattr(session, "_foreground_stream_task", None)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    await http.aclose()


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
async def test_drain_pushes_each_beat_to_renderer_so_spinner_clears(
    tmp_path: Path,
) -> None:
    """As the foreground drain installs each later beat into the tree
    it must PUSH that beat to the renderer (``session.emit`` →
    ``s2c/state/patch`` adding ``/dialog_tree/nodes/<id>``) — WITHOUT
    the player clicking Continue.

    Symptom this pins: the InteractionPanel's Continue glyph shows a
    spinner while ``hasContinueChild`` is false. The drain used to
    mutate only server-side state via ``install_game``, so the
    renderer's dialog_tree mirror never gained the next beat until its
    own next Continue walked to it — the glyph kept spinning even
    though the beat the player could walk to was already in the tree.
    """
    payload_text = json.dumps(_three_beat_payload())
    chunks = _split_after_each_beat(payload_text)
    assert len(chunks) == 4

    # Beat 1 lands immediately; beats 2/3 + tail are spaced out so the
    # drain installs them one at a time after the initial advance
    # returns. No further Continue clicks in this test — the whole
    # point is that the renderer learns about the beats passively.
    delays = [0.0, 0.2, 0.2, 0.2]
    transport = _make_streaming_transport(
        chunks=chunks,
        delay_before_chunk=delays,
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

    pushed: list[tuple[MessageType, object]] = []

    def _emit(message_type: MessageType, payload: object) -> None:
        pushed.append((message_type, payload))

    session = Session(
        settings=settings,
        llm_client=llm_client,
        image_client=_NullImage(),
        saves_root=tmp_path / "saves",
        emit=_emit,
    )
    session.install_game(_bootstrap_game())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    # First (and only) Continue — kicks off the stream + drain.
    async for _msg_type, _payload in registry.dispatch(
        Envelope(
            type=MessageType.c2s_play_advance,
            payload={"option_id": "o1"},
        ),
        ctx,
    ):
        pass

    # Let the drain finish installing the remaining beats.
    task = getattr(session, "_foreground_stream_task", None)
    assert task is not None
    await asyncio.wait_for(task, timeout=10.0)

    # Collect the node ids the drain ADDED to the tree via patch.
    added_node_ids: set[str] = set()
    for msg_type, payload in pushed:
        if msg_type != MessageType.s2c_state_patch:
            continue
        for op in payload.ops:
            if op.op == "add" and op.path.startswith("/dialog_tree/nodes/"):
                added_node_ids.add(op.path.rsplit("/", 1)[-1])

    # Beat 1 was walked + emitted by the dispatch itself. The drain is
    # responsible for the 2 later beats of the SAME continue-chain.
    # (The tree may also hold extra speculative nodes the post-chain
    # _ensure_speculative_branches generated against the tail, so we
    # follow parent links down the main chain rather than counting all
    # nodes.) Each later beat must have been pushed so the renderer's
    # tree gains the walkable child and the spinner flips to the ready
    # triangle without the player clicking.
    nodes = session.game.dialog_tree.nodes
    walked_first = session.game.current_node_id

    def _continue_child(parent_id: str) -> str | None:
        for n in nodes.values():
            if n.parent_id == parent_id and n.chosen_option_id is None:
                return n.id
        return None

    chain_after_first: list[str] = []
    cursor = _continue_child(walked_first)
    while cursor is not None:
        chain_after_first.append(cursor)
        cursor = _continue_child(cursor)

    assert len(chain_after_first) == 2, (
        f"expected 2 later beats in the continue-chain, got {chain_after_first}"
    )
    for nid in chain_after_first:
        assert nid in added_node_ids, (
            f"beat {nid} was installed in the tree but never pushed to "
            f"the renderer — its Continue glyph would keep spinning"
        )

    await http.aclose()


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
async def test_speculation_walks_as_first_beat_lands_not_full_chain(
    tmp_path: Path,
) -> None:
    """When a speculation task is streaming a chain, play_advance
    must walk into spec's first beat as soon as that beat closes
    in the stream — NOT block until the whole chain finishes.

    Before the per-beat-install + polling-await refactor, speculation
    ran with stream=False so ``_await_speculation`` blocked on the
    full LLM response, which was the "Continue 2+ wait for the full
    response" symptom the user reported.
    """
    payload_text = json.dumps(_three_beat_payload())
    chunks = _split_after_each_beat(payload_text)
    assert len(chunks) == 4

    # First chunk (beat 1) arrives immediately; the rest are
    # well-delayed so a successful early walk has lots of headroom
    # to land before the tail.
    delays = [0.0, 0.8, 0.8, 1.5]
    transport = _make_streaming_transport(
        chunks=chunks,
        delay_before_chunk=delays,
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
    session.latency.enable()
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    # Manually kick off a speculation task for option "o1" — this is
    # what _ensure_speculative_branches would do under the natural
    # flow but skipping that machinery lets the test focus on the
    # await behavior under controlled timing.
    from lucidium.api import handlers as h

    parent_node = session.game.dialog_tree.nodes["n0"]
    spec_task = asyncio.create_task(
        h._speculate_branch(
            session,
            parent_id=parent_node.id,
            option_id="o1",
            option_text=parent_node.options[0].text,
        )
    )
    session._speculative_tasks = {
        h._speculation_key(parent_node.id, "o1"): spec_task,
    }

    # Let the task get past its initial yield point and start
    # consuming chunks. The first chunk lands immediately (delay=0)
    # so beat 1 will be installed essentially right away.
    await asyncio.sleep(0.05)

    started = time.monotonic()
    patch_at: float | None = None
    async for msg_type, _msg_payload in registry.dispatch(
        Envelope(
            type=MessageType.c2s_play_advance,
            payload={"option_id": "o1"},
        ),
        ctx,
    ):
        if msg_type == MessageType.s2c_state_patch and patch_at is None:
            patch_at = time.monotonic()
    elapsed = (patch_at - started) if patch_at is not None else None
    assert patch_at is not None, "dispatch never emitted state/patch"
    # The full chain tail arrives at +0.8 + 0.8 + 1.5 = +3.1 s. The
    # walk must land WELL before that — spec's beat 1 is already
    # installed by the time the dispatch starts polling.
    assert elapsed is not None and elapsed < 1.0, (
        f"dispatch took {elapsed:.2f} s; would have been ≥3.1 s if "
        f"_await_speculation still blocked on the full LLM response"
    )

    # Cleanup.
    if not spec_task.done():
        spec_task.cancel()
        try:
            await spec_task
        except (asyncio.CancelledError, Exception):
            pass
    await http.aclose()


def _byte_chunks_with_paced_delays(
    text: str, *, total_seconds: float, every_n_chars: int = 4
) -> tuple[list[str], list[float]]:
    """Slice ``text`` into ``every_n_chars``-sized chunks and assign
    each a tiny delay so the whole payload streams over
    ``total_seconds`` of wall time — simulating real per-token SSE.
    """
    chunks = [text[i : i + every_n_chars] for i in range(0, len(text), every_n_chars)]
    if not chunks:
        return [], []
    per_chunk = total_seconds / len(chunks)
    delays = [per_chunk] * len(chunks)
    return chunks, delays


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
async def test_continues_walk_during_token_paced_stream(
    tmp_path: Path,
) -> None:
    """Stress: deliver the JSON in tiny token-sized chunks paced over
    several seconds. Each Continue click must walk to the next beat as
    soon as that beat's closing brace lands in the stream, not at the
    end of the whole response.
    """
    payload_text = json.dumps(_three_beat_payload())
    chunks, delays = _byte_chunks_with_paced_delays(
        payload_text,
        total_seconds=2.4,
        every_n_chars=6,
    )

    transport = _make_streaming_transport(
        chunks=chunks,
        delay_before_chunk=delays,
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
    session.latency.enable()
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    dispatch_results: list[dict] = []

    async def _do_advance(*, option_id: str | None) -> None:
        click_at = time.monotonic()
        patch_at: float | None = None
        node_landed: str | None = None
        async for msg_type, msg_payload in registry.dispatch(
            Envelope(
                type=MessageType.c2s_play_advance,
                payload={"option_id": option_id},
            ),
            ctx,
        ):
            if msg_type == MessageType.s2c_state_patch and patch_at is None:
                patch_at = time.monotonic()
                for op in msg_payload.ops:
                    if op.path == "/current_node_id":
                        node_landed = op.value
                        break
        dispatch_results.append(
            {"click_at": click_at, "patch_at": patch_at, "node_landed": node_landed},
        )

    await _do_advance(option_id="o1")
    await _do_advance(option_id=None)
    await _do_advance(option_id=None)

    # Same invariants apply: each display must follow its beat_ready
    # tightly.
    beat_ready_events = [e for e in session.latency.events if e.name == "beat_ready"]
    BUDGET_MS = 250.0
    for i, info in enumerate(dispatch_results):
        match = next(
            (e for e in beat_ready_events if e.node_id == info["node_landed"]),
            None,
        )
        assert match is not None, (
            f"dispatch #{i + 1}: no beat_ready for landed node {info['node_landed']}"
        )
        delay_ms = (info["patch_at"] - match.ts) * 1000
        assert delay_ms < BUDGET_MS, (
            f"dispatch #{i + 1}: state/patch landed "
            f"{delay_ms:.0f} ms AFTER its beat_ready — display should "
            f"happen within {BUDGET_MS:.0f} ms of beat close"
        )

    task = getattr(session, "_foreground_stream_task", None)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    await http.aclose()
