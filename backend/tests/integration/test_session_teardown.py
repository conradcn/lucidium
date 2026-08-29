"""Cancellation and teardown.

Three things used to be true at once, and together they meant a
Lucidium window could be closed mid-turn and keep spending money:

1. **No cancel affordance.** Nothing in the protocol let the player
   stop a generation they'd already decided against.
2. **The GPU kept going.** Every embedded render is
   ``asyncio.to_thread(...)``; cancelling the awaiting coroutine
   detaches the future but the worker thread runs the full denoise
   loop, still holding ``gpu_lock``.
3. **Disconnect cancelled nothing.** ``Session`` had no teardown at
   all, so speculation / summarizer / music / render tasks kept
   running against an orphaned session.

These tests pin the fixes: ``Session.aclose`` cancels the session's
whole task surface and closes its provider clients; ``c2s/play/cancel``
ends the foreground generation; and a cancelled embedded render aborts
its denoise loop and releases ``gpu_lock`` instead of running to
completion.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

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
from lucidium.domain.settings import Settings
from lucidium.domain.world import WorldState
from lucidium.orchestration.session import Session
from lucidium.providers.embedded_image_client import EmbeddedImageClient

_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)


class _RecordingImageClient:
    """Image client that records whether its ``aclose`` was awaited.

    A synchronous ``closed`` flag isn't enough — the bug being pinned
    is teardown that FIRES ``aclose()`` without awaiting it (the old
    ``loop.create_task(aclose())`` pattern), which sets no flag until
    some later tick. So the flag is set INSIDE the coroutine, after a
    suspension point: it can only be True if the coroutine actually
    ran to completion before ``aclose`` returned."""

    def __init__(self) -> None:
        self.closed = False

    async def generate(self, *_a: Any, **_kw: Any) -> bytes:
        return _PNG_1X1

    async def aclose(self) -> None:
        await asyncio.sleep(0)
        self.closed = True


class _HangingLlm:
    """LLM client whose stream opens and then never yields. Stands in
    for a real provider mid-generation: the handler is suspended
    inside ``llm_stream_chunks`` waiting for beat 1."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def complete(self, *_a: Any, **_kw: Any) -> AsyncIterator[str]:
        async def _stream() -> AsyncIterator[str]:
            self.started.set()
            await asyncio.Event().wait()
            yield ""  # pragma: no cover -- unreachable

        return _stream()


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
        premise_hash="h0",
    )
    return Game(
        world=WorldState(
            game_name="Test",
            setting="harbor",
            genre="mystery",
            visual_style="ink",
        ),
        characters={"pc": pc},
        dialog_tree=DialogTree(
            nodes={"n0": n0},
            root_id="n0",
            committed_path=["n0"],
        ),
        environments={},
        current_node_id="n0",
        on_stage=[],
    )


# ---------------------------------------------------------------------------
# (1) Session.aclose
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aclose_cancels_every_task_the_session_owns(tmp_path: Path) -> None:
    """Every handle on the session's task surface is torn down.

    The tasks are planted directly rather than provoked through
    handlers on purpose: the point of the test is that ``aclose``'s
    inventory covers each ATTRIBUTE the runtime grafts work onto. A
    test that only exercised the two or three attributes a particular
    play flow happens to populate would pass while the rest leaked.
    """
    image = _RecordingImageClient()
    session = Session(
        settings=Settings(),
        image_client=image,
        saves_root=tmp_path / "saves",
    )

    async def forever() -> None:
        await asyncio.Event().wait()

    planted: dict[str, asyncio.Task] = {}

    def plant(name: str) -> asyncio.Task:
        task = asyncio.create_task(forever(), name=name)
        planted[name] = task
        return task

    session._foreground_task = plant("foreground")
    session._foreground_stream_task = plant("foreground_stream")
    session._world_init_task = plant("world_init")
    session._char_desc_task = plant("char_desc")
    session._name_options_task = plant("name_options")
    session._preview_bg_task = plant("preview_bg")
    session._preview_guide_task = plant("preview_guide")
    session._pc_portrait_task = plant("pc_portrait")
    session._summarizer_tasks = [plant("summarizer")]
    session._music_tasks = [plant("music")]
    session._asset_tasks = [plant("asset")]
    session._speculative_tasks = {"n0:o1": plant("speculative")}
    session.render_scheduler._pump = plant("render_pump")

    # Let every task reach its first suspension point, so cancellation
    # has somewhere to land (a never-started task would report
    # ``cancelled()`` trivially and prove nothing).
    await asyncio.sleep(0)
    assert all(not t.done() for t in planted.values())

    await session.aclose()

    not_cancelled = sorted(name for name, task in planted.items() if not task.cancelled())
    assert not_cancelled == [], f"session.aclose left these tasks un-cancelled: {not_cancelled}"
    assert image.closed, "aclose() on the image client was not awaited"


@pytest.mark.asyncio
async def test_aclose_is_idempotent_and_clears_handles(tmp_path: Path) -> None:
    """The WS ``finally`` can run twice on a messy shutdown; a second
    ``aclose`` must not blow up or re-close the provider clients."""
    image = _RecordingImageClient()
    session = Session(
        settings=Settings(),
        image_client=image,
        saves_root=tmp_path / "saves",
    )
    task = asyncio.create_task(asyncio.Event().wait())
    session._summarizer_tasks = [task]
    await asyncio.sleep(0)

    await session.aclose()
    assert task.cancelled()
    assert session._summarizer_tasks == []
    assert session.owned_tasks() == []

    image.closed = False
    await session.aclose()  # must be a no-op, not a second teardown
    assert image.closed is False


# ---------------------------------------------------------------------------
# (2) c2s/play/cancel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_play_cancel_ends_the_foreground_generation(tmp_path: Path) -> None:
    """A ``c2s/play/cancel`` frame arriving mid-generation ends the
    advance task and hands the play lock back.

    The dispatch is run as its own task, mirroring ``ws_server``'s
    per-envelope dispatch — which is exactly what makes a cancel frame
    reachable at all. Dispatched inline (the old shape) the read loop
    would still be blocked on the advance and would not even have
    parsed the cancel."""
    llm = _HangingLlm()
    session = Session(
        settings=Settings(),
        llm_client=llm,
        image_client=_RecordingImageClient(),
        saves_root=tmp_path / "saves",
    )
    session.install_game(_bootstrap_game())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    async def drive(message_type: MessageType, payload: dict) -> list:
        envelope = Envelope(type=message_type, payload=payload)
        return [m async for m in registry.dispatch(envelope, ctx)]

    advance = asyncio.create_task(drive(MessageType.c2s_play_advance, {"option_id": "o1"}))
    await asyncio.wait_for(llm.started.wait(), timeout=5)
    assert session._foreground_task is not None

    out = await asyncio.wait_for(
        drive(MessageType.c2s_play_cancel, {}),
        timeout=5,
    )
    assert out and out[0][0] is MessageType.s2c_play_cancelled
    assert out[0][1].cancelled is True

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(advance, timeout=5)
    assert advance.cancelled()
    assert not session._ensure_play_lock().locked(), (
        "the cancelled advance did not release the play lock"
    )


@pytest.mark.asyncio
async def test_play_cancel_with_nothing_in_flight_reports_false(
    tmp_path: Path,
) -> None:
    """No generation running — the ack says so, so the renderer can
    drop its spinner instead of waiting for messages that will never
    arrive."""
    session = Session(settings=Settings(), saves_root=tmp_path / "saves")
    ctx = HandlerContext(session=session)
    registry = build_default_registry()
    envelope = Envelope(type=MessageType.c2s_play_cancel, payload={})
    out = [m async for m in registry.dispatch(envelope, ctx)]
    assert out[0][1].cancelled is False


# ---------------------------------------------------------------------------
# (3) the GPU actually stops
# ---------------------------------------------------------------------------


class _SlowPipeline:
    """Fake diffusers pipeline with a real denoise loop.

    Calls ``callback_on_step_end`` once per step exactly as diffusers
    does, so aborting through that hook is exercised for real rather
    than mocked. ``steps_run`` is how the test tells "the loop stopped
    early" from "the loop ran to completion and the result was thrown
    away"."""

    TOTAL_STEPS = 200

    def __init__(self) -> None:
        self.config = type("Cfg", (), {"_name_or_path": "test-stub"})()
        self.entered = threading.Event()
        self.finished = threading.Event()
        self.steps_run = 0

    def __call__(self, **kwargs: Any) -> Any:
        self.entered.set()
        callback = kwargs.get("callback_on_step_end")
        try:
            for step in range(self.TOTAL_STEPS):
                self.steps_run = step + 1
                if callback is not None:
                    callback(self, step, 0.0, {})
                threading.Event().wait(0.005)
        finally:
            self.finished.set()
        width = kwargs.get("width", 512)
        height = kwargs.get("height", 512)
        result = type("Result", (), {})()
        result.images = [Image.new("RGB", (width, height), color=(1, 2, 3))]
        return result


@pytest.mark.asyncio
async def test_cancelled_render_aborts_the_denoise_loop_and_frees_gpu_lock(
    tmp_path: Path,
) -> None:
    """Cancelling the coroutine awaiting a render stops the worker
    thread and releases ``gpu_lock``.

    Before the abort event existed, the ``await`` raised
    ``CancelledError`` immediately while the thread ran all 200 steps —
    so the next render acquired ``gpu_lock`` and started competing for
    VRAM with work whose output was already garbage.
    """
    (tmp_path / "model.safetensors").write_bytes(b"stub")
    pipeline = _SlowPipeline()
    gpu_lock = asyncio.Lock()
    client = EmbeddedImageClient(
        models_dir=str(tmp_path),
        model_name="model.safetensors",
        pipeline_factory=lambda *_a, **_kw: pipeline,
        bg_remover=lambda data: data,
        gpu_lock=gpu_lock,
    )

    render = asyncio.create_task(
        client.generate(
            "background.json",
            {"positive_prompt": "a quiet harbor"},
            seed=7,
        )
    )
    # Wait for the worker thread to be inside the denoise loop AND the
    # lock to be held, so we're cancelling a genuinely in-flight render.
    await asyncio.wait_for(
        asyncio.to_thread(pipeline.entered.wait, 10),
        timeout=15,
    )
    assert gpu_lock.locked()
    steps_at_cancel = pipeline.steps_run

    render.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(render, timeout=15)

    assert pipeline.finished.is_set(), (
        "the worker thread was still running after the awaiting coroutine finished unwinding"
    )
    assert pipeline.steps_run < _SlowPipeline.TOTAL_STEPS, (
        f"denoise loop ran to completion ({pipeline.steps_run} steps) "
        f"despite cancellation at step {steps_at_cancel}"
    )
    assert not gpu_lock.locked(), "gpu_lock was not released after cancellation"

    await client.aclose()
