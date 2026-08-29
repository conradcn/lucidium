"""Measure the four event-loop blocks called out in T8.1.

    python scripts/bench_loop_blocking.py --label after

Metrics 2-4 report TWO numbers each — ``[inline]`` / ``[envelope]`` is
the pre-T8.1 shape, reproduced here in the script, and ``[current]`` is
what the code does now. That way a single run on one machine gives a
like-for-like comparison instead of needing a checkout of the previous
revision, and the comparison keeps working as a regression check.
Metric 1 is a real subprocess launch, so it only has a ``current``
number; compare it across revisions.

``startup_to_port_ms``
    Wall time from ``python -m lucidium.app`` process spawn to the
    ``LUCIDIUM_WS_PORT=`` line landing on stdout. This is exactly how
    long the Electron splash sits there before the renderer can
    connect. Measured in a subprocess with a throwaway app-data dir so
    it never touches a real install.

``portrait_post_block_ms``
    The *loop-blocking* portion of one portrait's post-render work
    (``_render_or_await``'s ``safety_check`` + ``post_process`` +
    atomic write) at 832x1216 RGBA — see :class:`LoopWatch` for how
    "blocking" is measured. ``crop_to_figure`` is the real thing;
    ``safety_check`` is a fixed ``time.sleep`` stand-in for the two ONNX
    inferences (the real detectors need model weights that aren't in
    CI), so it exercises the same call path at a deterministic cost.
    ``_total_`` sums every stretch; the bare name is the worst single
    stretch.

``commit_block_ms``
    Same watchdog, around ``Session.commit()`` on a 1000-node tree.

``encode_ms`` / ``encode_bytes``
    ``ws_server._encode`` on a 1000-node ``s2c/state_full`` payload.

``dismiss_wire_bytes``
    Bytes the Cast tab's dismiss/show handlers put on the wire for a
    one-character change, at the same 1000 nodes: the full state they
    used to push versus the three targeted ops they push now.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

NODES = 1000
PORTRAIT_SIZE = (832, 1216)
# Stand-in cost for NudeDetector + insightface on an already-warm
# process: two ONNX inferences plus a tempfile write.
SAFETY_CHECK_STAND_IN_S = 0.050


# ---------- loop-lag watchdog -------------------------------------------------


class LoopWatch:
    """Measure how much of the loop's time a piece of work stole.

    Ticks with ``await asyncio.sleep(0)`` — a bare yield, NOT a timed
    sleep. A timed sleep would be useless here: Windows' event-loop
    timer granularity is ~15 ms, so every "1 ms" tick reports a ~15 ms
    gap whether or not anything blocked, which swamps exactly the effect
    we're trying to see. A bare yield reschedules on the next loop
    iteration, so a gap between ticks is real work the loop did instead
    of running us.

    ``worst_ms`` is the longest single stretch the loop couldn't run
    anything else (the hitch a player would see as dropped frames).
    ``total_ms`` sums every stretch over the measured window — the same
    number a profiler would attribute to loop-thread work.
    """

    # Yields shorter than this are the watchdog's own scheduling
    # overhead, not somebody blocking.
    _FLOOR_S = 0.0005

    def __init__(self) -> None:
        self._gaps: list[float] = []
        self._stop = False
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> LoopWatch:
        self._task = asyncio.create_task(self._run())
        await asyncio.sleep(0)
        self._gaps.clear()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self._stop = True
        if self._task is not None:
            await self._task

    async def _run(self) -> None:
        last = time.perf_counter()
        while not self._stop:
            await asyncio.sleep(0)
            now = time.perf_counter()
            gap = now - last
            if gap > self._FLOOR_S:
                self._gaps.append(gap)
            last = now

    @property
    def worst_ms(self) -> float:
        return max(self._gaps, default=0.0) * 1000.0

    @property
    def total_ms(self) -> float:
        """Everything the loop lost, not just the worst single stretch —
        catches work that blocks in several medium-sized chunks."""
        return sum(self._gaps) * 1000.0


# ---------- fixtures ----------------------------------------------------------


def _portrait_png() -> bytes:
    """A figure-on-transparent PNG at portrait resolution, with the
    opaque region inset so ``crop_to_figure`` has real work to do."""
    from PIL import Image, ImageDraw

    width, height = PORTRAIT_SIZE
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (width // 4, height // 8, width * 3 // 4, height * 7 // 8),
        fill=(180, 120, 90, 255),
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _big_game(nodes: int = NODES):
    from lucidium.domain.character import Character
    from lucidium.domain.dialog import (
        DialogNode,
        DialogTree,
        premise_hash,
        world_snapshot_vector,
    )
    from lucidium.domain.game import Game
    from lucidium.domain.world import WorldState

    snapshot = world_snapshot_vector(
        on_stage_character_ids=[],
        character_attributes={},
        world_fields={"plot": "opening"},
    )
    built: dict[str, DialogNode] = {}
    order: list[str] = []
    parent: str | None = None
    for index in range(nodes):
        node = DialogNode(
            text=(
                f"Beat {index}: the harbor lanterns gutter as the tide "
                "turns, and somewhere below deck a door closes."
            ),
            premise_hash=premise_hash(
                parent_id=parent, chosen_option_id=None, snapshot_vector=snapshot
            ),
            parent_id=parent,
        )
        built[node.id] = node
        order.append(node.id)
        parent = node.id

    cast = [
        Character(name=f"Extra {i}", description="a dock hand", seed=1000 + i) for i in range(12)
    ]
    characters = {c.id: c for c in cast}
    return Game(
        world=WorldState(
            game_name="Benchmark",
            setting="harbor",
            genre="mystery",
            visual_style="ink",
        ),
        dialog_tree=DialogTree(nodes=built, root_id=order[0], committed_path=list(order)),
        current_node_id=order[-1],
        characters=characters,
        on_stage=[c.id for c in cast],
    )


# ---------- metric 1: startup -------------------------------------------------


def measure_startup(samples: int = 3) -> dict[str, float]:
    from lucidium.config import WS_PORT_ANNOUNCEMENT_PREFIX

    timings: list[float] = []
    for _ in range(samples):
        # ignore_cleanup_errors: the backend's log handler keeps
        # session.log open until Windows finishes reaping the process,
        # which loses a race with the rmtree here often enough to
        # matter. A leaked temp dir is not worth failing a benchmark.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as app_data:
            env = dict(os.environ)
            env["LUCIDIUM_APP_DATA"] = app_data
            env["LUCIDIUM_OFFLINE"] = "1"
            env["PYTHONPATH"] = str(_ROOT / "src")
            env["PYTHONUNBUFFERED"] = "1"
            started = time.perf_counter()
            proc = subprocess.Popen(
                [sys.executable, "-m", "lucidium.app"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                env=env,
                cwd=str(_ROOT),
            )
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    if line.startswith(WS_PORT_ANNOUNCEMENT_PREFIX):
                        timings.append((time.perf_counter() - started) * 1000.0)
                        break
                else:
                    raise RuntimeError("backend exited before announcing a port")
            finally:
                proc.kill()
                proc.wait(timeout=30)
    return {
        "startup_to_port_ms": round(statistics.median(timings), 1),
        "startup_to_port_ms_best": round(min(timings), 1),
    }


# ---------- metric 2: portrait post-processing --------------------------------


async def measure_portrait_post() -> dict[str, float]:
    from lucidium.orchestration.assets import _render_or_await
    from lucidium.orchestration.portrait_post import crop_to_figure
    from lucidium.persistence.atomic import atomic_write_bytes

    payload = _portrait_png()

    async def render() -> bytes:
        # Stand in for the diffusers call, which already runs on a
        # worker thread — we're measuring what happens after it.
        await asyncio.sleep(0)
        return payload

    def safety_check(_data: bytes) -> bool:
        time.sleep(SAFETY_CHECK_STAND_IN_S)
        return True

    async def inline(target: Path) -> None:
        """The old shape: safety check, crop and write straight on the
        loop thread. Kept here so one run reports both numbers rather
        than needing a checkout of the previous revision."""
        data = await render()
        if not safety_check(data):
            return
        atomic_write_bytes(target, crop_to_figure(data))

    async def current(target: Path) -> None:
        await _render_or_await(
            target,
            render=render,
            post_process=crop_to_figure,
            safety_check=safety_check,
        )

    results: dict[str, float] = {}
    for label, run in (("inline", inline), ("current", current)):
        worst: list[float] = []
        total: list[float] = []
        wall: list[float] = []
        with tempfile.TemporaryDirectory() as tmp:
            for index in range(5):
                target = Path(tmp) / f"portrait-{label}-{index}.png"
                async with LoopWatch() as watch:
                    started = time.perf_counter()
                    await run(target)
                    wall.append((time.perf_counter() - started) * 1000.0)
                worst.append(watch.worst_ms)
                total.append(watch.total_ms)
        results[f"portrait_post_block_ms[{label}]"] = round(statistics.median(worst), 1)
        results[f"portrait_post_block_total_ms[{label}]"] = round(statistics.median(total), 1)
        results[f"portrait_post_wall_ms[{label}]"] = round(statistics.median(wall), 1)
    return results


# ---------- metric 3: Session.commit ------------------------------------------


async def measure_commit() -> dict[str, float]:
    from lucidium.domain.settings import Settings
    from lucidium.orchestration.session import Session

    game = _big_game()

    async def inline(session: Session) -> None:
        # The old shape: the save_store write on the loop thread.
        session.commit_blocking()

    async def current(session: Session) -> None:
        await session.commit()

    results: dict[str, float] = {}
    for label, run in (("inline", inline), ("current", current)):
        worst: list[float] = []
        total: list[float] = []
        wall: list[float] = []
        with tempfile.TemporaryDirectory() as tmp:
            session = Session(settings=Settings(), saves_root=Path(tmp))
            session.game = game
            for _ in range(5):
                async with LoopWatch() as watch:
                    started = time.perf_counter()
                    await run(session)
                    wall.append((time.perf_counter() - started) * 1000.0)
                worst.append(watch.worst_ms)
                total.append(watch.total_ms)
        results[f"commit_block_ms[{label}]"] = round(statistics.median(worst), 1)
        results[f"commit_block_total_ms[{label}]"] = round(statistics.median(total), 1)
        results[f"commit_wall_ms[{label}]"] = round(statistics.median(wall), 1)
    return results


# ---------- metric 4: _encode + cast-tab wire size ----------------------------


def measure_encode() -> dict[str, float]:
    from lucidium.api.messages import (
        Envelope,
        MessageType,
        PatchOp,
        S2CStateFull,
        S2CStatePatch,
    )
    from lucidium.api.ws_server import _encode
    from lucidium.domain.settings import Settings

    game = _big_game()
    full = S2CStateFull(game=game, settings=Settings())

    def encode_via_envelope(message_type, payload) -> str:
        """The old shape: dump to dicts, re-validate through Envelope,
        re-serialise."""
        return Envelope(
            type=message_type, payload=payload.model_dump(mode="json")
        ).model_dump_json()

    results: dict[str, float] = {}
    for label, encoder in (("envelope", encode_via_envelope), ("current", _encode)):
        encoder(MessageType.s2c_state_full, full)  # warm the pydantic caches
        timings: list[float] = []
        for _ in range(5):
            started = time.perf_counter()
            encoded = encoder(MessageType.s2c_state_full, full)
            timings.append((time.perf_counter() - started) * 1000.0)
        results[f"encode_ms[{label}]"] = round(statistics.median(timings), 2)
        results[f"encode_bytes[{label}]"] = len(encoded.encode("utf-8"))

    dismissed = next(iter(game.characters.values()))
    patch = S2CStatePatch(
        ops=[
            PatchOp(
                op="replace",
                path=f"/characters/{dismissed.id}/removed",
                value=True,
            ),
            PatchOp(
                op="replace",
                path=f"/characters/{dismissed.id}/removed_reason",
                value="dismissed",
            ),
            PatchOp(
                op="replace",
                path="/on_stage",
                value=[c for c in game.on_stage if c != dismissed.id],
            ),
        ]
    )
    # What a Cast-tab dismiss puts on the wire: the full state before,
    # three targeted ops now.
    results["dismiss_wire_bytes[full_state]"] = results["encode_bytes[current]"]
    results["dismiss_wire_bytes[patch]"] = len(
        _encode(MessageType.s2c_state_patch, patch).encode("utf-8")
    )
    return results


# ---------- driver ------------------------------------------------------------


async def _async_metrics() -> dict[str, float]:
    results: dict[str, float] = {}
    results.update(await measure_portrait_post())
    results.update(await measure_commit())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="run", help="tag for the output row")
    parser.add_argument(
        "--skip-startup",
        action="store_true",
        help="skip the subprocess startup metric (the slow one)",
    )
    parser.add_argument("--json", type=Path, help="also write results here")
    args = parser.parse_args()

    results: dict[str, float] = {}
    if not args.skip_startup:
        results.update(measure_startup())
    results.update(asyncio.run(_async_metrics()))
    results.update(measure_encode())

    print(f"\n=== {args.label} ===")
    for key, value in results.items():
        print(f"{key:34s} {value}")
    if args.json:
        args.json.write_text(
            json.dumps({"label": args.label, **results}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
