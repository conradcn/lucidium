#!/usr/bin/env python
"""New-game LOAD-TIME harness: drive N new games with varied parameters
through the REAL handler path and measure time-to-first-playable-paint.

Providers are fakes that inject *realistic per-call latency* (scaled down
by ``SPEEDUP``) so the harness reproduces the true wall-clock STRUCTURE of
the confirm critical path — what is awaited serially vs. what overlaps —
without a GPU or a live model. Reported numbers are projected back to real
seconds (measured * SPEEDUP).

The single number that matters: ``paint_s`` — seconds from dispatching
``c2s/new_game/confirm`` to the handler yielding the first
``s2c/state/full`` (the moment the renderer can paint the opening and the
player can act). Everything the confirm handler awaits *before* that yield
is load the player sits through.

Run::

    cd backend
    python scripts/measure_new_game_load.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lucidium.api.handlers import (  # noqa: E402
    HandlerContext,
    build_default_registry,
)
from lucidium.api.messages import (  # noqa: E402
    Envelope,
    InterviewStep,
    MessageType,
)
from lucidium.domain.settings import Settings  # noqa: E402
from lucidium.orchestration.session import Session  # noqa: E402

# ---- realistic per-op latency (REAL seconds), scaled by SPEEDUP ------------
SPEEDUP = 10.0
REAL = {
    "world_init": 5.0,
    "char_desc": 2.0,
    "name": 2.0,
    "side_expand": 2.5,
    "summary": 1.0,
    "options_other": 1.5,
    "portrait": 8.0,
    "background": 8.0,
    "music": 60.0,
}


def _sleep_for(kind: str) -> float:
    return REAL[kind] / SPEEDUP


# ---------------------------------------------------------------------------
# Event log — every provider call records (t_start, t_end, kind) so we can
# decompose the confirm critical path afterwards.
# ---------------------------------------------------------------------------


@dataclass
class Clock:
    origin: float = field(default_factory=time.monotonic)
    events: list[dict] = field(default_factory=list)

    def now(self) -> float:
        return (time.monotonic() - self.origin) * SPEEDUP  # projected real s

    def mark(self, kind: str, t0: float, t1: float) -> None:
        self.events.append({"kind": kind, "start": t0, "end": t1})


# ---------------------------------------------------------------------------
# Fake providers
# ---------------------------------------------------------------------------


def _pc_block() -> dict:
    return {
        "name": "Iris",
        "description": "a wry archivist with salt in her hair",
        "gender": "female",
        "age": 28,
        "ethnicity": "local",
        "skin": "pale",
        "hair_color": "auburn",
        "hairstyle": "braid",
        "eye_color": "grey",
        "build": "slight",
        "bust": "moderate",
        "outfit": "charcoal wool coat",
        "pose": "standing",
        "expression": "alert",
        "effects": "",
    }


def _beat(
    text: str, *, entering: list[str], new_chars: list[dict], location_prompt: str | None
) -> dict:
    return {
        "text": text,
        "speaker_id": None,
        "entering_character_ids": entering,
        "leaving_character_ids": [],
        "new_characters": new_chars,
        "location_id": None,
        "location_prompt": location_prompt,
        "location_lighting": "cool dawn light",
        "character_changes": [],
        "music_change": None,
    }


def _npc(i: int) -> dict:
    return {
        "id": f"npc-{i}",
        "name": f"Harbor NPC {i}",
        "description": f"a weathered dockhand number {i}",
        "gender": "male",
        "age": 40,
        "outfit": "oilcloth coat",
        "pose": "standing",
        "expression": "wary",
    }


class FakeLlm:
    """Content-aware fake, latency-injected. ``beats_npcs`` is the number
    of NPCs entering per opening beat (one dialog line each): NPCs in the
    first three beats are AWAITED on the confirm path; the rest stream."""

    def __init__(self, clock: Clock, *, beats_npcs: list[int], music: bool) -> None:
        self.clock = clock
        self.beats_npcs = beats_npcs
        self.music = music
        self.calls = 0

    def _world_init(self) -> str:
        beats = []
        idx = 0
        for bi, count in enumerate(self.beats_npcs):
            npcs = [_npc(idx + k) for k in range(count)]
            idx += count
            beats.append(
                _beat(
                    f"Opening beat {bi} unfolds under a grey dawn.",
                    entering=[p["id"] for p in npcs],
                    new_chars=npcs,
                    # Only the first beat carries the initial scene; later
                    # beats reuse it (no new location_prompt).
                    location_prompt="a stone harbor at dawn, fishing boats" if bi == 0 else None,
                )
            )
        return json.dumps(
            {
                "game_name": "The Salt Lantern",
                "plot_outline": [
                    {
                        "id": "stage-arrival",
                        "title": "Arrival",
                        "goal": "Establish the harbor and the missing keeper.",
                        "summary": "The protagonist reaches the harbor at dawn.",
                    }
                ],
                "active_plot_threads": [],
                "opening_node": {
                    "beats": beats,
                    "options": [
                        {"id": "inn", "text": "Walk to the inn."},
                        {"id": "cliff", "text": "Climb the cliff path."},
                    ],
                },
                "player_character": _pc_block(),
                "initial_music_prompt": "slow maritime ambient" if self.music else "",
            }
        )

    def _side_expansion(self) -> str:
        return json.dumps(
            {
                "name": "Expanded Side",
                "description": "a fully drawn ally",
                "gender": "female",
                "age": 33,
                "ethnicity": "local",
                "skin": "tan",
                "hair_color": "black",
                "hairstyle": "short",
                "eye_color": "brown",
                "build": "average",
                "bust": "moderate",
                "outfit": "sailor's jacket",
                "pose": "leaning",
                "expression": "amused",
            }
        )

    def _respond(self, text: str) -> tuple[str, str]:
        if '"summarizer_assessment"' in text:
            return "summary", json.dumps(
                {
                    "summarizer_assessment": "underway",
                    "direction_signal": "none",
                    "new_facts_by_character": {},
                    "pruned_fact_ids": [],
                    "current_stage_id": None,
                    "revised_outline": None,
                    "user_profile_additions": {"likes": [], "dislikes": [], "notes": []},
                    "characters_to_offstage": [],
                }
            )
        if "One-line side character description" in text:
            return "side_expand", self._side_expansion()
        if '"opening_node"' in text or "opening_node" in text:
            return "world_init", self._world_init()
        if "Propose" in text and "first-name" in text:
            return "name", json.dumps({"options": [f"Name {i} Last" for i in range(8)]})
        if "Propose" in text:
            return "char_desc", json.dumps({"options": [f"desc {i}" for i in range(6)]})
        return "options_other", json.dumps({"options": []})

    async def complete(self, prompt, *, model, temperature, max_tokens, stream=False):
        self.calls += 1
        text = "\n".join(m.get("content", "") for m in prompt)
        kind, body = self._respond(text)
        t0 = self.clock.now()
        await asyncio.sleep(_sleep_for(kind if kind in REAL else "options_other"))
        self.clock.mark(f"llm:{kind}", t0, self.clock.now())

        async def gen() -> AsyncIterator[str]:
            yield body

        return gen()


class LatencyImage:
    """Fake image client. Serialises through a shared lock to model a
    single GPU, and sleeps a per-render latency so portrait/background
    stacking shows up in the timeline."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.lock = asyncio.Lock()

    async def generate(self, workflow, params=None, *, seed=None, **kw):
        pos = ""
        if isinstance(params, dict):
            pos = str(params.get("positive_prompt", ""))
        kind = "background" if "harbor" in pos and "boats" in pos else "portrait"
        t0 = self.clock.now()
        async with self.lock:
            await asyncio.sleep(_sleep_for(kind))
        self.clock.mark(f"img:{kind}", t0, self.clock.now())
        # 1x1 PNG
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00"
            b"\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc"
            b"\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )


class LatencyMusic:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock

    async def generate_music(self, text, *, seed=None, **kw):
        t0 = self.clock.now()
        await asyncio.sleep(_sleep_for("music"))
        self.clock.mark("music", t0, self.clock.now())
        return b"ID3fakeaudio"

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def _drain(handler_result, sink) -> list:
    out = []
    async for mt, payload in handler_result:
        sink.append((mt, payload))
        out.append((mt, payload))
    return out


async def _dispatch(registry, ctx, sink, mtype, payload):
    return await _drain(registry.dispatch(Envelope(type=mtype, payload=payload), ctx), sink)


async def _answer(registry, ctx, sink, step, answer, pronouns=""):
    await _dispatch(
        registry,
        ctx,
        sink,
        MessageType.c2s_new_game_answer,
        {
            "step": step.value,
            "answer": answer,
            "is_free_text": False,
            "pronouns": pronouns,
        },
    )


async def _cancel_all(session) -> None:
    tasks: list = []
    for attr in (
        "_world_init_task",
        "_char_desc_task",
        "_name_options_task",
        "_preview_bg_task",
        "_preview_guide_task",
        "_pc_portrait_task",
        "_summarizer_tasks",
        "_asset_tasks",
        "_foreground_stream_task",
    ):
        val = getattr(session, attr, None)
        if val is None:
            continue
        seq = val if isinstance(val, (list, tuple)) else [val]
        for t in seq:
            if t is not None and hasattr(t, "done") and not t.done():
                t.cancel()
                tasks.append(t)
    for t in (getattr(session, "_speculative_tasks", None) or {}).values():
        if not t.done():
            t.cancel()
            tasks.append(t)
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@dataclass
class Config:
    name: str
    side_chars: int
    beats_npcs: list[int]  # NPCs entering per opening beat (line)
    music: bool


async def run_one(cfg: Config, saves_root: Path) -> dict:
    clock = Clock()
    settings = Settings()
    settings.music.enabled = cfg.music
    llm = FakeLlm(clock, beats_npcs=cfg.beats_npcs, music=cfg.music)
    image = LatencyImage(clock)
    music = LatencyMusic(clock)

    sink: list = []
    session = Session(
        settings=settings,
        llm_client=llm,
        image_client=image,
        saves_root=saves_root,
        emit=lambda mt, p: sink.append(("EMIT", mt, p)),
    )
    if cfg.music:
        session.music_client = lambda: music  # type: ignore[method-assign]
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    # --- interview -----------------------------------------------------
    await _dispatch(registry, ctx, sink, MessageType.c2s_new_game_start, {})
    await _answer(registry, ctx, sink, InterviewStep.setting, "A stone harbor at dawn")
    vs = session.interview.visual_style_options[0]
    await _answer(registry, ctx, sink, InterviewStep.visual_style, vs)
    await _answer(registry, ctx, sink, InterviewStep.genre, "Mystery")
    # wait for prefetched char-desc options, then answer
    for _ in range(200):
        if session.interview.character_description_options:
            break
        await asyncio.sleep(0.01)
    cd = (session.interview.character_description_options or ["a wry archivist"])[0]
    await _answer(registry, ctx, sink, InterviewStep.character_description, cd)
    for _ in range(200):
        if session.interview.name_options:
            break
        await asyncio.sleep(0.01)
    nm = (session.interview.name_options or ["Iris Vale"])[0]
    await _answer(registry, ctx, sink, InterviewStep.name, nm, pronouns="she/her")

    # side characters (deferred stubs, expanded at confirm)
    for i in range(cfg.side_chars):
        await _dispatch(
            registry,
            ctx,
            sink,
            MessageType.c2s_new_game_add_side_character,
            {"description": f"a mysterious stranger {i}"},
        )

    # Let the world_init prefetch (kicked off at Name) resolve, mimicking a
    # player who lingers on the Review screen — isolates the CONFIRM
    # critical path (assets + music + side expansion) from world_init.
    t = getattr(session, "_world_init_task", None)
    if t is not None:
        try:
            await t
        except Exception:
            pass

    # --- confirm: measure time-to-first-paint --------------------------
    sink.clear()
    t_confirm = clock.now()
    result = registry.dispatch(
        Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
        ctx,
    )
    paint_s = None
    paint_ts = None
    confirm_img_ready = 0  # first-frame image_ready yielded inline by confirm
    async for mt, _payload in result:
        if mt == MessageType.s2c_state_full and paint_s is None:
            paint_ts = clock.now()
            paint_s = paint_ts - t_confirm
        if mt == MessageType.s2c_image_ready:
            confirm_img_ready += 1
    if paint_ts is None:
        paint_ts = clock.now()

    # Drain confirm-spawned background work (music render; the asset
    # pump, a no-op once the first frame covered the on-stage cast).
    async def _live_tasks():
        live = []
        for attr in ("_asset_tasks", "_music_tasks"):
            for t in getattr(session, attr, None) or []:
                if not t.done():
                    live.append(t)
        return live

    for _ in range(8000):
        if not await _live_tasks():
            break
        await asyncio.sleep(0.005)

    eps = 1e-3
    cev = [e for e in clock.events if e["start"] >= t_confirm - 1e-6]
    portraits = [e for e in cev if e["kind"] == "img:portrait"]
    bg_ev = [e for e in cev if e["kind"] == "img:background"]
    music_ev = [e for e in cev if e["kind"] == "music"]
    awaited_portraits = [e for e in portraits if e["end"] <= paint_ts + eps]
    bg_before_paint = sum(1 for e in bg_ev if e["end"] <= paint_ts + eps)
    music_emits = sum(1 for e in sink if e[0] == "EMIT" and e[1] == MessageType.s2c_music_ready)

    await _cancel_all(session)

    expected_awaited_npcs = sum(cfg.beats_npcs[:3])
    deferred_npcs = sum(cfg.beats_npcs[3:])
    return {
        "config": cfg.name,
        "side_chars": cfg.side_chars,
        "beats_npcs": cfg.beats_npcs,
        "music": cfg.music,
        "paint_s": round(paint_s if paint_s is not None else 0.0, 2),
        "awaited_before_paint": {
            "npc_portraits": len(awaited_portraits),
            "expected_npc_portraits": expected_awaited_npcs,
            "background": bg_before_paint,
            "inline_image_ready": confirm_img_ready,
        },
        "deferred": {
            "npcs_beyond_line3": deferred_npcs,
            "portraits_rendered_at_confirm": len(portraits) - len(awaited_portraits),
        },
        "music_streamed": len(music_ev),
        "music_ready_emits": music_emits,
    }


CONFIGS = [
    #     name                                         side  beats-npcs (per line)  music
    Config("A baseline (no NPC, music off)", 0, [0], False),
    Config("B music on, 1 NPC line 1", 0, [1], True),
    Config("C music off, 3 NPC across lines 1-3", 0, [1, 1, 1], False),
    Config("D music on, 2 side, 2 NPC line 1", 2, [2], True),
    Config("E music off, 3 early + 5 later NPCs", 0, [2, 0, 1, 2, 3], False),
    Config("F music on, 1 early + 4 later NPCs", 1, [1, 0, 0, 4], True),
    Config("G music off, 4 side, 3 NPC lines 1-3", 4, [1, 1, 1], False),
    Config("H music on, no NPC", 0, [0], True),
    Config("I music off, 6 NPC line 1 (all awaited)", 0, [6], False),
    Config("J music on, 3 early + 6 later (heaviest)", 4, [3, 0, 0, 3, 3], True),
]


async def main() -> int:
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="lucidium-load-"))
    print(f"== new-game load audit ==  (latencies projected x{SPEEDUP:.0f} to real s)")
    print("await = initial background + NPCs visible in first 3 lines; rest deferred\n")
    results = []
    for cfg in CONFIGS:
        saves = root / cfg.name.split()[0]
        saves.mkdir(parents=True, exist_ok=True)
        r = await run_one(cfg, saves)
        results.append(r)
        a = r["awaited_before_paint"]
        d = r["deferred"]
        ok = (
            a["npc_portraits"] == a["expected_npc_portraits"]
            and d["portraits_rendered_at_confirm"] == 0
        )
        print(f"{r['config']}   beats_npcs={r['beats_npcs']}")
        print(f"    time-to-first-paint: {r['paint_s']:.2f}s   [{'OK' if ok else 'CHECK'}]")
        print(
            f"    awaited before paint: bg={a['background']} "
            f"npc_faces={a['npc_portraits']}/{a['expected_npc_portraits']} "
            f"(inline image_ready x{a['inline_image_ready']})"
        )
        print(
            f"    deferred (NOT blocking paint): {d['npcs_beyond_line3']} later-line NPC(s); "
            f"music streamed x{r['music_ready_emits']}\n"
        )

    # Blocking path I/O is fine here: one-shot benchmark script, no other
    # tasks are waiting on this event loop at report-writing time.
    out = Path(__file__).resolve().parents[1] / "test-results" / "new_game_load.json"  # noqa: ASYNC240
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    worst = max(results, key=lambda r: r["paint_s"])
    best = min(results, key=lambda r: r["paint_s"])
    print("=" * 70)
    print(f"BEST : {best['paint_s']:.2f}s  ({best['config']})")
    print(f"WORST: {worst['paint_s']:.2f}s  ({worst['config']})")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
