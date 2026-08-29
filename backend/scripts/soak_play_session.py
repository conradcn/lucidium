#!/usr/bin/env python
"""200-turn offline soak test for the play loop.

Drives a full session — interview → confirm → 200 forward turns — against
a content-aware *fake* LLM (no network, no GPU), mixing three input modes:

  * **suggested options** — picking an LLM-suggested choice button
  * **manual options**    — free-text ("do something else") input
  * **undos**             — ``c2s/play/undo`` bursts at VARYING intervals

It is NOT a correctness oracle for story content; it is a *loading-state
and performance* oracle. After every operation it asserts the invariants
the renderer relies on to leave its "loading" affordances:

  F (freeze)   advance/free_text MUST move ``current_node_id``. If it
               doesn't, ``useOptimisticAction``'s guard never releases
               and the option/Continue buttons stay disabled forever —
               the production "buttons frozen" bug.
  T (text)     the landed node MUST carry text + a ready/committed
               state, else the renderer paints "…" indefinitely.
  S (spinner)  a no-options node MUST, once speculation settles, have a
               walkable Continue child that the RENDERER KNOWS ABOUT
               (was emitted via state/full or a state/patch add). The
               backend installing it silently isn't enough — the
               Continue glyph spins forever if the renderer mirror
               never learned the next beat exists.
  U (undo)     undo MUST move ``current_node_id`` to a real committed
               node (so the store rebuilds the game ref and the guard
               releases).

Performance / leak signals tracked across the run: per-op latency
(avg / p95 / max), dialog-node count growth, live speculative-task
count, and undo-stack depth.

Run::

    python backend/scripts/soak_play_session.py            # 200 turns
    python backend/scripts/soak_play_session.py 50         # quick

A JSON report is written under ``backend/test-results/``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import statistics
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lucidium.api.handlers import HandlerContext, build_default_registry  # noqa: E402
from lucidium.api.messages import (  # noqa: E402
    Envelope,
    InterviewStep,
    MessageType,
)
from lucidium.domain.settings import Settings  # noqa: E402
from lucidium.orchestration.session import Session  # noqa: E402

TURNS_DEFAULT = 200
# How long (in event-loop yields) we let post-commit speculation settle
# before judging the spinner invariant. The fake LLM resolves in a
# single yield, so this is generous.
SETTLE_YIELDS = 60


# ---------------------------------------------------------------------------
# Content-aware fake LLM
# ---------------------------------------------------------------------------


def _hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def _beat_obj(text: str) -> dict:
    return {
        "text": text,
        "speaker_id": None,
        "entering_character_ids": [],
        "leaving_character_ids": [],
        "new_characters": [],
        "location_id": None,
        "location_prompt": None,
        "location_lighting": "",
        "character_changes": [],
        "music_change": None,
    }


def _world_init() -> str:
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
                "beats": [_beat_obj("The harbor wakes slow under a grey dawn.")],
                "options": [
                    {"id": "inn", "text": "Walk to the inn."},
                    {"id": "cliff", "text": "Climb the cliff path."},
                ],
            },
            "player_character": {
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
            },
            "initial_music_prompt": "",
        }
    )


def _summary() -> str:
    # Deliberately empty so facts/profile consolidation never crosses its
    # cap — keeps the soak focused on the play state machine.
    return json.dumps(
        {
            "summarizer_assessment": "The story is underway.",
            "direction_signal": "none",
            "new_facts_by_character": {},
            "pruned_fact_ids": [],
            "current_stage_id": None,
            "revised_outline": None,
            "user_profile_additions": {"likes": [], "dislikes": [], "notes": []},
            "characters_to_offstage": [],
        }
    )


def _beats(prompt_text: str) -> str:
    h = _hash(prompt_text)
    line = f"The scene turns, and beat {h % 100000} comes to pass."
    give_options = (h % 2) == 0
    options: list[dict] = []
    if give_options:
        n = 2 + (h % 2)  # 2 or 3
        options = [
            {"id": f"o-{h % 100003}-{k}", "text": f"Option {k} at junction {h % 997}"}
            for k in range(n)
        ]
    return json.dumps({"beats": [_beat_obj(line)], "options": options})


def _respond(prompt_text: str) -> str:
    # Order matters: world_init and summary skeletons both mention
    # "options"/"beats", so test their distinctive keys first.
    if '"summarizer_assessment"' in prompt_text:
        return _summary()
    if '"opening_node"' in prompt_text:
        return _world_init()
    if "Propose" in prompt_text and '"options"' in prompt_text:
        # character_description (6) or name (8) interview lists.
        count = 8 if "first-name" in prompt_text else 6
        return json.dumps({"options": [f"suggestion {i}" for i in range(count)]})
    if '"beats"' in prompt_text:
        return _beats(prompt_text)
    # Unknown structured call (consolidation, etc.) — never reached in a
    # clean run because we emit no facts. Empty option list is harmless.
    return json.dumps({"options": []})


class FakeLlm:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, prompt, *, model, temperature, max_tokens, stream: bool = False):
        self.calls += 1
        text = "\n".join(m.get("content", "") for m in prompt)
        body = _respond(text)

        async def gen() -> AsyncIterator[str]:
            yield body

        return gen()


class NullImage:
    async def generate(self, *_a, **_kw) -> bytes:
        return b""


# ---------------------------------------------------------------------------
# Renderer mirror — replays emitted messages to track which dialog nodes
# the renderer actually KNOWS about (the spinner invariant turns on this).
# ---------------------------------------------------------------------------


_NODE_PREFIX = "/dialog_tree/nodes/"


class RendererMirror:
    def __init__(self) -> None:
        self.known: set[str] = set()

    def feed(self, mt: MessageType, payload: object) -> None:
        if mt == MessageType.s2c_state_full:
            game = getattr(payload, "game", None)
            if game is not None:
                self.known = set(game.dialog_tree.nodes.keys())
            return
        if mt == MessageType.s2c_state_patch:
            for op in getattr(payload, "ops", []) or []:
                self._apply_op(op)

    def _apply_op(self, op: object) -> None:
        path = getattr(op, "path", "") or ""
        kind = getattr(op, "op", "") or ""
        value = getattr(op, "value", None)
        if path.startswith(_NODE_PREFIX):
            nid = path[len(_NODE_PREFIX) :].split("/")[0]
            if kind in ("add", "replace"):
                self.known.add(nid)
            elif kind == "remove":
                self.known.discard(nid)
        elif path in ("/dialog_tree", "/dialog_tree/nodes") and isinstance(value, dict):
            nodes = value.get("nodes", value)
            if isinstance(nodes, dict):
                self.known |= set(nodes.keys())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def _drain(handler_result, mirror: RendererMirror) -> list:
    out = []
    async for mt, payload in handler_result:
        mirror.feed(mt, payload)
        out.append((mt, payload))
    return out


async def _dispatch(registry, ctx, mirror, mtype, payload):
    return await _drain(registry.dispatch(Envelope(type=mtype, payload=payload), ctx), mirror)


async def _settle(session: Session) -> None:
    """Let post-commit speculation finish so the spinner invariant is
    judged against steady state, not mid-flight."""
    for _ in range(SETTLE_YIELDS):
        await asyncio.sleep(0)
        spec = getattr(session, "_speculative_tasks", None) or {}
        fg = getattr(session, "_foreground_stream_task", None)
        live = [t for t in spec.values() if not t.done()]
        if fg is not None and not fg.done():
            live.append(fg)
        if not live:
            return


def _walkable_continue_child(session: Session, parent_id: str):
    """Mirror InteractionPanel.hasContinueChild: a non-invalidated child
    reached via Continue (chosen_option_id is None)."""
    for node in session.game.dialog_tree.nodes.values():
        if (
            node.parent_id == parent_id
            and node.chosen_option_id is None
            and node.state.value != "invalidated"
        ):
            return node
    return None


async def _run_interview(registry, ctx, mirror) -> None:
    await _dispatch(registry, ctx, mirror, MessageType.c2s_new_game_start, {})
    for step, ans in [
        (InterviewStep.setting, "A stone harbor at dawn"),
        (InterviewStep.visual_style, "ink wash"),
        (InterviewStep.genre, "Mystery"),
        (InterviewStep.character_description, "a wry archivist"),
        (InterviewStep.name, "Iris"),
    ]:
        await _dispatch(
            registry,
            ctx,
            mirror,
            MessageType.c2s_new_game_answer,
            {"step": step.value, "answer": ans, "is_free_text": False},
        )
    await _dispatch(registry, ctx, mirror, MessageType.c2s_new_game_confirm, {"overrides": {}})


# Varying undo schedule: at these forward-turn indices, run a burst of N
# undos before continuing. Gaps and burst sizes both vary.
_UNDO_PLAN = {
    8: 1,
    13: 2,
    19: 1,
    28: 3,
    34: 1,
    41: 2,
    47: 1,
    56: 2,
    63: 1,
    71: 3,
    78: 1,
    86: 2,
    93: 1,
    101: 2,
    108: 1,
    116: 3,
    123: 1,
    131: 2,
    139: 1,
    148: 2,
    156: 1,
    163: 3,
    171: 1,
    178: 2,
    186: 1,
    194: 2,
}

# Free-text ("manual option") at a varying cadence interleaved with
# suggested-option picks.
_FREE_TEXT_TURNS = {n for n in range(1, 400) if n % 5 == 3 or n % 7 == 0}

_FREE_TEXT_POOL = [
    "I take a slow look around before doing anything.",
    "I follow the sound and see where it leads.",
    "I keep going, but stay alert.",
    "I ask whether anyone has news from the harbor.",
    "I quietly check my pockets for anything useful.",
]


async def run(turns: int) -> dict:
    settings = Settings()
    settings.music.enabled = False
    saves_root = Path(__import__("tempfile").mkdtemp(prefix="lucidium-soak-"))
    mirror = RendererMirror()
    llm = FakeLlm()
    session = Session(
        settings=settings,
        llm_client=llm,
        image_client=NullImage(),
        saves_root=saves_root,
        emit=mirror.feed,
    )
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    anomalies: list[dict] = []
    latencies: list[float] = []
    node_counts: list[int] = []
    spec_counts: list[int] = []

    def flag(kind: str, turn: int, detail: str) -> None:
        anomalies.append({"kind": kind, "turn": turn, "detail": detail})
        print(f"  ANOMALY [{kind}] turn {turn}: {detail}")

    await _run_interview(registry, ctx, mirror)
    await _settle(session)
    assert session.game is not None and session.game.current_node_id is not None

    for turn in range(1, turns + 1):
        # --- undo bursts at varying intervals ---
        for _ in range(_UNDO_PLAN.get(turn, 0)):
            if not session.undo_stack:
                break
            before = session.game.current_node_id
            t0 = time.perf_counter()
            await _dispatch(registry, ctx, mirror, MessageType.c2s_play_undo, {})
            latencies.append((time.perf_counter() - t0) * 1000)
            after = session.game.current_node_id
            if after == before:
                flag("U-no-move", turn, f"undo left current_node_id at {before}")
            elif after is None or after not in session.game.dialog_tree.nodes:
                flag("U-dangling", turn, f"undo landed on missing node {after}")
            else:
                landed = session.game.dialog_tree.nodes[after]
                if not (landed.text and landed.state.value in ("ready", "committed")):
                    flag(
                        "U-loading",
                        turn,
                        f"undo landed on node state={landed.state.value} "
                        f"text={'set' if landed.text else 'EMPTY'}",
                    )
            await _settle(session)

        # --- forward turn: suggested option vs manual free-text ---
        before = session.game.current_node_id
        node = session.game.dialog_tree.nodes[before]
        use_free_text = turn in _FREE_TEXT_TURNS
        if use_free_text:
            mtype = MessageType.c2s_play_free_text
            payload = {"text": _FREE_TEXT_POOL[turn % len(_FREE_TEXT_POOL)]}
            action = "free_text"
        else:
            mtype = MessageType.c2s_play_advance
            if node.options:
                # alternate first / last suggested option
                opt = node.options[0] if turn % 2 else node.options[-1]
                payload = {"option_id": opt.id}
                action = f"option:{opt.id}"
            else:
                payload = {"option_id": None}
                action = "continue"

        t0 = time.perf_counter()
        await _dispatch(registry, ctx, mirror, mtype, payload)
        latencies.append((time.perf_counter() - t0) * 1000)

        after = session.game.current_node_id

        # F — freeze invariant
        if after is None:
            flag("F-none", turn, f"{action}: current_node_id became None")
            continue
        if after == before:
            flag(
                "F-no-move",
                turn,
                f"{action}: current_node_id did not advance (stuck on {before})",
            )

        # T — landed-node text/state invariant
        landed = session.game.dialog_tree.nodes.get(after)
        if landed is None:
            flag("T-missing", turn, f"{action}: landed node {after} absent from tree")
            await _settle(session)
            continue
        if not landed.text:
            flag("T-empty", turn, f"{action}: landed node {after} has empty text")
        if landed.state.value not in ("ready", "committed"):
            flag(
                "T-state",
                turn,
                f"{action}: landed node state={landed.state.value} (expected ready/committed)",
            )

        # let speculation settle, then judge the spinner invariant
        await _settle(session)

        # S — spinner invariant (only meaningful on no-options nodes)
        landed = session.game.dialog_tree.nodes[after]
        if not landed.options:
            child = _walkable_continue_child(session, after)
            if child is None:
                spec = getattr(session, "_speculative_tasks", None) or {}
                live = [t for t in spec.values() if not t.done()]
                flag(
                    "S-no-continue",
                    turn,
                    f"{action}: no-options node {after} has NO walkable continue "
                    f"child after settle ({len(live)} spec tasks live) — Continue "
                    f"glyph would spin forever",
                )
            elif child.id not in mirror.known:
                flag(
                    "S-not-emitted",
                    turn,
                    f"{action}: walkable continue child {child.id} exists in backend "
                    f"tree but was NEVER emitted to renderer — spinner persists",
                )

        node_counts.append(len(session.game.dialog_tree.nodes))
        spec_counts.append(len(getattr(session, "_speculative_tasks", None) or {}))

    # Leak guard: the speculative-task dict must stay bounded to
    # in-flight work. Legitimate steady state is a small multiple of
    # SPECULATIVE_DEPTH_CAP (10); anything near the turn count means
    # finished tasks are accumulating instead of being pruned.
    spec_max = max(spec_counts) if spec_counts else 0
    if spec_max > 50:
        flag(
            "PERF-spec-leak",
            turns,
            f"_speculative_tasks peaked at {spec_max} entries (bound 50) — "
            f"completed speculations are not being pruned",
        )

    # cancel outstanding background tasks for a tidy exit
    await _cancel_all(session)

    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    report = {
        "turns": turns,
        "operations": len(latencies),
        "llm_calls": llm.calls,
        "latency_ms": {
            "avg": round(statistics.mean(latencies), 2),
            "p95": round(p95, 2),
            "max": round(max(latencies), 2),
        },
        "nodes": {
            "final": node_counts[-1] if node_counts else 0,
            "per_turn_growth": round(
                (node_counts[-1] - node_counts[0]) / max(1, len(node_counts) - 1), 2
            )
            if len(node_counts) > 1
            else 0,
        },
        "speculative_tasks": {
            "final": spec_counts[-1] if spec_counts else 0,
            "max": max(spec_counts) if spec_counts else 0,
        },
        "undo_stack_final": len(session.undo_stack),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }
    return report


async def _cancel_all(session: Session) -> None:
    tasks: list = []
    for attr in (
        "_summarizer_tasks",
        "_asset_tasks",
    ):
        val = getattr(session, attr, None) or []
        tasks.extend(t for t in val if not t.done())
    for t in (getattr(session, "_speculative_tasks", None) or {}).values():
        if not t.done():
            tasks.append(t)
    fg = getattr(session, "_foreground_stream_task", None)
    if fg is not None and not fg.done():
        tasks.append(fg)
    for t in tasks:
        t.cancel()
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


def main() -> int:
    turns = int(sys.argv[1]) if len(sys.argv) > 1 else TURNS_DEFAULT
    print(f"[soak] running {turns} forward turns (offline, fake LLM)...")
    started = time.perf_counter()
    report = asyncio.run(run(turns))
    report["wall_seconds"] = round(time.perf_counter() - started, 2)

    out_dir = Path(__file__).resolve().parents[1] / "test-results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "soak_play_session.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[soak] ===== REPORT =====")
    print(json.dumps({k: v for k, v in report.items() if k != "anomalies"}, indent=2))
    print(f"[soak] full report -> {out_file}")
    print(f"[soak] anomalies: {report['anomaly_count']}")
    return 1 if report["anomaly_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
