"""Per-dispatch latency audit against the live LLM and live image backend.

Drives 10 "decision iterations" against the real LLM provider configured
in user settings and the real image backend (ComfyUI or embedded SDXL).
Each iteration:

  1. Walks Continue at ``PACING_S``/click until landing on a node with
     options (the tail of the current chain).
  2. Waits one more ``PACING_S`` interval.
  3. Submits either a selected option (odd iterations) or a free-text
     answer (even iterations).

Every dispatch (Continue + decision) is timed against the invariant::

    displayed_at - click_at <= max(0, llm_end - max(click_at, predictable_at))
                               + TOLERANCE

Usage::

    cd backend
    .\\.venv\\Scripts\\python.exe scripts/audit_turn_latency.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "backend" / "src"))

from lucidium.api.handlers import (  # noqa: E402
    HandlerContext,
    build_default_registry,
)
from lucidium.api.messages import (  # noqa: E402
    Envelope,
    InterviewStep,
    MessageType,
)
from lucidium.orchestration.session import Session  # noqa: E402
from lucidium.persistence import settings_store  # noqa: E402

DECISIONS = 10
PACING_S = 1.0  # 1 click per second, per user spec
PER_TURN_BUDGET_S = 180.0
INVARIANT_TOLERANCE_MS = 120  # live envs are noisier than the mock


def _comfy_reachable(url: str) -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{url}/system_stats")
            return r.status_code == 200
    except Exception:
        return False


def _llm_configured(settings) -> tuple[bool, str]:
    base = (settings.llm.base_url or "").strip()
    key = (
        settings.llm.api_key.get_secret_value() or os.environ.get("OPENAI_API_KEY") or ""
    ).strip()
    model = (settings.llm.model or "").strip()
    if not base:
        return False, "settings.llm.base_url is unset"
    if not model:
        return False, "settings.llm.model is unset"
    if not key:
        return False, "settings.llm.api_key (and $OPENAI_API_KEY) are unset"
    return True, ""


async def _drain(handler_result: AsyncIterator) -> list:
    out = []
    async for msg in handler_result:
        out.append(msg)
    return out


async def _dispatch(registry, ctx, message_type, payload) -> list:
    return await _drain(
        registry.dispatch(
            Envelope(type=message_type, payload=payload),
            ctx,
        )
    )


async def _answer(registry, ctx, step: InterviewStep, answer: str) -> None:
    await _dispatch(
        registry,
        ctx,
        MessageType.c2s_new_game_answer,
        {"step": step.value, "answer": answer, "is_free_text": False},
    )


async def _wait_for_options(session, field: str, label: str, *, deadline_s: float) -> str:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        opts = getattr(session.interview, field, None) or []
        if opts:
            return opts[0]
        await asyncio.sleep(0.1)
    raise RuntimeError(f"{label} options never arrived within {deadline_s:.0f}s")


async def _drain_speculation(session, *, max_wait_s: float = 60.0) -> None:
    """Yield until in-flight speculative LLM/image tasks settle, with a
    timeout so a slow provider doesn't lock the audit in a wait loop."""
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        await asyncio.sleep(0.1)
        spec = getattr(session, "_speculative_tasks", None) or {}
        live = [t for t in spec.values() if not t.done()]
        if not live:
            return


def _first(events, name, *, parent_id=None, option_id=None, node_id=None):
    for e in events:
        if e.name != name:
            continue
        if parent_id is not None and e.parent_id != parent_id:
            continue
        if option_id is not None and e.option_id != option_id:
            continue
        if node_id is not None and e.node_id != node_id:
            continue
        return e
    return None


def _classify(click_event) -> str:
    if click_event.extra.get("kind") == "free_text":
        return "free_text"
    if click_event.option_id is None:
        return "continue"
    return "selected"


async def _cleanup_session(session) -> None:
    task_attrs = (
        "_char_desc_task",
        "_world_init_task",
        "_preview_bg_task",
        "_preview_guide_task",
        "_pc_portrait_task",
        "_name_options_task",
    )
    tasks_to_wait: list = []
    for attr in task_attrs:
        t = getattr(session, attr, None)
        if t is not None and not t.done():
            t.cancel()
            tasks_to_wait.append(t)
    for spec in (getattr(session, "_speculative_tasks", None) or {}).values():
        if not spec.done():
            spec.cancel()
            tasks_to_wait.append(spec)
    for asset in getattr(session, "_asset_tasks", None) or []:
        if not asset.done():
            asset.cancel()
            tasks_to_wait.append(asset)
    for t in tasks_to_wait:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


async def main() -> int:
    print("== Lucidium turn-latency audit ==")
    settings = settings_store.load_settings()

    llm_ok, why = _llm_configured(settings)
    if not llm_ok:
        print(f"ABORT: live audit requires real LLM: {why}")
        return 1
    comfy_url = settings.image.base_url or "http://127.0.0.1:8000"
    embedded = settings.image.backend.value if hasattr(settings.image, "backend") else "unknown"
    if embedded != "embedded" and not _comfy_reachable(comfy_url):
        print(f"ABORT: live audit requires reachable image backend at {comfy_url}")
        return 1

    print(f"LLM model:        {settings.llm.model}")
    print(f"Image backend:    {embedded}")
    print(f"Decisions:        {DECISIONS} (interleaved selected + free_text)")
    print(f"Pacing:           {PACING_S:.1f}s per click")
    print(f"Tolerance:        {INVARIANT_TOLERANCE_MS} ms")
    print()

    saves_root = Path(os.environ.get("TEMP", "/tmp")) / "lucidium_audit_saves"
    saves_root.mkdir(parents=True, exist_ok=True)
    session = Session(settings=settings, saves_root=saves_root)
    session.latency.enable()
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    try:
        # --- interview -----------------------------------------------------
        print("interview ...")
        t0 = time.monotonic()
        await _dispatch(registry, ctx, MessageType.c2s_new_game_start, {})
        await _answer(registry, ctx, InterviewStep.setting, "A stone harbor at dawn")
        visual = session.interview.visual_style_options[0]
        await _answer(registry, ctx, InterviewStep.visual_style, visual)
        await _answer(registry, ctx, InterviewStep.genre, "Mystery")
        char_desc = await _wait_for_options(
            session,
            "character_description_options",
            "character_description",
            deadline_s=PER_TURN_BUDGET_S,
        )
        await _answer(
            registry,
            ctx,
            InterviewStep.character_description,
            char_desc,
        )
        name = await _wait_for_options(
            session,
            "name_options",
            "name",
            deadline_s=PER_TURN_BUDGET_S,
        )
        await _answer(registry, ctx, InterviewStep.name, name)
        await _dispatch(
            registry,
            ctx,
            MessageType.c2s_new_game_confirm,
            {"overrides": {}},
        )
        print(f"  interview done in {time.monotonic() - t0:.1f}s")

        if session.game is None or session.game.current_node_id is None:
            print("ABORT: no game / no current_node_id after confirm")
            return 1

        # Drain speculation before clearing audit so the opening's
        # speculation_spawn events don't get mis-attributed to turn 1.
        await _drain_speculation(session)
        session.latency.clear()

        free_text_pool = [
            "I take a careful look around.",
            "I keep going, but stay alert.",
            "I follow the sound.",
            "I ask if anyone has news.",
            "I check my pockets.",
        ]
        dispatch_log: list[dict] = []

        async def _do_dispatch(*, kind: str, option_id: str | None, free_text: str | None) -> None:
            # Pacing wait BEFORE every click so speculation has time to
            # progress. This is the player's think-time, not part of
            # the invariant; the audit measures click->displayed which
            # the wait does not enter.
            await asyncio.sleep(PACING_S)
            node_before = session.game.current_node_id
            if free_text is not None:
                payload: dict = {"text": free_text}
                msg_type = MessageType.c2s_play_free_text
            else:
                payload = {"option_id": option_id}
                msg_type = MessageType.c2s_play_advance
            await asyncio.wait_for(
                _dispatch(registry, ctx, msg_type, payload),
                timeout=PER_TURN_BUDGET_S,
            )
            node_after = session.game.current_node_id
            if node_after == node_before:
                raise RuntimeError(
                    f"dispatch ({kind}, option={option_id}, "
                    f"free_text={free_text}): current_node_id did not "
                    f"advance — stuck on {node_before}"
                )
            await _drain_speculation(session)
            dispatch_log.append(
                {
                    "kind": kind,
                    "click_option_id": option_id,
                    "parent_id": node_before,
                    "destination_id": node_after,
                }
            )
            short_parent = (node_before or "NA")[-6:]
            print(f"  dispatch #{len(dispatch_log)} ({kind}): {short_parent} -> {node_after[-6:]}")

        for decision_idx in range(1, DECISIONS + 1):
            print(f"decision {decision_idx} ...")
            # Walk Continue at PACING_S/click until reaching a tail
            # node with options.
            while True:
                node = session.game.dialog_tree.nodes[session.game.current_node_id]
                if node.options:
                    break
                await _do_dispatch(kind="continue", option_id=None, free_text=None)

            is_selected = (decision_idx % 2) == 1
            if is_selected:
                node = session.game.dialog_tree.nodes[session.game.current_node_id]
                option_id = node.options[0].id
                await _do_dispatch(
                    kind="selected",
                    option_id=option_id,
                    free_text=None,
                )
            else:
                await _do_dispatch(
                    kind="free_text",
                    option_id=None,
                    free_text=free_text_pool[decision_idx % len(free_text_pool)],
                )

        # --- report --------------------------------------------------------
        events = session.latency.events
        click_events = [e for e in events if e.name == "click"]
        if len(click_events) != len(dispatch_log):
            print(
                f"WARNING: expected {len(dispatch_log)} click events, "
                f"saw {len(click_events)} — audit may be incomplete"
            )

        print()
        print("=" * 120)
        print("per-dispatch latency audit")
        print("=" * 120)
        header = (
            f"{'#':>3}  {'kind':<10}  {'click->disp':>12}  "
            f"{'budget':>12}  {'excess(ms)':>10}  {'spec':>6}  parent..dest"
        )
        print(header)
        print("-" * 120)

        violations = 0
        for i, info in enumerate(dispatch_log):
            if i >= len(click_events):
                print(f"#{i + 1}: NO CLICK EVENT")
                continue
            click = click_events[i]
            actual_kind = _classify(click)
            if actual_kind != info["kind"]:
                print(
                    f"WARN: dispatch #{i + 1} kind mismatch "
                    f"(sent={info['kind']}, audit={actual_kind})"
                )

            beat_ready = _first(
                events,
                "beat_ready",
                node_id=info["destination_id"],
            )
            displayed = _first(
                events,
                "displayed",
                node_id=info["destination_id"],
            )

            if info["kind"] == "free_text":
                predictable_ts = click.ts
            else:
                spec_spawn = _first(
                    events,
                    "speculation_spawn",
                    parent_id=info["parent_id"],
                    option_id=info["click_option_id"],
                )
                predictable_ts = spec_spawn.ts if spec_spawn is not None else click.ts

            if displayed is None:
                print(
                    f"{i + 1:>3}  {info['kind']:<10}  "
                    f"{'?':>12}  {'?':>12}  {'?':>10}  "
                    f"{'?':>6}  (no displayed event)"
                )
                continue

            click_to_disp_ms = (displayed.ts - click.ts) * 1000
            if beat_ready is None:
                budget_ms = 0.0
                spec_label = "PRE"
            else:
                floor_ref = max(click.ts, predictable_ts)
                budget_ms = max(0.0, (beat_ready.ts - floor_ref) * 1000)
                if beat_ready.ts <= click.ts:
                    spec_label = "DONE"
                elif info["kind"] == "free_text":
                    spec_label = "FG"
                else:
                    spec_label = "FLT"
            excess_ms = click_to_disp_ms - budget_ms
            print(
                f"{i + 1:>3}  {info['kind']:<10}  "
                f"{click_to_disp_ms:>10.1f}ms  {budget_ms:>10.1f}ms  "
                f"{excess_ms:>+8.1f}ms  {spec_label:>6}  "
                f"{(info['parent_id'] or 'NA')[-6:]}..{info['destination_id'][-6:]}"
            )
            if excess_ms > INVARIANT_TOLERANCE_MS:
                violations += 1

        print("-" * 120)
        print(
            f"violations: {violations}/{len(dispatch_log)} (tolerance: {INVARIANT_TOLERANCE_MS} ms)"
        )
        return 0 if violations == 0 else 2
    finally:
        await _cleanup_session(session)


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
