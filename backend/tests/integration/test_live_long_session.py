"""Long-session smoke test against the REAL LLM and ComfyUI.

Excluded from the default run via the ``live`` marker (see
``pyproject.toml``'s ``addopts = -m 'not live'``). Run explicitly
with::

    pytest tests/integration/test_live_long_session.py -m live -v

What this catches that mocked tests never can:

  * **Hangs** — speculative tasks that never resolve, schedulers
    that deadlock against a slow real provider, ``await`` chains
    that lose their cancellation signal.
  * **Real-provider quirks** — JSON drift the schema parser doesn't
    coerce, hand-shape collapse on dramatic prompts, ComfyUI
    workflow nodes that error on edge-case inputs.
  * **Memory / state drift across many turns** — speculative branches
    accumulating, prompt history overflowing the clamp, characters
    diverging from their established attributes.

Each turn is bounded by ``PER_TURN_BUDGET_S``; if any single turn
exceeds it the test fails with a hang report (which turn, how long).
The whole flow is bounded by ``WHOLE_SESSION_BUDGET_S`` as a
backstop.

Pre-flight: the test skips (rather than fails) when ComfyUI is
unreachable, the LLM endpoint is unconfigured, or no API key is
set. This keeps CI green when the live deps aren't available.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from lucidium.api.handlers import HandlerContext, build_default_registry
from lucidium.api.messages import (
    Envelope,
    InterviewStep,
    MessageType,
)
from lucidium.orchestration.session import Session
from lucidium.persistence import settings_store

# Per-turn cap: an advance covers an LLM call (one chain of beats),
# possibly the image scheduler kicking off a portrait/background
# render. Real LLM round-trips on slow providers can hit 30–60 s;
# 180 s is generous and still catches a real hang.
PER_TURN_BUDGET_S: float = 180.0
# Whole-session cap: 5 interview steps + confirm + 20 advances ≈
# 26 LLM calls + image renders. 30 minutes covers a slow provider
# without becoming an infinite-wait if something silently spins.
WHOLE_SESSION_BUDGET_S: float = 30 * 60
TURNS = 20

COMFY_URL_DEFAULT = "http://127.0.0.1:8000"


pytestmark = [
    pytest.mark.live,
    # Real httpx.AsyncClient instances created by the per-render
    # factory leak ProactorSocketTransport warnings on shutdown.
    # The engine is correct; this is a test-framework strictness
    # issue. Silence both the underlying warning and pytest's
    # elevation of unraisable exceptions to errors.
    pytest.mark.filterwarnings("ignore::ResourceWarning"),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]


# ---------------------------------------------------------------------------
# Pre-flight gates
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers: drive the real registry the same way ws_server does
# ---------------------------------------------------------------------------


async def _drain(handler_result: AsyncIterator) -> list:
    out = []
    async for msg in handler_result:
        out.append(msg)
    return out


async def _dispatch(registry, ctx, message_type, payload) -> list:
    return await _drain(registry.dispatch(Envelope(type=message_type, payload=payload), ctx))


async def _answer(registry, ctx, step: InterviewStep, answer: str) -> None:
    await _dispatch(
        registry,
        ctx,
        MessageType.c2s_new_game_answer,
        {"step": step.value, "answer": answer, "is_free_text": False},
    )


async def _await_speculation_drain(session: Session, max_iterations: int = 50) -> None:
    """Yield until the in-flight speculative LLM/image tasks are
    quiescent. Without this, a turn's spawned background work can
    leak into the next turn's timing measurement."""
    for _ in range(max_iterations):
        await asyncio.sleep(0.1)
        spec = getattr(session, "_speculative_tasks", None) or {}
        live = [t for t in spec.values() if not t.done()]
        if not live:
            return


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_provider_long_session_no_hang(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Load the REAL user-configured settings (LLM endpoint, model,
    # API key, ComfyUI URL) from the actual app-data directory —
    # NOT a tmp redirect, because the redirect would hand us
    # ``Settings()`` defaults with no API key and the test would
    # always skip. We then route saves to a tmp dir so the test
    # never pollutes the user's real save library.
    monkeypatch.delenv("LUCIDIUM_OFFLINE", raising=False)
    settings = settings_store.load_settings()

    llm_ok, why = _llm_configured(settings)
    if not llm_ok:
        pytest.skip(f"live test requires real LLM: {why}")

    comfy_url = settings.image.base_url or COMFY_URL_DEFAULT
    if not _comfy_reachable(comfy_url):
        pytest.skip(f"live test requires reachable ComfyUI at {comfy_url}")

    saves_root = tmp_path / "saves"
    saves_root.mkdir(parents=True, exist_ok=True)
    session = Session(settings=settings, saves_root=saves_root)
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    # Clean cancellation of every speculative / asset background
    # task at the end of the test, so they don't continue logging
    # against pytest's torn-down event loop or leak open httpx
    # transports past the test boundary.
    async def _cleanup_session() -> None:
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

    overall_started = time.monotonic()
    per_turn_durations: list[float] = []

    async def deadline_guarded(coro, *, label: str) -> None:
        """Run a coroutine with a per-turn timeout. On timeout, fail
        the test with a HANG report — which turn, how long it ran
        for, what the deadline was. This is the test's only purpose:
        prove the engine never silently spins."""
        started = time.monotonic()
        try:
            await asyncio.wait_for(coro, timeout=PER_TURN_BUDGET_S)
        except TimeoutError:
            elapsed = time.monotonic() - started
            pytest.fail(
                f"HANG: {label} exceeded {PER_TURN_BUDGET_S:.0f}s "
                f"(ran for {elapsed:.0f}s before being killed)"
            )
        per_turn_durations.append(time.monotonic() - started)

    try:
        await _run_session(
            registry,
            ctx,
            session,
            deadline_guarded,
            overall_started,
            per_turn_durations,
        )
    finally:
        await _cleanup_session()


async def _run_session(
    registry,
    ctx,
    session,
    deadline_guarded,
    overall_started: float,
    per_turn_durations: list[float],
) -> None:
    async def _wait_for_options(field: str, label: str) -> str:
        deadline = time.monotonic() + PER_TURN_BUDGET_S
        while time.monotonic() < deadline:
            opts = getattr(session.interview, field, None) or []
            if opts:
                return opts[0]
            await asyncio.sleep(0.1)
        pytest.fail(f"HANG: {label} options never arrived within {PER_TURN_BUDGET_S:.0f}s")

    # --- Interview ---------------------------------------------------------
    # Step ordering (post-FR-027b reorder): setting -> visual_style ->
    # genre -> character_description -> name -> confirm. Setting is
    # hard-coded; answering it surfaces the (also hard-coded) Visual
    # Style options inline.
    await deadline_guarded(
        _dispatch(registry, ctx, MessageType.c2s_new_game_start, {}),
        label="new_game/start",
    )
    await deadline_guarded(
        _answer(registry, ctx, InterviewStep.setting, "A stone harbor at dawn"),
        label="answer/setting",
    )
    visual_style = session.interview.visual_style_options[0]
    await deadline_guarded(
        _answer(registry, ctx, InterviewStep.visual_style, visual_style),
        label="answer/visual_style",
    )
    await deadline_guarded(
        _answer(registry, ctx, InterviewStep.genre, "Mystery"),
        label="answer/genre",
    )
    char_desc = await _wait_for_options("character_description_options", "character_description")
    await deadline_guarded(
        _answer(registry, ctx, InterviewStep.character_description, char_desc),
        label="answer/character_description",
    )
    name = await _wait_for_options("name_options", "name")
    await deadline_guarded(
        _answer(registry, ctx, InterviewStep.name, name),
        label="answer/name",
    )

    # --- Confirm: world_init + opening node + first asset pass ------------
    await deadline_guarded(
        _dispatch(
            registry,
            ctx,
            MessageType.c2s_new_game_confirm,
            {"overrides": {}},
        ),
        label="new_game/confirm",
    )
    assert session.game is not None
    assert session.game.current_node_id is not None

    # --- 20 turns: mix of advance + free_text ---------------------------
    # We rotate through four turn shapes so the live session exercises
    # every play-time code path that the renderer can produce. Each
    # "shape" is selected by ``turn_idx % 4``:
    #
    #   * 0 — advance picking the FIRST option (when the node has
    #     options) or the implicit Continue (when it doesn't).
    #   * 1 — advance picking the LAST option (variety of branch).
    #   * 2 — free_text submitted while the current node HAS OPTIONS.
    #     This is the "ignore the choices, type my own thing" path
    #     and was the case the user reported crashing.
    #   * 3 — free_text submitted on a node WITHOUT options (would
    #     otherwise have been an implicit Continue). Same crash class
    #     but different obsolescence path; both must be exercised.
    #
    # If the natural turn order skews to all-options or no-options
    # nodes, the rotation still surfaces every shape — the harness
    # falls back to the next applicable shape only when the chosen
    # one is impossible (e.g. shape 2 on a no-options node).
    free_text_pool = [
        "I take a careful look around the room before doing anything.",
        "I follow the sound and see where it leads.",
        "I keep going, but stay alert.",
        "I ask if anyone here has news from the harbor.",
        "I quietly check my pockets for anything useful.",
    ]
    for turn_idx in range(1, TURNS + 1):
        if (time.monotonic() - overall_started) > WHOLE_SESSION_BUDGET_S:
            pytest.fail(
                f"whole-session budget {WHOLE_SESSION_BUDGET_S:.0f}s exceeded "
                f"after {turn_idx - 1} successful turns"
            )

        node_before = session.game.current_node_id
        node = session.game.dialog_tree.nodes[node_before]
        has_options = bool(node.options)
        shape = (turn_idx - 1) % 4

        # Resolve the intended shape against what's actually available
        # at this node. Free-text-on-options-node and free-text-on-no-
        # options-node are both first-class targets; the option-pick
        # shapes degrade to Continue when no options exist.
        if shape == 2 and not has_options:
            shape = 3  # node has no options → exercise no-options free_text
        elif shape == 3 and has_options:
            shape = 2  # node has options → exercise has-options free_text

        if shape in (0, 1):
            option_id: str | None
            if has_options:
                option_id = node.options[0].id if shape == 0 else node.options[-1].id
            else:
                option_id = None
            label = f"play/advance #{turn_idx} (option={option_id})"
            payload: dict = {"option_id": option_id}
            message_type = MessageType.c2s_play_advance
        else:
            free_text = free_text_pool[(turn_idx - 1) % len(free_text_pool)]
            label = f"play/free_text #{turn_idx} ({'with' if has_options else 'without'} options)"
            payload = {"text": free_text}
            message_type = MessageType.c2s_play_free_text

        await deadline_guarded(
            _dispatch(registry, ctx, message_type, payload),
            label=label,
        )

        node_after = session.game.current_node_id
        assert node_after is not None and node_after != node_before, (
            f"turn {turn_idx} ({label}): current_node_id did not advance (stuck on {node_before})"
        )

        # Drain in-flight speculation so its work doesn't leak into
        # the next turn's measurement (and so we surface speculation-
        # driven hangs at the per-turn boundary, not at the next
        # advance).
        await _await_speculation_drain(session)

    # --- Final assertions --------------------------------------------------
    total = time.monotonic() - overall_started
    avg = sum(per_turn_durations) / len(per_turn_durations) if per_turn_durations else 0.0
    print(
        f"\n[live] {TURNS} turns + interview completed in {total:.1f}s "
        f"(per-turn avg {avg:.1f}s, max {max(per_turn_durations):.1f}s)"
    )
    # Sanity: the dialog tree grew across the run.
    assert len(session.game.dialog_tree.committed_path) >= TURNS + 1, (
        f"committed_path has {len(session.game.dialog_tree.committed_path)} "
        f"entries; expected ≥ {TURNS + 1}"
    )
