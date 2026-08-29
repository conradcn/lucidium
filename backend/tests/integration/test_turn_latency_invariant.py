"""Per-dispatch latency-invariant test for the end-of-beat -> next-line
pipeline.

For each of 10 decision iterations, the test:

  1. Walks Continue at ``PACING_S`` per click until landing on a node
     with options (the "tail" of the current chain).
  2. Waits one more ``PACING_S`` interval.
  3. Makes the decision: a selected option (odd iterations) or a free-
     text submission (even iterations).

The pacing gives speculation time to fire and complete between clicks
(``LLM_CALL_DELAY_S`` < ``PACING_S``), so selected decisions should
consistently hit pre-generated chains; free-text decisions always
require a fresh foreground LLM call.

Every dispatch (Continue + decision) is checked against the invariant::

    displayed_at - click_at <= max(0, llm_end - max(click_at, predictable_at))
                               + INVARIANT_TOLERANCE_MS

i.e. the player never waits longer than the LLM call counted from the
moment the engine could have started speculating. The ``max(0, ...)``
floor handles tree-walks where the destination beat was already in
the tree at click time — there the player should pay zero, not a
negative budget.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from lucidium.api.handlers import HandlerContext, build_default_registry
from lucidium.api.messages import (
    Envelope,
    InterviewStep,
    MessageType,
)
from lucidium.orchestration.session import Session

# Per-LLM-call deterministic wall time. Kept short so the mocked test
# stays under ~15 s even with a real-time pacing wait between clicks.
LLM_CALL_DELAY_S = 0.2

# Pacing between clicks. Mirrors the "every 1 s" cadence the live
# script uses, scaled down for the mock so the test still finishes
# in CI-friendly time. Must remain strictly greater than
# ``LLM_CALL_DELAY_S`` so speculation has time to complete between
# successive clicks.
PACING_S = 0.3

# Tolerance for non-LLM overhead in the invariant.
INVARIANT_TOLERANCE_MS = 80

# Number of decision iterations (walk-to-tail + selected/free-text).
DECISIONS = 10


# ---------------------------------------------------------------------------
# Mocked LLM: delays a fixed amount then returns a 3-beat chain with two
# options on the tail beat. The intermediate beats have no options so the
# player walks Continue twice before reaching the decision point.
# ---------------------------------------------------------------------------


def _branch_payload(text_prefix: str, *, beats: int = 3, with_options: bool = True) -> str:
    beat_list = [
        {
            "text": f"{text_prefix} (beat {i + 1}).",
            "speaker_id": None,
            "entering_character_ids": [],
            "leaving_character_ids": [],
            "new_characters": [],
            "location_id": None,
            "location_prompt": None,
            "character_changes": [],
        }
        for i in range(beats)
    ]
    opts: list[dict[str, str]] = []
    if with_options:
        opts = [
            {"id": "a", "text": "Take the lit path."},
            {"id": "b", "text": "Slip down the alley."},
        ]
    return json.dumps({"beats": beat_list, "options": opts})


def _opt_list(items: list[str]) -> str:
    return json.dumps({"options": items})


def _world_init_with_two_options() -> str:
    return json.dumps(
        {
            "game_name": "Salt Lantern",
            "overall_plot_direction": "Investigate the keeper.",
            "active_plot_threads": [],
            "opening_node": {
                "beats": [
                    {
                        "text": "The harbor wakes slow.",
                        "speaker_id": None,
                        "entering_character_ids": [],
                        "leaving_character_ids": [],
                        "new_characters": [],
                        "location_id": "harbor",
                        "location_prompt": "stone harbor at dawn",
                        "character_changes": [],
                    },
                ],
                "options": [
                    {"id": "a", "text": "Walk to the inn."},
                    {"id": "b", "text": "Climb the cliff."},
                ],
            },
            "player_character": {
                "name": "Iris",
                "description": "wry archivist",
                "gender": "female",
                "age": 28,
                "ethnicity": "local",
                "skin": "pale",
                "hair_color": "auburn",
                "hairstyle": "braid",
                "eye_color": "grey",
                "build": "slight",
                "bust": "moderate",
                "outfit": "wool coat",
                "pose": "standing",
                "expression": "alert",
            },
        }
    )


class _DelayingLlm:
    """Returns deterministic chains; every ``complete()`` call costs
    ``LLM_CALL_DELAY_S`` of wall time so the audit has a measurable
    floor against which to check the invariant."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []
        self._scripted: list[str] = []
        self._next_idx = 0

    def script(self, payloads: list[str]) -> None:
        self._scripted = list(payloads)

    async def complete(self, prompt, *_a, **_kw):
        self.calls.append(prompt)
        if self._scripted:
            body = self._scripted.pop(0)
        else:
            self._next_idx += 1
            body = _branch_payload(
                f"Beat #{self._next_idx}",
                beats=3,
                with_options=True,
            )
        await asyncio.sleep(LLM_CALL_DELAY_S)

        async def gen() -> AsyncIterator[str]:
            yield body

        return gen()


class _NullImage:
    async def generate(self, *_a, **_kw) -> bytes:
        return b""


async def _drain(handler_result) -> list:
    out = []
    async for msg in handler_result:
        out.append(msg)
    return out


async def _run_interview(ctx: HandlerContext, registry, llm: _DelayingLlm) -> None:
    llm.script(
        [
            _opt_list([f"c{i}" for i in range(6)]),  # character_description
            _opt_list([f"n{i}" for i in range(8)]),  # name
            _world_init_with_two_options(),  # confirm
        ]
    )
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_start, payload={}),
            ctx,
        )
    )
    for step, ans in [
        (InterviewStep.setting, "stone harbor"),
        (InterviewStep.visual_style, "ink wash"),
        (InterviewStep.genre, "mystery"),
        (InterviewStep.character_description, "wry archivist"),
        (InterviewStep.name, "Iris"),
    ]:
        await _drain(
            registry.dispatch(
                Envelope(
                    type=MessageType.c2s_new_game_answer,
                    payload={"step": step.value, "answer": ans, "is_free_text": False},
                ),
                ctx,
            )
        )
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )


async def _drain_speculation(session, *, max_iters: int = 200) -> None:
    """Yield until speculative tasks complete, bounded so a stuck task
    can't hang the test."""
    for _ in range(max_iters):
        await asyncio.sleep(0.05)
        spec = getattr(session, "_speculative_tasks", None) or {}
        live = [t for t in spec.values() if not t.done()]
        if not live:
            return


# ---------------------------------------------------------------------------
# Audit-event matching helpers
# ---------------------------------------------------------------------------


def _first(events, name, *, parent_id=None, option_id=None, after=None, node_id=None):
    for e in events:
        if e.name != name:
            continue
        if parent_id is not None and e.parent_id != parent_id:
            continue
        if option_id is not None and e.option_id != option_id:
            continue
        if node_id is not None and e.node_id != node_id:
            continue
        if after is not None and e.ts < after:
            continue
        return e
    return None


def _classify(click_event) -> str:
    """Return one of {"continue", "selected", "free_text"} for a click."""
    if click_event.extra.get("kind") == "free_text":
        return "free_text"
    if click_event.option_id is None:
        return "continue"
    return "selected"


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_dispatch_latency_invariant_with_pacing(
    tmp_app_data: Path,
) -> None:
    llm = _DelayingLlm()
    session = Session(llm_client=llm, image_client=_NullImage())
    session.latency.enable()
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _run_interview(ctx, registry, llm)
    await _drain_speculation(session)

    # Reset the audit so the interview's events don't pollute the
    # decision loop. The game state remains; only the recorder clears.
    session.latency.clear()

    free_text_pool = [
        "I take a careful look around.",
        "I keep going, but stay alert.",
        "I follow the sound.",
        "I ask if anyone has news.",
        "I check my pockets.",
    ]

    dispatch_log: list[dict] = []

    async def _do_dispatch(*, kind: str, option_id: str | None, free_text: str | None):
        # Pacing wait BEFORE every click so speculation has time to
        # progress (LLM_CALL_DELAY_S < PACING_S). The audit treats each
        # dispatch as a "turn" — the pacing wait is the player's
        # think-time, not part of the invariant.
        await asyncio.sleep(PACING_S)
        node_before = session.game.current_node_id
        if free_text is not None:
            payload: dict = {"text": free_text}
            msg_type = MessageType.c2s_play_free_text
        else:
            payload = {"option_id": option_id}
            msg_type = MessageType.c2s_play_advance
        await _drain(registry.dispatch(Envelope(type=msg_type, payload=payload), ctx))
        node_after = session.game.current_node_id
        assert node_after != node_before, (
            f"dispatch ({kind}, option={option_id}, "
            f"free_text={free_text}): current_node_id did not advance"
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

    for decision_idx in range(1, DECISIONS + 1):
        # Walk Continue at PACING_S/click until we reach a node with
        # options. The opening node and every chain-tail have options;
        # the chain's intermediate beats do not.
        while True:
            node = session.game.dialog_tree.nodes[session.game.current_node_id]
            if node.options:
                break
            await _do_dispatch(kind="continue", option_id=None, free_text=None)

        # Decision: 5 selected (odd) and 5 free-text (even).
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

    # ----- analysis -------------------------------------------------------
    events = session.latency.events
    click_events = [e for e in events if e.name == "click"]
    assert len(click_events) == len(dispatch_log), (
        f"expected {len(dispatch_log)} click events, saw {len(click_events)}"
    )

    failures: list[str] = []
    rows: list[str] = []
    rows.append(
        f"{'#':>3}  {'kind':<10}  {'click->disp':>11}  "
        f"{'budget':>11}  {'excess(ms)':>10}  {'spec':>6}  parent..dest"
    )
    for i, info in enumerate(dispatch_log):
        click = click_events[i]
        actual_kind = _classify(click)
        # Sanity: handler-recorded kind should match what the test sent.
        # ("continue" and "selected" both come in as advance-kind clicks
        # at the handler layer; we distinguish by whether option_id is
        # set. The test's ``kind`` and the audit's ``actual_kind`` must
        # agree so the report can be trusted.)
        assert actual_kind == info["kind"], (
            f"dispatch #{i + 1}: test recorded kind={info['kind']} but "
            f"audit recorded kind={actual_kind} for click event"
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
            predictable = click
        else:
            spec_spawn = _first(
                events,
                "speculation_spawn",
                parent_id=info["parent_id"],
                option_id=info["click_option_id"],
            )
            predictable = spec_spawn if spec_spawn is not None else click

        assert displayed is not None, (
            f"dispatch #{i + 1}: no displayed event for destination {info['destination_id']}"
        )

        click_to_disp_ms = (displayed.ts - click.ts) * 1000
        if beat_ready is None:
            # Destination beat was generated by a chain that committed
            # before the dispatch (so its beat_ready event isn't in
            # the post-clear log) — at click time the beat was already
            # in the tree, so the player's LLM-cost budget is zero.
            budget_ms = 0.0
            spec_label = "PRE"
        else:
            floor_ref = max(click.ts, predictable.ts)
            budget_ms = max(0.0, (beat_ready.ts - floor_ref) * 1000)
            if beat_ready.ts <= click.ts:
                spec_label = "DONE"
            elif info["kind"] == "free_text":
                spec_label = "FG"
            else:
                spec_label = "FLT"
        excess_ms = click_to_disp_ms - budget_ms

        rows.append(
            f"{i + 1:>3}  {info['kind']:<10}  {click_to_disp_ms:>9.1f}ms  "
            f"{budget_ms:>9.1f}ms  {excess_ms:>+8.1f}ms  {spec_label:>6}  "
            f"{(info['parent_id'] or 'NA')[-6:]}..{info['destination_id'][-6:]}"
        )

        if excess_ms > INVARIANT_TOLERANCE_MS:
            failures.append(
                f"dispatch #{i + 1} ({info['kind']}): "
                f"click->displayed = {click_to_disp_ms:.1f} ms, "
                f"budget = {budget_ms:.1f} ms, "
                f"excess = {excess_ms:.1f} ms "
                f"(> {INVARIANT_TOLERANCE_MS} ms tolerance)"
            )

    print("\n=== per-dispatch latency audit ===")
    for row in rows:
        print(row)

    if failures:
        pytest.fail(
            "latency invariant violated for one or more dispatches:\n  " + "\n  ".join(failures)
        )
