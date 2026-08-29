"""50-turn soak of the per-entity redraw scheduler.

Drives a full session through 50 turns mixing custom (free-text),
suggested (option), and undo commands, plus explicit character /
environment rerenders, against a deliberately SLOW image client so the
render worker falls behind and several entities are stale with
*different* staleness ages at once.

On every selection the scheduler makes, an INDEPENDENT reference picker
— written straight from the contract, not from the implementation —
recomputes the entity that *should* render next (highest score; explicit
ahead of on-screen staleness ahead of speculative; characters over the
background on a tie; lowest id last). Any divergence is a "did not pick
the most stale image" bug and fails the test.

It also pins "always rendering something": whenever the reference picker
finds schedulable stale work, the scheduler must return a pick (never
idle while stale enqueued work remains), and after the run fully drains
no on-screen entity is left stale.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from lucidium.api.handlers import HandlerContext, build_default_registry
from lucidium.api.messages import Envelope, InterviewStep, MessageType
from lucidium.orchestration.assets import (
    _active_lighting,
    _background_prompt_hash,
    _portrait_prompt_hash,
    _resolve_active_environment_id,
)
from lucidium.orchestration.render_scheduler import RenderScheduler
from lucidium.orchestration.session import Session

_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
    b"\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


# --- LLM: scripted preamble, then varied beats forever --------------------


def _player_payload() -> dict:
    return {
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
    }


def _descriptor(cid: str, name: str) -> dict:
    return {
        "id": cid,
        "name": name,
        "description": "a face in the crowd",
        "gender": "male",
        "age": 44,
        "ethnicity": "local",
        "skin": "weathered",
        "hair_color": "iron",
        "hairstyle": "short",
        "eye_color": "hazel",
        "build": "lean",
        "bust": "n/a",
        "outfit": "oilskin",
        "pose": "standing",
        "expression": "guarded",
    }


def _world_init_payload() -> str:
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
                    }
                ],
                "options": [
                    {"id": "a", "text": "Walk to the inn."},
                    {"id": "b", "text": "Wait at the dock."},
                ],
            },
            "player_character": _player_payload(),
        }
    )


def _opt(items: list[str]) -> str:
    return json.dumps({"options": items})


class _VariedLlm:
    """Scripted interview + opening, then deterministically-varied beats
    that introduce characters, change their outfits, and move the scene
    — so multiple portraits and the background go stale over time."""

    def __init__(self) -> None:
        self._preamble = [
            _opt([f"c{i}" for i in range(6)]),
            _opt([f"n{i}" for i in range(8)]),
            _world_init_payload(),
        ]
        self._count = 0
        self.roster: list[str] = []

    async def complete(self, *_a, **_kw) -> AsyncIterator[str]:
        nv = self._preamble.pop(0) if self._preamble else self._next_beat()

        async def gen() -> AsyncIterator[str]:
            yield nv

        return gen()

    def _next_beat(self) -> str:
        c = self._count
        self._count += 1
        beat = {
            "text": f"Beat {c}.",
            "speaker_id": None,
            "entering_character_ids": [],
            "leaving_character_ids": [],
            "new_characters": [],
            "location_id": None,
            "location_prompt": None,
            "character_changes": [],
        }
        if c % 3 == 0 and len(self.roster) < 5:
            cid = f"npc-{c}"
            self.roster.append(cid)
            beat["entering_character_ids"] = [cid]
            beat["new_characters"] = [_descriptor(cid, f"NPC{c}")]
        if c % 4 == 1 and self.roster:
            target = self.roster[c % len(self.roster)]
            beat["character_changes"] = [
                {"character_id": target, "field": "outfit", "new_value": f"coat-{c}"},
            ]
        if c % 5 == 2:
            beat["location_prompt"] = f"chamber {c // 5}"
        return json.dumps(
            {
                "beats": [beat],
                "options": [{"id": "a", "text": "go a"}, {"id": "b", "text": "go b"}],
            }
        )


class _SlowImage:
    """Records every render and sleeps a little so the single render
    worker falls behind rapid turns — that is what makes staleness
    accumulate to different ages across entities."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def generate(self, workflow, prompts, *, seed):
        self.calls.append((workflow, dict(prompts), seed))
        await asyncio.sleep(0.01)
        return _PNG_BYTES


# --- Independent reference picker (from the contract, not the impl) -------


def _is_stale(game, eid: str, kind: str, lighting: str) -> bool:
    if kind == "portrait":
        ch = game.characters.get(eid)
        if ch is None:
            return False
        desired = _portrait_prompt_hash(game.world, ch, lighting)
        return (not ch.images) or ch.images[0].prompt_hash != desired
    env = game.environments.get(eid)
    if env is None:
        return False
    if not env.image_path or not Path(env.image_path).exists():
        return True
    return env.prompt_hash != _background_prompt_hash(game.world, env)


# Redraw-reason priority — the PRIMARY ordering key within a tier, per
# the scheduler contract: new character > new scene > clothing >
# expression (higher renders first). Score only breaks ties WITHIN a
# reason tier. Mirrors RenderScheduler._redraw_priority straight from
# the documented contract (module docstring), not the implementation.
_R_NEW_CHARACTER = 4
_R_NEW_SCENE = 3
_R_CLOTHING = 2
_R_EXPRESSION = 1


def _reason_priority(game, eid: str, kind: str) -> int:
    if kind == "background":
        return _R_NEW_SCENE
    ch = game.characters.get(eid)
    if ch is None:
        return _R_EXPRESSION
    latest = next((im for im in ch.images if Path(im.path).exists()), None)
    if latest is None:
        return _R_NEW_CHARACTER
    snap = latest.attributes_snapshot or {}

    def changed(field: str) -> bool:
        return str(snap.get(field, "")) != str(getattr(ch, field, "") or "")

    if any(changed(f) for f in ("outfit", "pose", "hair_color", "eye_color", "effects", "decals")):
        return _R_CLOTHING
    if changed("expression"):
        return _R_EXPRESSION
    # Stale for an un-snapshotted reason (e.g. room lighting) → clothing tier.
    return _R_CLOTHING


def _expected_pick(sched: RenderScheduler):
    """The entity the scheduler SHOULD pick, derived only from the
    documented contract: explicit > on-screen > speculative; within a
    tier, highest redraw-reason PRIORITY (new character > new scene >
    clothing > expression), then highest accumulated staleness score,
    then lowest id. A portrait's priority is never 3 and a background's
    is always 3, so a portrait and the background can never tie on
    (priority, score) — the surviving ties are between same-tier
    portraits, broken by lowest id (mirrors ``_better``)."""
    game = sched._session.game
    if game is None:
        return None
    lighting = _active_lighting(game)
    char_ids = [
        cid
        for cid in game.on_stage
        if (c := game.characters.get(cid)) is not None and not c.removed and not c.is_player
    ]
    env_id = _resolve_active_environment_id(game)
    onscreen = set(char_ids) | ({env_id} if env_id else set())

    def classify(eid: str):
        if eid in game.characters:
            return "portrait"
        if eid in game.environments:
            return "background"
        return None

    explicit, onscreen_c = [], []
    for eid, score in sched._scores.items():
        kind = classify(eid)
        if kind is None or not _is_stale(game, eid, kind, lighting):
            continue
        rank = (_reason_priority(game, eid, kind), score)
        if eid in sched._explicit:
            explicit.append((rank, eid, kind))
        elif eid in onscreen:
            onscreen_c.append((rank, eid, kind))

    def best(cands):
        if not cands:
            return None
        # highest priority, then highest score, then lowest id
        chosen = min(cands, key=lambda x: (-x[0][0], -x[0][1], x[1]))
        return (chosen[2], chosen[1])

    return best(explicit) or best(onscreen_c) or _best_spec(sched, game, lighting)


def _best_spec(sched: RenderScheduler, game, lighting: str):
    spec = []
    for cid, score in sched._spec.items():
        ch = game.characters.get(cid)
        if ch is None or ch.is_player:
            continue
        if not _is_stale(game, cid, "portrait", lighting):
            continue
        spec.append((score, cid))
    if not spec:
        return None
    chosen = min(spec, key=lambda x: (-x[0], x[1]))
    return ("portrait", chosen[1])


# --- Driver ---------------------------------------------------------------


async def _drain(handler_result) -> list:
    out = []
    async for msg in handler_result:
        out.append(msg)
    return out


async def _settle(session: Session, max_iter: int = 400) -> None:
    """Yield until every background task (asset pump, speculation, the
    foreground stream drain) has settled."""
    for _ in range(max_iter):
        await asyncio.sleep(0.01)
        spec = list((getattr(session, "_speculative_tasks", None) or {}).values())
        asset = list(getattr(session, "_asset_tasks", None) or [])
        fg = getattr(session, "_foreground_stream_task", None)
        live = [t for t in (*spec, *asset, fg) if t is not None and not t.done()]
        if not live:
            return


async def _walk_interview(ctx: HandlerContext, registry) -> None:
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


def _current_option_id(session: Session) -> str | None:
    if session.game is None or session.game.current_node_id is None:
        return None
    node = session.game.dialog_tree.nodes.get(session.game.current_node_id)
    if node is None or not node.options:
        return None
    return node.options[0].id


def _live_npc_id(session: Session) -> str | None:
    if session.game is None:
        return None
    for cid, ch in session.game.characters.items():
        if not ch.is_player and not ch.removed:
            return cid
    return None


@pytest.mark.asyncio
async def test_50_turns_always_picks_most_stale(tmp_app_data: Path) -> None:
    image = _SlowImage()
    llm = _VariedLlm()
    session = Session(llm_client=llm, image_client=image, emit=lambda *_a: None)
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    # --- instrument the scheduler's selection -----------------------------
    mismatches: list[tuple] = []
    max_score_seen = [0]
    pick_count = [0]
    original_select = RenderScheduler._select

    def checked_select(self):
        expected = _expected_pick(self)
        if self._scores:
            max_score_seen[0] = max(max_score_seen[0], max(self._scores.values()))
        actual = original_select(self)
        pick_count[0] += 1
        if actual != expected:
            mismatches.append((expected, actual, dict(self._scores), dict(self._spec)))
        return actual

    RenderScheduler._select = checked_select
    try:
        await _walk_interview(ctx, registry)
        await _drain(
            registry.dispatch(
                Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
                ctx,
            )
        )
        await _settle(session)

        # A rotating script of every command kind the player can issue.
        plan = [
            "free_text",
            "option",
            "rerender_char",
            "option",
            "free_text",
            "undo",
            "option",
            "rerender_env",
            "rerender_char",
            "free_text",
        ]
        # Issue turns in BURSTS, draining only at burst boundaries. The
        # LLM mock is instant but renders take ~10 ms, so during a burst
        # the worker falls behind and several entities pile up stale with
        # DIFFERENT ages — that is what exercises "pick the most stale
        # among several", not just one-at-a-time renders.
        BURST = 5
        for turn in range(50):
            cmd = plan[turn % len(plan)]
            if cmd == "option" and _current_option_id(session) is not None:
                await _drain(
                    registry.dispatch(
                        Envelope(
                            type=MessageType.c2s_play_advance,
                            payload={"option_id": _current_option_id(session)},
                        ),
                        ctx,
                    )
                )
            elif cmd == "undo" and session.undo_stack:
                await _drain(
                    registry.dispatch(
                        Envelope(type=MessageType.c2s_play_undo, payload={}),
                        ctx,
                    )
                )
            elif cmd == "rerender_char" and _live_npc_id(session) is not None:
                cid = _live_npc_id(session)
                await _drain(
                    registry.dispatch(
                        Envelope(
                            type=MessageType.c2s_edit_character,
                            payload={
                                "character_id": cid,
                                "field": "outfit",
                                "value": f"rerendered cloak {turn}",
                            },
                        ),
                        ctx,
                    )
                )
                await _drain(
                    registry.dispatch(
                        Envelope(
                            type=MessageType.c2s_edit_character_rerender,
                            payload={"character_id": cid},
                        ),
                        ctx,
                    )
                )
            elif cmd == "rerender_env" and session.game.environments:
                env_id = _resolve_active_environment_id(session.game) or next(
                    iter(session.game.environments)
                )
                await _drain(
                    registry.dispatch(
                        Envelope(
                            type=MessageType.c2s_edit_environment_rerender,
                            payload={"environment_id": env_id},
                        ),
                        ctx,
                    )
                )
            else:  # free_text (and fallback for any guarded-out command)
                await _drain(
                    registry.dispatch(
                        Envelope(
                            type=MessageType.c2s_play_free_text,
                            payload={"text": f"I improvise on turn {turn}."},
                        ),
                        ctx,
                    )
                )
            # Let the loop schedule the pump a step, but DON'T fully
            # drain mid-burst — staleness must be allowed to accumulate.
            await asyncio.sleep(0)
            if turn % BURST == BURST - 1:
                await _settle(session)
        await _settle(session)
    finally:
        RenderScheduler._select = original_select

    # 1) Every selection picked the most-stale schedulable entity.
    assert not mismatches, (
        f"{len(mismatches)} selection(s) did not pick the most-stale entity. "
        f"First: expected={mismatches[0][0]} actual={mismatches[0][1]} "
        f"scores={mismatches[0][2]} spec={mismatches[0][3]}"
    )

    # 2) The run actually exercised the scheduler, with real accumulation
    #    (staleness reaching >1 means the "most stale among several" path
    #    was hit, not just trivial one-at-a-time renders).
    assert pick_count[0] > 30, f"scheduler barely ran: {pick_count[0]} selections"
    assert max_score_seen[0] > 1, (
        f"staleness never accumulated past 1 (max={max_score_seen[0]}); the "
        f"'most stale among several' path was not exercised"
    )
    assert len(image.calls) > 20, f"too few renders: {len(image.calls)}"

    # 3) After the final drain nothing on-screen is left stale — the
    #    scheduler always rendered whatever was pending.
    await _settle(session)
    game = session.game
    lighting = _active_lighting(game)
    leftover_stale = [
        cid
        for cid in game.on_stage
        if (c := game.characters.get(cid)) is not None
        and not c.removed
        and not c.is_player
        and _is_stale(game, cid, "portrait", lighting)
    ]
    env_id = _resolve_active_environment_id(game)
    if env_id is not None and _is_stale(game, env_id, "background", lighting):
        leftover_stale.append(env_id)
    assert not leftover_stale, (
        f"on-screen entities left stale after drain (scheduler stopped "
        f"rendering pending work): {leftover_stale}"
    )
