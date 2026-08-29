"""Live retcon coverage test against the REAL LLM.

Excluded from the default run via the ``live`` marker. Run explicitly
with::

    pytest tests/integration/test_live_retcon.py -m live -v

The bug this guards against: a player issues a sweeping retcon
("the entire scene now takes place in a torrential downpour") and
the engine rewrites only the first 1–3 beats before stopping —
either because the LLM's response truncates against the default
max_tokens cap, or because the prompt lets the model emit a
"representative sample" instead of comprehensive coverage. Either
failure mode leaves the tail of the history un-retconned, which the
player reads as "rewrite is broken — most lines didn't change."

What this catches that mocked tests can't:
  * **Truncation against the real provider's token limits** — only
    visible when the LLM actually generates a long JSON payload and
    runs out of budget mid-response.
  * **Lazy-LLM partial coverage** — the model genuinely deciding
    "three examples is enough", which only surfaces with real
    inference, not with a hand-built fixture.
  * **JSON-shape drift on long outputs** — the model emitting
    valid JSON for the first N items and mangled JSON later (a
    pattern recorded fixtures don't simulate).

Pre-flight: the test skips (rather than fails) when the LLM endpoint
is unconfigured or no API key is set.
"""

from __future__ import annotations

import asyncio
import os
import time
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
from lucidium.persistence import settings_store

# How long to give a single LLM-bearing call (interview answer,
# advance, retcon). Retcon over a long history is the heaviest
# single call in the engine; 240s leaves headroom for slow providers
# without becoming an infinite-wait if something genuinely hangs.
PER_CALL_BUDGET_S: float = 240.0
# How many turns to play before issuing the retcon. Need enough that
# truncation is visible — 1-3 beats fits in any token budget. 10
# committed beats reliably forces the response to be long enough to
# expose the default-1024-token cap that caused the bug.
TURNS_BEFORE_RETCON: int = 10


pytestmark = [
    pytest.mark.live,
    pytest.mark.filterwarnings("ignore::ResourceWarning"),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]


# ---------------------------------------------------------------------------
# Pre-flight gates
# ---------------------------------------------------------------------------


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
# Helpers
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


async def _wait_for_options(session, field: str, label: str) -> str:
    deadline = time.monotonic() + PER_CALL_BUDGET_S
    while time.monotonic() < deadline:
        opts = getattr(session.interview, field, None) or []
        if opts:
            return opts[0]
        await asyncio.sleep(0.1)
    pytest.fail(f"HANG: {label} options never arrived within {PER_CALL_BUDGET_S:.0f}s")


async def _drive_until_history(
    registry,
    ctx,
    session: Session,
    *,
    setting: str,
    genre: str,
    turns: int,
) -> None:
    """Take a session from a fresh state through interview + ``turns``
    play advances. Picks the FIRST option each time for determinism so
    the retcon assertions don't have to guess which branch was taken."""
    await _dispatch(registry, ctx, MessageType.c2s_new_game_start, {})
    await _answer(registry, ctx, InterviewStep.setting, setting)
    visual_style = session.interview.visual_style_options[0]
    await _answer(registry, ctx, InterviewStep.visual_style, visual_style)
    await _answer(registry, ctx, InterviewStep.genre, genre)
    char_desc = await _wait_for_options(
        session, "character_description_options", "character_description"
    )
    await _answer(registry, ctx, InterviewStep.character_description, char_desc)
    name = await _wait_for_options(session, "name_options", "name")
    await _answer(registry, ctx, InterviewStep.name, name)
    await _dispatch(registry, ctx, MessageType.c2s_new_game_confirm, {"overrides": {}})

    assert session.game is not None
    assert session.game.current_node_id is not None

    for turn in range(turns):
        node = session.game.dialog_tree.nodes[session.game.current_node_id]
        option_id = node.options[0].id if node.options else None
        await _dispatch(
            registry,
            ctx,
            MessageType.c2s_play_advance,
            {"option_id": option_id},
        )
        assert session.game.current_node_id is not None, (
            f"turn {turn + 1} did not advance current_node_id"
        )


def _committed_beats(session: Session) -> list:
    if session.game is None:
        return []
    nodes = session.game.dialog_tree.nodes
    return [nodes[nid] for nid in session.game.dialog_tree.committed_path if nid in nodes]


def _snapshot_beat_texts(session: Session) -> dict[str, str]:
    return {b.id: (b.text or "") for b in _committed_beats(session)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_retcon_rewrites_most_relevant_beats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sweeping atmospheric retcon ("the entire scene now takes
    place in a torrential downpour") MUST rewrite the bulk of the
    committed beats — not just the first few. Pre-fix the default
    max_tokens=1024 cap truncated the LLM's JSON response after a
    handful of rewrites; this asserts the retcon now lands across
    the full history.

    Coverage threshold rationale: with ~10 committed beats, a
    correct retcon over a sweeping weather change should rewrite at
    least 4. The previous truncated behaviour landed 1-3. Picking 4
    as the floor catches truncation regressions without flaking on
    runs where the LLM legitimately judges some early beats (bare
    dialogue with no atmospheric cue) untouched."""
    monkeypatch.delenv("LUCIDIUM_OFFLINE", raising=False)
    settings = settings_store.load_settings()
    llm_ok, why = _llm_configured(settings)
    if not llm_ok:
        pytest.skip(f"live retcon test requires real LLM: {why}")

    saves_root = tmp_path / "saves"
    saves_root.mkdir(parents=True, exist_ok=True)

    # Stub image generation so the test doesn't need ComfyUI / SDXL
    # — retcon is an LLM-only feature; image clients aren't on its
    # critical path.
    class _NullImage:
        async def generate(self, *_a, **_kw) -> bytes:
            return b""

    session = Session(
        settings=settings,
        saves_root=saves_root,
        image_client=_NullImage(),
    )
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    try:
        await _drive_until_history(
            registry,
            ctx,
            session,
            setting="A stone harbor at dusk, lanterns just lit",
            genre="Mystery",
            turns=TURNS_BEFORE_RETCON,
        )

        before_texts = _snapshot_beat_texts(session)
        beat_count_before = len(before_texts)
        assert beat_count_before >= 6, (
            f"expected at least 6 committed beats before retcon, got {beat_count_before}"
        )

        # Retcon: the kind of sweeping change a player would actually
        # type. Tone, weather, and atmosphere are referenced across
        # most beats, so the rewrite SHOULD hit a majority of them.
        retcon_text = (
            "The entire scene is now happening in a torrential "
            "downpour. Rain is everywhere — sluicing off rooftops, "
            "soaking everyone's clothes, blurring the lantern light, "
            "drumming on stone. Every beat should reflect that the "
            "characters are wet, cold, and dealing with the rain."
        )
        await asyncio.wait_for(
            _dispatch(
                registry,
                ctx,
                MessageType.c2s_edit_history_retcon,
                {"instructions": retcon_text},
            ),
            timeout=PER_CALL_BUDGET_S,
        )

        after_texts = _snapshot_beat_texts(session)
        assert set(after_texts) == set(before_texts), (
            "retcon must not add or remove committed beats; only rewrite existing ones"
        )

        rewritten_ids = [nid for nid in before_texts if after_texts[nid] != before_texts[nid]]
        # Headline coverage assertion — this is the regression target.
        assert len(rewritten_ids) >= 4, (
            f"sweeping retcon rewrote only {len(rewritten_ids)}/"
            f"{beat_count_before} committed beats — likely the "
            f"max_tokens truncation regression. "
            f"Rewritten ids: {rewritten_ids}"
        )

        # Most rewrites should reflect the new reality. Any beat
        # rewritten by the retcon must mention rain / wet / soaked /
        # downpour at least once — otherwise the LLM accepted the
        # rewrite slot but didn't actually apply the instruction.
        rain_words = (
            "rain",
            "wet",
            "soak",
            "downpour",
            "drench",
            "drumming",
            "puddl",
            "pour",
            "torrent",
            "sluic",
        )
        off_target = []
        for nid in rewritten_ids:
            text = after_texts[nid].lower()
            if not any(w in text for w in rain_words):
                off_target.append((nid, after_texts[nid]))
        # Allow at most one off-target rewrite (a beat the LLM
        # decided to tweak for an unrelated reason); more than that
        # means the retcon prompt isn't constraining the model.
        assert len(off_target) <= 1, (
            f"{len(off_target)} of {len(rewritten_ids)} rewritten "
            f"beats don't mention the rain. Off-target rewrites:\n"
            + "\n".join(f"  {nid}: {text!r}" for nid, text in off_target)
        )

    finally:
        # Drain any in-flight async work before the loop tears down.
        for attr in (
            "_char_desc_task",
            "_world_init_task",
            "_preview_bg_task",
            "_preview_guide_task",
            "_pc_portrait_task",
            "_name_options_task",
        ):
            t = getattr(session, attr, None)
            if t is not None and not t.done():
                t.cancel()
        for spec in (getattr(session, "_speculative_tasks", None) or {}).values():
            if not spec.done():
                spec.cancel()
        for asset in getattr(session, "_asset_tasks", None) or []:
            if not asset.done():
                asset.cancel()


@pytest.mark.asyncio
async def test_live_retcon_propagates_character_wardrobe_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A character-targeted retcon ("the player is now wearing a
    bright red leather jacket throughout") MUST update both the
    character record AND any beats that explicitly describe what
    the player is wearing.

    This is a SECOND axis of completeness: the first test guards
    beat coverage in the ``rewritten_beats`` array; this one guards
    that ``character_updates`` actually fires for the player on a
    wardrobe instruction (it was being silently dropped pre-fix when
    the truncated response cut off the character_updates section)."""
    monkeypatch.delenv("LUCIDIUM_OFFLINE", raising=False)
    settings = settings_store.load_settings()
    llm_ok, why = _llm_configured(settings)
    if not llm_ok:
        pytest.skip(f"live retcon test requires real LLM: {why}")

    saves_root = tmp_path / "saves"
    saves_root.mkdir(parents=True, exist_ok=True)

    class _NullImage:
        async def generate(self, *_a, **_kw) -> bytes:
            return b""

    session = Session(
        settings=settings,
        saves_root=saves_root,
        image_client=_NullImage(),
    )
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    try:
        await _drive_until_history(
            registry,
            ctx,
            session,
            setting="A neon-lit night market on the rim of a port city",
            genre="Noir thriller",
            turns=TURNS_BEFORE_RETCON,
        )

        # Identify the player character and their pre-retcon outfit.
        player = next(
            (c for c in session.game.characters.values() if c.is_player),
            None,
        )
        assert player is not None, "playthrough should have created a player character"
        before_outfit = (player.outfit or "").lower()

        retcon_text = (
            f"From this point forward (and retroactively across "
            f"all prior beats), {player.name} is wearing a bright "
            f"red leather jacket. Update the player's outfit "
            f"attribute to reflect the red leather jacket, and "
            f"rewrite any beat that describes what {player.name} "
            f"or 'you' are wearing so it matches."
        )
        await asyncio.wait_for(
            _dispatch(
                registry,
                ctx,
                MessageType.c2s_edit_history_retcon,
                {"instructions": retcon_text},
            ),
            timeout=PER_CALL_BUDGET_S,
        )

        # The player's canon outfit must reflect the red jacket.
        after_player = session.game.characters[player.id]
        after_outfit = (after_player.outfit or "").lower()
        assert "jacket" in after_outfit and "red" in after_outfit, (
            f"player.outfit was not updated to a red leather jacket. "
            f"before={before_outfit!r} after={after_outfit!r}"
        )

    finally:
        for attr in (
            "_char_desc_task",
            "_world_init_task",
            "_preview_bg_task",
            "_preview_guide_task",
            "_pc_portrait_task",
            "_name_options_task",
        ):
            t = getattr(session, attr, None)
            if t is not None and not t.done():
                t.cancel()
        for spec in (getattr(session, "_speculative_tasks", None) or {}).values():
            if not spec.done():
                spec.cancel()
        for asset in getattr(session, "_asset_tasks", None) or []:
            if not asset.done():
                asset.cancel()
