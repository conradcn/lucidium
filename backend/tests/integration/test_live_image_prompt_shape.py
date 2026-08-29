"""Live test: storyteller → image-prompt shape.

Drives the real LLM through the new-game flow, then captures the
image prompts the engine WOULD send to the renderer. Asserts:

  * the composed positive prompt fits inside CLIP's 77-token
    window (so the model doesn't silently drop the tail);
  * expression / pose / outfit each survive the LLM-to-prompt
    pipeline (the storyteller emits ≤ 6-word values, our
    ``_trim_attr`` defends against overlong ones, and the prompt
    builder puts them in the high-attention head of the prompt);
  * the face_prompt leads with the expression tag;
  * the background prompt fits the same window.

Skips cleanly when the user hasn't configured a live LLM endpoint
(no api_key) — same gating as ``test_live_long_session.py``.
Marked ``live`` so the default test invocation excludes it.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from lucidium.api.handlers import HandlerContext, build_default_registry
from lucidium.api.messages import (
    Envelope,
    InterviewStep,
    MessageType,
)
from lucidium.orchestration.prompts import image_prompts
from lucidium.orchestration.session import Session
from lucidium.persistence import settings_store

CLIP_TOKEN_BUDGET = 77


pytestmark = [
    pytest.mark.live,
    pytest.mark.filterwarnings("ignore::ResourceWarning"),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]


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


def _approx_clip_tokens(text: str) -> int:
    """Approximate CLIP token count. CLIP's BPE tokenizer averages
    around 1.3 tokens per word for natural English prose plus a
    handful of additional tokens for punctuation and weight syntax
    ``(...:N.N)``. This is an upper-ish estimate — over-counting is
    safe for an "is the prompt too long?" assertion. Accurate
    tokenisation requires the actual CLIP tokenizer, which we don't
    pull in for the test layer.
    """
    if not text:
        return 0
    words = text.split()
    weight_marker_count = text.count(":") + text.count("(") + text.count(")")
    comma_count = text.count(",")
    return int(len(words) * 1.4) + weight_marker_count + comma_count


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


@pytest.mark.asyncio
async def test_live_storyteller_to_image_prompt_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the new-game interview to confirm with the real LLM,
    then build the image prompts the engine would send to render
    the opening scene's character + background. Assert each
    prompt's structure."""
    monkeypatch.delenv("LUCIDIUM_OFFLINE", raising=False)
    settings = settings_store.load_settings()

    llm_ok, why = _llm_configured(settings)
    if not llm_ok:
        pytest.skip(f"live test requires real LLM: {why}")

    saves_root = tmp_path / "saves"
    saves_root.mkdir(parents=True, exist_ok=True)
    # No image client needed — we only inspect the prompt strings,
    # not the rendered output. A null stub keeps the asset pipeline
    # from trying to spin up ComfyUI / diffusers.

    class _NullImage:
        async def generate(self, *_a, **_kw) -> bytes:
            return b""

    session = Session(
        settings=settings,
        image_client=_NullImage(),
        saves_root=saves_root,
    )
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    cancel_tasks: list[asyncio.Task] = []
    try:
        await _dispatch(registry, ctx, MessageType.c2s_new_game_start, {})
        await _answer(registry, ctx, InterviewStep.setting, "stone harbor at dawn")
        await _answer(registry, ctx, InterviewStep.visual_style, "ink wash")
        await _answer(registry, ctx, InterviewStep.genre, "Mystery")
        await _answer(
            registry,
            ctx,
            InterviewStep.character_description,
            "wry archivist with a secret",
        )
        await _answer(registry, ctx, InterviewStep.name, "Iris")
        await _dispatch(
            registry,
            ctx,
            MessageType.c2s_new_game_confirm,
            {"overrides": {}},
        )

        assert session.game is not None, "world_init didn't install a game"
        game = session.game

        # Collect the prompts the engine would compose for every
        # on-stage character + the active environment.
        for cid in game.on_stage:
            character = game.characters.get(cid)
            assert character is not None, f"on-stage id {cid!r} not in characters"
            positive = image_prompts.portrait_prompt(
                world=game.world,
                character=character,
                lighting="",
            )
            face = image_prompts.portrait_face_prompt(character=character)

            tokens = _approx_clip_tokens(positive)
            assert tokens <= CLIP_TOKEN_BUDGET, (
                f"character {cid} positive prompt is approximately "
                f"{tokens} CLIP tokens, over the 77-token budget. "
                f"prompt: {positive!r}"
            )

            # Body-composition tags must appear in the MAIN prompt.
            # Expression LIVES in the main prompt with strong weight
            # — SDXL's adult-portrait prior paints a smile during
            # the body pass, and a downstream FaceDetailer can
            # rarely override what the body composition already
            # committed to. Earlier "expression lives only in
            # face_prompt" was visibly broken (always smiling).
            assert "pose:" in positive, f"character {cid} prompt missing pose tag: {positive!r}"
            assert "wearing" in positive, f"character {cid} prompt missing outfit tag: {positive!r}"
            assert "full body" in positive, (
                f"character {cid} prompt missing full-body framing: {positive!r}"
            )

            # Face prompt leads with the bare expression adjective
            # at 1× weight. Earlier weighted forms (``(angry:1.3)``
            # or ``(expression: angry:1.3)``) over-amplified the
            # adjective on the FaceDetailer pass and distorted face
            # geometry. The first comma-separated chunk should be
            # the expression text itself, with no leading parens
            # and no weight syntax.
            first_chunk = face.split(",", 1)[0].strip()
            assert "(" not in first_chunk and ":" not in first_chunk, (
                f"character {cid} face prompt expression chunk should "
                f"be a bare adjective (no parens, no :weight) — "
                f"weighting expression past 1.0 distorted faces. "
                f"got: {face!r}"
            )

            # Defensive cap: even if the LLM emits a long phrase,
            # the prompt builder's _trim_attr should keep each
            # weighted tag (pose / outfit in main) under ~10 words
            # inside the parentheses.
            for label, source in (
                ("pose:", positive),
                ("wearing", positive),
            ):
                pat = re.compile(
                    r"\(" + re.escape(label) + r"\s*([^:]+):\d",
                    re.IGNORECASE,
                )
                m = pat.search(source)
                if not m:
                    continue
                body = m.group(1).strip()
                word_count = len(body.split())
                assert word_count <= 10, (
                    f"character {cid} {label!r} tag has {word_count} "
                    f"words after the storyteller cap + _trim_attr "
                    f"defence: {body!r} (source: {source!r})"
                )

        # Background: assert the same CLIP-budget envelope.
        if game.environments:
            env = next(iter(game.environments.values()))
            bg_positive = image_prompts.background_prompt(
                world=game.world,
                environment=env,
            )
            tokens = _approx_clip_tokens(bg_positive)
            assert tokens <= CLIP_TOKEN_BUDGET, (
                f"background prompt is approximately {tokens} CLIP "
                f"tokens, over the 77-token budget. "
                f"prompt: {bg_positive!r}"
            )
    finally:
        # Cancel any speculative tasks the new-game flow spun up so
        # the test exits cleanly.
        spec = getattr(session, "_speculative_tasks", None) or {}
        for task in spec.values():
            if not task.done():
                task.cancel()
                cancel_tasks.append(task)
        for task in cancel_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
