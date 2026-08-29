"""LLM calls are kicked off in parallel with player decision time.

Step ordering: Setting → Visual Style → Genre → Character Description
→ Name. Two of the five steps (Visual Style and Genre) are entirely
hard-coded — no LLM round-trip ever — so the player can NEVER stall
on a slow LLM at those transitions. The remaining three steps
(Setting's char-desc prefetch, Character's name prefetch, Name's
world_init prefetch) all run speculatively in the background while
the player reads the next question.

These tests pin the speculative tasks to the session's private slots
(``_char_desc_task``, ``_name_options_task``, ``_world_init_task``)
AND assert that no LLM call ever fires for Visual Style or Genre
(if either regressed to an LLM fetch, the player would block on a
network round-trip during onboarding).
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


class CountingLlm:
    """LlmClient that records every prompt it sees, returning queued JSON."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, prompt, *_a, **_kw) -> AsyncIterator[str]:
        self.calls.append(prompt)
        if not self._responses:
            raise AssertionError(
                f"LLM was called more times than fixtures provided ({len(self.calls)} so far)"
            )
        next_value = self._responses.pop(0)

        async def gen() -> AsyncIterator[str]:
            yield next_value

        return gen()


class _NullImage:
    async def generate(self, *_a, **_kw) -> bytes:
        return b""


def _options(items: list[str]) -> str:
    return json.dumps({"options": items})


def _world_init_payload() -> str:
    return json.dumps(
        {
            "game_name": "The Salt Lantern",
            "overall_plot_direction": "Find the missing keeper.",
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
                "options": [{"id": "opt-1", "text": "Walk."}],
            },
            "player_character": {
                "name": "Iris",
                "description": "a wry archivist",
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


async def _drain(handler_result):
    out = []
    async for msg in handler_result:
        out.append(msg)
    return out


async def _answer(ctx: HandlerContext, registry, step: InterviewStep, value: str):
    await _drain(
        registry.dispatch(
            Envelope(
                type=MessageType.c2s_new_game_answer,
                payload={"step": step.value, "answer": value, "is_free_text": False},
            ),
            ctx,
        )
    )


@pytest.mark.asyncio
async def test_start_does_not_call_llm(tmp_app_data: Path) -> None:
    """``c2s/new_game/start`` must not trigger ANY LLM call. The
    Setting step's options are hard-coded; Visual Style and Genre
    are too. The first LLM round-trip in the interview happens at
    Setting-answer time (char-desc prefetch), nothing earlier.
    """

    llm = CountingLlm(responses=[])
    session = Session(llm_client=llm, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _drain(registry.dispatch(Envelope(type=MessageType.c2s_new_game_start, payload={}), ctx))
    # Yield generously so any background task that *would* fire an
    # LLM call gets a chance to do so.
    for _ in range(5):
        await asyncio.sleep(0)

    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_setting_answer_serves_hardcoded_visual_style(tmp_app_data: Path) -> None:
    """The Visual Style step is HARD-CODED — answering Setting must
    NOT trigger an LLM call to fetch visual styles. The handler's
    response carries the hard-coded list inline (no follow-up
    state/patch needed). In parallel, the character-description
    prefetch fires (depends on Setting only) so it's ready by the
    time Genre is answered.

    Regression guard: the previous design fetched visual styles from
    the LLM, which made Step-1→Step-2 transitions block on the
    network. If anyone re-introduces an LLM call for visual styles,
    this assertion fires.
    """

    llm = CountingLlm(
        responses=[
            _options([f"char-{i}" for i in range(6)]),
        ]
    )
    session = Session(llm_client=llm, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _drain(registry.dispatch(Envelope(type=MessageType.c2s_new_game_start, payload={}), ctx))
    assert len(llm.calls) == 0

    # Capture the raw outbound state/patch so we can check that
    # visual_style_options is populated inline — no follow-up
    # required.
    out = await _drain(
        registry.dispatch(
            Envelope(
                type=MessageType.c2s_new_game_answer,
                payload={"step": "setting", "answer": "stone harbor", "is_free_text": False},
            ),
            ctx,
        )
    )
    # One state/patch should land in this turn (no fire-and-emit
    # follow-up because options are inline).
    assert len(out) == 1
    msg_type, msg = out[0]
    assert msg_type is MessageType.s2c_state_patch
    options_op = next(op for op in msg.ops if op.path == "/interview/visual_style_options")
    assert options_op.value, "visual_style_options must be populated inline"
    # The handler samples one option from each of the four flavour
    # buckets (hyperrealistic / cartoony / artsy / oddball), so the
    # surfaced list is always exactly four entries.
    assert len(options_op.value) == 4
    # Each surfaced option must be drawn from one of the four buckets.
    from lucidium.orchestration.prompts.interview import VISUAL_STYLE_BUCKETS

    all_styles = {s for bucket in VISUAL_STYLE_BUCKETS.values() for s in bucket}
    for surfaced in options_op.value:
        assert surfaced in all_styles, f"surfaced style {surfaced!r} not in any bucket"

    # The setting handler kicked off the char_desc prefetch.
    assert getattr(session, "_char_desc_task", None) is not None
    # Yield so the char_desc prefetch completes.
    for _ in range(5):
        await asyncio.sleep(0)
    # Exactly ONE LLM call in total — char_desc. Visual style was
    # served from the hard-coded list, not the LLM.
    assert len(llm.calls) == 1, (
        f"expected one LLM call (char_desc), got {len(llm.calls)} — "
        "did visual_style regress to an LLM fetch?"
    )


@pytest.mark.asyncio
async def test_visual_style_answer_does_not_call_llm(tmp_app_data: Path) -> None:
    """Answering Visual Style must NOT trigger any LLM call. The
    next step (Genre) is also hard-coded, and the only side effect
    of the visual_style answer is kicking off a ComfyUI background
    render task — never an LLM call."""

    llm = CountingLlm(
        responses=[
            _options([f"char-{i}" for i in range(6)]),
        ]
    )
    session = Session(llm_client=llm, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _drain(registry.dispatch(Envelope(type=MessageType.c2s_new_game_start, payload={}), ctx))
    await _answer(ctx, registry, InterviewStep.setting, "stone harbor at dawn")
    # Yield so the speculative char_desc prefetch lands; that one
    # LLM call is expected.
    for _ in range(5):
        await asyncio.sleep(0)
    calls_after_setting = len(llm.calls)
    assert calls_after_setting == 1

    await _answer(ctx, registry, InterviewStep.visual_style, "ink wash, monochrome")
    # Yield so any (unwanted) background LLM call would have a chance
    # to fire and get counted.
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(llm.calls) == calls_after_setting, "visual_style answer must not trigger an LLM call"


@pytest.mark.asyncio
async def test_name_answer_kicks_off_world_init_prefetch(tmp_app_data: Path) -> None:
    """The longest LLM call (world_init) starts the moment Name is
    answered, while the player is reading the Review screen.
    """

    llm = CountingLlm(
        responses=[
            _options([f"char-{i}" for i in range(6)]),
            _options([f"name-{i}" for i in range(8)]),
            _world_init_payload(),
        ]
    )
    session = Session(llm_client=llm, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _drain(registry.dispatch(Envelope(type=MessageType.c2s_new_game_start, payload={}), ctx))
    # Reordered flow: setting → visual_style → genre → char_desc → name.
    # Visual Style + Genre are hard-coded (no LLM round-trip).
    await _answer(ctx, registry, InterviewStep.setting, "stone harbor at dawn")
    await _answer(ctx, registry, InterviewStep.visual_style, "ink wash, monochrome")
    await _answer(ctx, registry, InterviewStep.genre, "occult mystery")
    await _answer(ctx, registry, InterviewStep.character_description, "wry archivist")
    # Yield so the async name_options prefetch (kicked off at
    # character_description) actually completes.
    for _ in range(5):
        await asyncio.sleep(0)
    calls_before_name = len(llm.calls)

    await _answer(ctx, registry, InterviewStep.name, "Iris")
    # The world_init task is now in flight on the session.
    assert getattr(session, "_world_init_task", None) is not None

    # Yield so the speculative task completes.
    for _ in range(5):
        await asyncio.sleep(0)

    # The world_init LLM call has happened during "review screen time."
    assert len(llm.calls) == calls_before_name + 1


@pytest.mark.asyncio
async def test_confirm_consumes_world_init_prefetch(tmp_app_data: Path) -> None:
    """``c2s/new_game/confirm`` does NOT trigger a fresh world_init LLM
    call — it awaits the prefetch from the Name handler.
    """

    llm = CountingLlm(
        responses=[
            _options([f"char-{i}" for i in range(6)]),
            _options([f"name-{i}" for i in range(8)]),
            _world_init_payload(),
        ]
    )
    session = Session(llm_client=llm, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _drain(registry.dispatch(Envelope(type=MessageType.c2s_new_game_start, payload={}), ctx))
    for step, answer in [
        (InterviewStep.setting, "stone harbor at dawn"),
        (InterviewStep.visual_style, "ink wash, monochrome"),
        (InterviewStep.genre, "occult mystery"),
        (InterviewStep.character_description, "wry archivist"),
        (InterviewStep.name, "Iris"),
    ]:
        await _answer(ctx, registry, step, answer)
        # The character_description handler fires off an async
        # name_options task; yield so it lands before the next step's
        # prefetch is awaited.
        for _ in range(3):
            await asyncio.sleep(0)

    # Drain the speculative world_init task.
    for _ in range(5):
        await asyncio.sleep(0)

    calls_before_confirm = len(llm.calls)
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )
    # Confirm did NOT trigger a fresh world_init call.
    assert len(llm.calls) == calls_before_confirm, (
        "world_init must come from the prefetch, not a fresh LLM call"
    )
    # Slot emptied.
    assert getattr(session, "_world_init_task", None) is None
