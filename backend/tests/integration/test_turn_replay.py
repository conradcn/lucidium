"""Replayability — committed dialog nodes carry the prompt parameters
needed to reproduce them. Constitution "Determinism" constraint.
"""

from __future__ import annotations

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


class QueuedLlm:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def complete(self, *_a, **_kw) -> AsyncIterator[str]:
        v = self._responses.pop(0)

        async def gen() -> AsyncIterator[str]:
            yield v

        return gen()


def _opt(items: list[str]) -> str:
    return json.dumps({"options": items})


def _beat(text: str) -> dict:
    return {
        "text": text,
        "speaker_id": None,
        "entering_character_ids": [],
        "leaving_character_ids": [],
        "location_id": None,
        "location_prompt": None,
        "character_changes": [],
    }


def _world() -> str:
    return json.dumps(
        {
            "game_name": "G",
            "overall_plot_direction": "do the thing",
            "active_plot_threads": [],
            "opening_node": {
                "beats": [_beat("open")],
                "options": [],
            },
            "player_character": {
                "name": "P",
                "description": "x",
                "gender": "x",
                "age": 30,
                "ethnicity": "x",
                "skin": "x",
                "hair_color": "x",
                "hairstyle": "x",
                "eye_color": "x",
                "build": "x",
                "bust": "x",
                "outfit": "x",
                "pose": "x",
                "expression": "x",
            },
        }
    )


def _next_node(text: str) -> str:
    return json.dumps({"beats": [_beat(text)], "options": []})


@pytest.mark.asyncio
async def test_committed_node_carries_generation_metadata(tmp_app_data: Path) -> None:
    # Setting, Visual Style, and Genre are all hard-coded (no LLM);
    # the character-description prefetch is the first LLM round-trip,
    # spawned when Setting is answered.
    queued = QueuedLlm(
        [
            _opt(["c"] * 6),
            _opt(["n"] * 8),
            _world(),
            _next_node("She walks toward the harbor."),
        ]
    )
    session = Session(llm_client=queued, image_client=None)
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    async def drain(envelope: Envelope) -> None:
        async for _msg in registry.dispatch(envelope, ctx):
            pass

    await drain(Envelope(type=MessageType.c2s_new_game_start, payload={}))
    for step in [
        InterviewStep.setting,
        InterviewStep.visual_style,
        InterviewStep.genre,
        InterviewStep.character_description,
        InterviewStep.name,
    ]:
        await drain(
            Envelope(
                type=MessageType.c2s_new_game_answer,
                payload={"step": step.value, "answer": "x", "is_free_text": False},
            )
        )
    await drain(Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}))
    await drain(Envelope(type=MessageType.c2s_play_advance, payload={"option_id": None}))

    assert session.game is not None
    current = session.game.current_node_id
    assert current is not None
    node = session.game.dialog_tree.nodes[current]
    metadata = node.generation_metadata
    # Replayability: model id is recorded so a reviewer knows which
    # provider/model produced the turn; temperature is captured so the
    # turn can be replayed deterministically with the same settings.
    assert metadata.model is not None
    assert "temperature" in metadata.seed_parameters
