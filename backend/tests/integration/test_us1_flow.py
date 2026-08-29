"""End-to-end test of the US1 (MVP) flow against a fake LLM.

Drives the handler registry directly (no WebSocket) with a queue-driven
fake ``LlmClient`` whose responses are JSON the handlers expect to
parse. This covers:

  * ``c2s/new_game/start`` -> setting options surfaced.
  * five ``c2s/new_game/answer`` rounds (setting, genre, visual_style,
    character_description, name) -> next-step options arrive.
  * ``c2s/new_game/confirm`` -> opening node committed and a save written.
  * ``c2s/play/advance`` -> next dialog node text streamed.
  * ``c2s/play/free_text`` -> speculative-only descendants invalidated;
    new node generated under the player's free-text input.
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
from lucidium.domain.dialog import DialogNodeState
from lucidium.orchestration.session import Session


class QueuedLlm:
    """LlmClient that returns the next queued response, regardless of prompt.

    When the queue is empty the client falls back to a no-op single-
    beat response so tests don't have to pad fixtures for every
    speculative branch the engine fires off the back of each commit.
    """

    _SPEC_FALLBACK = json.dumps(
        {
            "beats": [
                {
                    "text": "(speculative).",
                    "speaker_id": None,
                    "entering_character_ids": [],
                    "leaving_character_ids": [],
                    "new_characters": [],
                    "location_id": None,
                    "location_prompt": None,
                    "character_changes": [],
                }
            ],
            "options": [],
        }
    )

    def __init__(self, responses: list[str | list[str]]) -> None:
        self._responses = list(responses)

    async def complete(self, *_a, **_kw) -> AsyncIterator[str]:
        next_value = self._responses.pop(0) if self._responses else self._SPEC_FALLBACK

        async def gen() -> AsyncIterator[str]:
            if isinstance(next_value, list):
                for chunk in next_value:
                    yield chunk
            else:
                yield next_value

        return gen()


def _options_payload(items: list[str]) -> str:
    return json.dumps({"options": items})


def _beat(text: str, **kwargs: object) -> dict:
    return {
        "text": text,
        "speaker_id": kwargs.get("speaker_id"),
        "entering_character_ids": kwargs.get("entering_character_ids", []),
        "leaving_character_ids": kwargs.get("leaving_character_ids", []),
        "location_id": kwargs.get("location_id"),
        "location_prompt": kwargs.get("location_prompt"),
        "character_changes": kwargs.get("character_changes", []),
    }


def _world_init_payload() -> str:
    return json.dumps(
        {
            "game_name": "Embers of the Veil",
            "overall_plot_direction": "Find the missing archivist before the eclipse.",
            "active_plot_threads": [
                {
                    "id": "thread-1",
                    "title": "Missing archivist",
                    "summary": "Mira vanished before dawn; her notes are still warm.",
                    "last_referenced_node_id": None,
                }
            ],
            "opening_node": {
                "beats": [
                    _beat(
                        "The harbor wakes slow.",
                        location_id="loc-1",
                        location_prompt="stone harbor at dawn, sea-fog, low light",
                    ),
                    _beat("Iris turns up her collar against the sea-fog."),
                ],
                "options": [
                    {"id": "opt-1", "text": "Walk to the archive."},
                    {"id": "opt-2", "text": "Find Hale at the inn."},
                ],
            },
            "player_character": {
                "name": "Iris",
                "description": "A wry archivist who keeps her opinions in the margins.",
                "gender": "female",
                "age": 28,
                "ethnicity": "local",
                "skin": "pale",
                "hair_color": "auburn",
                "hairstyle": "braid",
                "eye_color": "grey",
                "build": "slight",
                "bust": "moderate",
                "outfit": "wool coat over dark linen",
                "pose": "standing",
                "expression": "alert",
            },
        }
    )


def _next_node_payload(text: str, *, options: list[str] | None = None) -> str:
    # Single-beat response (the simplest valid payload).
    return json.dumps(
        {
            "beats": [_beat(text)],
            "options": [{"id": f"opt-{i}", "text": o} for i, o in enumerate(options or [])],
        }
    )


async def _drain(handler_result):
    out = []
    async for message in handler_result:
        out.append(message)
    return out


@pytest.mark.asyncio
async def test_full_us1_round_trip(tmp_app_data: Path) -> None:
    queued = QueuedLlm(
        responses=[
            # Setting, Visual Style, and Genre are all hard-coded —
            # no LLM round-trip. The first LLM call fires when the
            # player answers Setting and the char-desc prefetch
            # spawns; name options follow at the character_description
            # answer; world_init at the name answer.
            _options_payload([f"char-{i}" for i in range(6)]),
            _options_payload([f"name-{i}" for i in range(8)]),
            _world_init_payload(),
            _next_node_payload(
                "She turns the brass key in the archive door.",
                options=["Light the lamp.", "Listen to the silence."],
            ),
            _next_node_payload("Iris kneels and pulls the loose floorboard up by its corner."),
        ]
    )
    session = Session(llm_client=queued, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    # 1) c2s/new_game/start
    out = await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_start, payload={}),
            ctx,
        )
    )
    assert any(t == MessageType.s2c_state_patch for t, _ in out)

    # 2-6) five answers
    for step, answer in [
        (InterviewStep.setting, "stone harbor at dawn"),
        (InterviewStep.genre, "occult mystery"),
        (InterviewStep.visual_style, "ink-wash painting"),
        (InterviewStep.character_description, "wry archivist"),
        (InterviewStep.name, "Iris"),
    ]:
        out = await _drain(
            registry.dispatch(
                Envelope(
                    type=MessageType.c2s_new_game_answer,
                    payload={"step": step.value, "answer": answer, "is_free_text": False},
                ),
                ctx,
            )
        )
        assert out, f"no messages for step {step}"

    # 7) confirm
    out = await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_confirm, payload={"overrides": {}}),
            ctx,
        )
    )
    assert any(t == MessageType.s2c_state_full for t, _ in out)
    assert session.game is not None
    assert session.game.world.game_name == "Embers of the Veil"
    assert session.game.current_node_id is not None

    # 8a) The opening node is now a chain — current is the FIRST beat
    # (text "The harbor wakes slow."). The player presses Continue
    # (option_id=None) to walk to the second beat. This walk should
    # NOT call the LLM — the beat is already in the tree.
    assert session.game is not None
    first_beat_id = session.game.current_node_id
    out = await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_play_advance, payload={"option_id": None}),
            ctx,
        )
    )
    assert session.game.current_node_id != first_beat_id, (
        "Continue must advance to the next pre-generated beat"
    )

    # 8b) Now we're on the LAST beat which carries the options. Click
    # opt-1 — this generates a NEW chain (single-beat in this fixture).
    out = await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_play_advance, payload={"option_id": "opt-1"}),
            ctx,
        )
    )
    text_messages = [p for t, p in out if t == MessageType.s2c_text_streaming]
    assert text_messages, "expected streaming text"
    completes = [p for t, p in out if t == MessageType.s2c_text_complete]
    assert completes, "expected text complete"
    assert session.game is not None
    assert session.game.current_node_id != session.game.dialog_tree.root_id

    # Now manually drop a speculative future child to verify free-text
    # invalidation marks it as such.
    from lucidium.domain.dialog import DialogNode

    speculative = DialogNode(
        parent_id=session.game.current_node_id,
        premise_hash="dummy",
        state=DialogNodeState.speculative,
    )
    nodes = dict(session.game.dialog_tree.nodes)
    nodes[speculative.id] = speculative
    new_tree = session.game.dialog_tree.model_copy(update={"nodes": nodes})
    session.game = session.game.model_copy(update={"dialog_tree": new_tree})

    # 9) free_text
    out = await _drain(
        registry.dispatch(
            Envelope(
                type=MessageType.c2s_play_free_text,
                payload={"text": "Iris pries up the floorboard."},
            ),
            ctx,
        )
    )
    assert any(t == MessageType.s2c_state_full for t, _ in out)
    assert session.game.dialog_tree.nodes[speculative.id].state == DialogNodeState.invalidated


class _NullImage:
    async def generate(self, *_a, **_kw) -> bytes:
        return b""
