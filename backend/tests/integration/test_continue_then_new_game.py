"""Round-trip regression test: Continue → Menu → New Game.

The reported symptom was that clicking ``New Game`` after
``Continue`` resumed the loaded save instead of starting fresh —
the player completed the interview, clicked Begin, and briefly
saw their old game's main view before the new one swapped in.

The cause: ``new_game_start_handler`` reset ``session.interview``
but left ``session.game`` pointing at the loaded save. The
renderer's auto-route (``main.tsx`` ``loading`` phase) bounces
to the main view as soon as ``game.current_node_id`` is set, so
when the user clicked Begin the loading screen flashed the OLD
save's content while the backend was still running ``world_init``
for the new run.

The fix is to clear ``session.game`` (and emit a state patch
nulling ``/game``) the moment the user clicks New Game. This
test pins both the backend state mutation AND the wire patch.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from lucidium.api.handlers import HandlerContext, build_default_registry
from lucidium.api.messages import (
    C2SNewGameAnswer,  # noqa: F401 -- referenced in step loop
    C2SNewGameConfirm,  # noqa: F401
    C2SNewGameStart,  # noqa: F401
    Envelope,
    InterviewStep,
    MessageType,
)
from lucidium.orchestration.session import Session


class _QueuedLlm:
    """Returns the next queued response regardless of prompt. Falls
    back to a no-op single-beat payload so speculative branches
    don't have to be padded into every fixture."""

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

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def complete(self, *_a, **_kw) -> AsyncIterator[str]:
        next_value = self._responses.pop(0) if self._responses else self._SPEC_FALLBACK

        async def gen() -> AsyncIterator[str]:
            yield next_value

        return gen()


class _NullImage:
    async def generate(self, *_a, **_kw) -> bytes:
        return b""


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


def _world_init_payload(*, game_name: str, character_name: str = "Iris") -> str:
    """Build a minimal valid world-init payload. Two distinct
    invocations of this helper produce two clearly-different
    games — game_name lets a test prove which one it's looking
    at."""
    return json.dumps(
        {
            "game_name": game_name,
            "overall_plot_direction": "Find the missing archivist before the eclipse.",
            "active_plot_threads": [],
            "opening_node": {
                "beats": [
                    _beat(
                        "The harbor wakes slow.",
                        location_id="loc-1",
                        location_prompt="stone harbor at dawn",
                    ),
                ],
                "options": [
                    {"id": "opt-1", "text": "Walk to the archive."},
                ],
            },
            "player_character": {
                "name": character_name,
                "description": "A wry archivist.",
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
    async for message in handler_result:
        out.append(message)
    return out


def _new_game_responses(game_name: str) -> list[str]:
    """Five LLM responses that the new-game flow consumes:
    char-desc options, name options, world-init, plus two
    next-node fallbacks for speculative branches off the
    opening node."""
    return [
        _options_payload([f"char-{i}" for i in range(6)]),
        _options_payload([f"name-{i}" for i in range(8)]),
        _world_init_payload(game_name=game_name),
    ]


async def _walk_new_game(
    registry,
    ctx: HandlerContext,
    *,
    setting: str,
    genre: str,
    visual_style: str,
    character: str,
    name: str,
) -> None:
    """Drive the new-game flow start → answer×5 → confirm."""
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_start, payload={}),
            ctx,
        )
    )
    for step, answer in [
        (InterviewStep.setting, setting),
        (InterviewStep.genre, genre),
        (InterviewStep.visual_style, visual_style),
        (InterviewStep.character_description, character),
        (InterviewStep.name, name),
    ]:
        await _drain(
            registry.dispatch(
                Envelope(
                    type=MessageType.c2s_new_game_answer,
                    payload={
                        "step": step.value,
                        "answer": answer,
                        "is_free_text": False,
                    },
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


@pytest.mark.asyncio
async def test_new_game_after_continue_replaces_loaded_save(
    tmp_app_data: Path,
) -> None:
    """The full Continue → Menu → New Game round-trip, end-to-end.

    Walks two new-game flows back to back: the first creates a save
    with ``game_name="First Game"`` (then ``saves_continue_handler``
    loads it, simulating Continue from the menu), the second emits
    a fresh ``c2s/new_game/start`` and runs through to confirm with
    ``game_name="Second Game"``. Pins:

      * ``c2s/new_game/start`` clears ``session.game`` (so the
        renderer's loading screen doesn't have a stale node id to
        auto-route on).
      * The patch emitted on start nulls ``/game`` on the wire.
      * After confirm, ``session.game.world.game_name`` is the
        SECOND game — the loaded save was actually replaced, not
        re-emitted.
    """
    queued = _QueuedLlm(
        responses=(_new_game_responses("First Game") + _new_game_responses("Second Game"))
    )
    session = Session(llm_client=queued, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    # 1. Walk the FIRST new-game flow end-to-end. This commits a
    # save under the test's tmp_app_data — the next Continue picks
    # it up.
    await _walk_new_game(
        registry,
        ctx,
        setting="stone harbor at dawn",
        genre="occult mystery",
        visual_style="ink-wash painting",
        character="wry archivist",
        name="Iris",
    )
    assert session.game is not None
    assert session.game.world.game_name == "First Game"

    # 2. Simulate the player going to the menu and clicking
    # Continue. Continue installs the most-recent save into
    # session.game.
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_saves_continue, payload={}),
            ctx,
        )
    )
    assert session.game is not None
    assert session.game.world.game_name == "First Game"
    loaded_node_id = session.game.current_node_id

    # 3. Player clicks New Game. ``c2s/new_game/start`` MUST clear
    # session.game — otherwise the renderer's ``loading`` phase
    # auto-routes to the main view on the loaded save's
    # current_node_id while the new world_init is still running.
    out = await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_new_game_start, payload={}),
            ctx,
        )
    )
    assert session.game is None, (
        "new_game/start must clear session.game so the renderer "
        "doesn't bounce back into the loaded save's main view."
    )

    # 3b. The wire-level patch the renderer receives must also
    # null /game so the store's ``game`` becomes ``null``.
    game_null_ops = []
    for msg_type, payload in out:
        if msg_type != MessageType.s2c_state_patch:
            continue
        for op in payload.ops:
            if op.path == "/game" and op.value is None:
                game_null_ops.append(op)
    assert game_null_ops, (
        "expected a state-patch op replacing /game with null on "
        "new_game/start; renderer's auto-route reads from store"
    )

    # 4. Walk the SECOND interview to completion with a different
    # game_name. The session must end up holding THAT game, not
    # the loaded one.
    for step, answer in [
        (InterviewStep.setting, "neon city after midnight"),
        (InterviewStep.genre, "noir thriller"),
        (InterviewStep.visual_style, "cel-shaded comic"),
        (InterviewStep.character_description, "former cop"),
        (InterviewStep.name, "Vance"),
    ]:
        await _drain(
            registry.dispatch(
                Envelope(
                    type=MessageType.c2s_new_game_answer,
                    payload={
                        "step": step.value,
                        "answer": answer,
                        "is_free_text": False,
                    },
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

    assert session.game is not None
    assert session.game.world.game_name == "Second Game", (
        f"expected Second Game after the new-game round-trip; got "
        f"{session.game.world.game_name!r} — the loaded save was "
        f"never replaced."
    )
    # Sanity: the new game's opening node id differs from the
    # loaded save's, so any node-id snapshot the renderer was
    # using also gets a fresh value.
    assert session.game.current_node_id != loaded_node_id
