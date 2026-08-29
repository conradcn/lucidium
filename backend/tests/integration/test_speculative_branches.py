"""Speculative tree generation: after committing a turn, the engine
spawns LLM calls for each option of the current node so that when the
player picks one, the next chain is already in the dialog tree and
``play_advance`` walks it without a fresh LLM round-trip.

Tests pin:
  * The presence of speculative tasks on the session.
  * That tasks fire AFTER each commit.
  * That play_advance for an option whose speculative chain finished
    consumes ZERO additional LLM calls (the foreground walk path).
  * That play_advance for an option whose speculative chain is still
    in flight awaits it instead of starting a fresh LLM call.
  * That free_text invalidation cancels speculative descendants.
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
    """Records every prompt; returns the next queued response or a
    safe single-beat fallback when out of fixtures.
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

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, prompt, *_a, **_kw) -> AsyncIterator[str]:
        self.calls.append(prompt)
        nv = self._responses.pop(0) if self._responses else self._SPEC_FALLBACK

        async def gen() -> AsyncIterator[str]:
            yield nv

        return gen()


class _NullImage:
    async def generate(self, *_a, **_kw) -> bytes:
        return b""


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
                    {"id": "left", "text": "Walk to the inn."},
                    {"id": "right", "text": "Climb the cliff."},
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


def _branch_payload(text: str) -> str:
    return json.dumps(
        {
            "beats": [
                {
                    "text": text,
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


async def _drain(handler_result):
    out = []
    async for msg in handler_result:
        out.append(msg)
    return out


async def _run_to_opening(ctx: HandlerContext, registry, llm: CountingLlm):
    await _drain(registry.dispatch(Envelope(type=MessageType.c2s_new_game_start, payload={}), ctx))
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


@pytest.mark.asyncio
async def test_confirm_spawns_speculative_branches_for_each_option(
    tmp_app_data: Path,
) -> None:
    """The opening node has two options. After confirm, two speculative
    tasks are tracked on the session — one per option."""

    llm = CountingLlm(
        responses=[
            _opt_list([f"c{i}" for i in range(6)]),  # character_description
            _opt_list([f"n{i}" for i in range(8)]),  # name
            _world_init_with_two_options(),
            _branch_payload("Iris steps into the inn."),  # speculation: left
            _branch_payload("Iris scrambles up the cliff."),  # speculation: right
        ]
    )
    session = Session(llm_client=llm, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _run_to_opening(ctx, registry, llm)

    tasks = getattr(session, "_speculative_tasks", None)
    assert tasks is not None
    # Two options → two speculative tasks.
    assert len(tasks) == 2

    # Wait for both speculative tasks to complete. Real sleeps, not
    # ``sleep(0)``: each branch ends in ``session.commit()``, whose disk
    # write runs on a worker thread, and a zero-delay yield never lets
    # the executor finish.
    opening_id = session.game.current_node_id

    def _children() -> list:
        return [n for n in session.game.dialog_tree.nodes.values() if n.parent_id == opening_id]

    for _ in range(100):
        if len(_children()) >= 2:
            break
        await asyncio.sleep(0.02)

    # Both branches now exist as committed children of the opening node.
    children = _children()
    assert len(children) == 2
    chosen_option_ids = sorted(c.chosen_option_id for c in children)
    assert chosen_option_ids == ["left", "right"]


@pytest.mark.asyncio
async def test_speculative_beat_is_pushed_to_renderer(
    tmp_app_data: Path,
) -> None:
    """A speculative beat installed into the tree must also be emitted
    to the renderer as a state/patch ``add`` for its node.

    Without this emit the renderer's dialog_tree mirror never learns the
    next beat exists until its own next Continue walks to it; until then
    ``hasContinueChild`` in the InteractionPanel is false and the
    Continue glyph keeps spinning even though a walkable next beat is
    already sitting in the backend tree — the reported "spinner persists
    even when there is a next line to enter" bug.
    """

    emitted: list[tuple[MessageType, object]] = []

    llm = CountingLlm(
        responses=[
            _opt_list([f"c{i}" for i in range(6)]),
            _opt_list([f"n{i}" for i in range(8)]),
            _world_init_with_two_options(),
            _branch_payload("Iris steps into the inn."),
            _branch_payload("Iris scrambles up the cliff."),
        ]
    )
    session = Session(
        llm_client=llm,
        image_client=_NullImage(),
        emit=lambda mt, payload: emitted.append((mt, payload)),
    )
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _run_to_opening(ctx, registry, llm)
    # Let both speculative branches finish installing.
    for _ in range(10):
        await asyncio.sleep(0)

    opening_id = session.game.current_node_id
    child_ids = {n.id for n in session.game.dialog_tree.nodes.values() if n.parent_id == opening_id}
    assert child_ids, "expected speculative children in the tree"

    # Every speculative child node must have been pushed to the renderer
    # via a state/patch add op at its dialog_tree path.
    added_node_paths = {
        op.path
        for mt, payload in emitted
        if mt == MessageType.s2c_state_patch
        for op in payload.ops
        if op.op == "add" and op.path.startswith("/dialog_tree/nodes/")
    }
    for child_id in child_ids:
        assert f"/dialog_tree/nodes/{child_id}" in added_node_paths, (
            f"speculative child {child_id} was installed into the tree but "
            f"never emitted to the renderer — the Continue glyph would keep "
            f"spinning despite a walkable next beat. Emitted add paths: "
            f"{sorted(added_node_paths)}"
        )


@pytest.mark.asyncio
async def test_advance_into_speculated_branch_is_zero_extra_llm_calls(
    tmp_app_data: Path,
) -> None:
    """When the player picks an option whose speculative chain has
    completed, ``play_advance`` walks it WITHOUT a fresh LLM call."""

    llm = CountingLlm(
        responses=[
            _opt_list([f"c{i}" for i in range(6)]),
            _opt_list([f"n{i}" for i in range(8)]),
            _world_init_with_two_options(),
            _branch_payload("Iris steps into the inn."),
            _branch_payload("Iris scrambles up the cliff."),
        ]
    )
    session = Session(llm_client=llm, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _run_to_opening(ctx, registry, llm)
    # Drain speculative work.
    for _ in range(10):
        await asyncio.sleep(0)
    calls_before_advance = len(llm.calls)

    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_play_advance, payload={"option_id": "left"}),
            ctx,
        )
    )

    # Each commit fires speculation for the new node's options. The
    # walked beat has options=[] (single-beat speculative chain), so
    # NO new speculation. The total LLM-call delta for the player's
    # click is therefore zero — they walked a pre-generated chain.
    assert len(llm.calls) == calls_before_advance, (
        f"play_advance should have walked the speculative chain without an LLM call, "
        f"but {len(llm.calls) - calls_before_advance} extra calls happened"
    )

    # And we landed on the speculative beat's text.
    current = session.game.dialog_tree.nodes[session.game.current_node_id]
    assert current.text == "Iris steps into the inn."


@pytest.mark.asyncio
async def test_advance_awaits_in_flight_speculation(tmp_app_data: Path) -> None:
    """If the player clicks an option BEFORE its speculative chain has
    finished, the handler awaits the in-flight task (one LLM call's
    worth of latency) rather than starting a competing fresh call.
    """

    class _DelayedLlm(CountingLlm):
        async def complete(self, prompt, *a, **kw):
            self.calls.append(prompt)
            nv = self._responses.pop(0) if self._responses else self._SPEC_FALLBACK
            # Yield once so the task is genuinely "in flight" when
            # play_advance starts checking.
            await asyncio.sleep(0)

            async def gen():
                yield nv

            return gen()

    llm = _DelayedLlm(
        responses=[
            _opt_list([f"c{i}" for i in range(6)]),
            _opt_list([f"n{i}" for i in range(8)]),
            _world_init_with_two_options(),
            _branch_payload("Iris steps into the inn."),
            _branch_payload("Iris scrambles up the cliff."),
        ]
    )
    session = Session(llm_client=llm, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _run_to_opening(ctx, registry, llm)
    # Don't drain speculation — fire play_advance with task pending.
    calls_before_advance = len(llm.calls)
    await _drain(
        registry.dispatch(
            Envelope(type=MessageType.c2s_play_advance, payload={"option_id": "left"}),
            ctx,
        )
    )
    delta = len(llm.calls) - calls_before_advance
    # Both speculative tasks were already in flight at confirm time
    # (one per option). They each consume one LLM call's worth, even
    # though only one was awaited; the other was already spawned and
    # finishes during the same await window. The walked branch has no
    # options, so NO additional foreground generation. Delta = 2.
    assert delta == 2, (
        f"expected the two pre-existing speculations to finish during the "
        f"play_advance await; saw {delta} LLM calls instead"
    )
    assert (
        session.game.dialog_tree.nodes[session.game.current_node_id].text
        == "Iris steps into the inn."
    )


@pytest.mark.asyncio
async def test_free_text_cancels_speculative_descendants(
    tmp_app_data: Path,
) -> None:
    """Free-text input invalidates downstream speculation: pending
    tasks rooted at the parent are cancelled."""

    class _PendingLlm(CountingLlm):
        async def complete(self, prompt, *a, **kw):
            self.calls.append(prompt)
            nv = self._responses.pop(0) if self._responses else self._SPEC_FALLBACK
            # Stall longer than this test will run for the spec branches
            # (5 yields are enough to make the cancellation observable).
            for _ in range(50):
                await asyncio.sleep(0)

            async def gen():
                yield nv

            return gen()

    llm = _PendingLlm(
        responses=[
            _opt_list([f"c{i}" for i in range(6)]),
            _opt_list([f"n{i}" for i in range(8)]),
            _world_init_with_two_options(),
        ]
    )
    session = Session(llm_client=llm, image_client=_NullImage())
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _run_to_opening(ctx, registry, llm)
    # Speculative tasks are mid-stall. Confirm they exist.
    tasks_before = dict(getattr(session, "_speculative_tasks", {}) or {})
    assert len(tasks_before) == 2
    assert all(not t.done() for t in tasks_before.values())

    # Fire free-text. The handler invalidates unwalked descendants AND
    # cancels speculative tasks under the current parent.
    free_text_response = _branch_payload("Iris does something unexpected.")
    llm._responses.append(free_text_response)

    await _drain(
        registry.dispatch(
            Envelope(
                type=MessageType.c2s_play_free_text,
                payload={"text": "Iris ignores both options."},
            ),
            ctx,
        )
    )

    # The two pre-existing speculative tasks must have been cancelled.
    for key, task in tasks_before.items():
        assert task.cancelled() or task.done(), (
            f"speculation {key} should have been cancelled by free_text"
        )


@pytest.mark.asyncio
async def test_save_load_kicks_off_next_beat_speculation(
    tmp_app_data: Path,
) -> None:
    """Loading a save mid-run must pre-generate the next beat (or one
    speculation per option) so the player's first Continue/choice
    lands as a tree-walk instead of stalling on an LLM round-trip.

    Setup: build a save whose current node is a 'continue' beat
    with no committed children, then call ``c2s/saves/load`` and
    verify a continue-speculation task was scheduled.
    """
    from lucidium.api.messages import C2SSavesLoad
    from lucidium.domain.character import Character
    from lucidium.domain.dialog import (
        DialogNode,
        DialogNodeState,
        DialogTree,
    )
    from lucidium.domain.game import Game
    from lucidium.domain.world import WorldState
    from lucidium.persistence import save_store

    pc = Character(
        id="pc",
        is_player=True,
        name="Iris",
        description="archivist",
        gender="female",
        age=28,
        ethnicity="local",
        skin="pale",
        hair_color="auburn",
        hairstyle="braid",
        eye_color="grey",
        build="slight",
        bust="moderate",
        outfit="wool coat",
        pose="standing",
        expression="alert",
        seed=1,
    )
    node = DialogNode(
        id="n0",
        parent_id=None,
        chosen_option_id=None,
        text="The harbor wakes slow.",
        options=[],  # continue node — exercises the new no-options path
        state=DialogNodeState.committed,
        premise_hash="h0",
    )
    game = Game(
        id="g-resume",
        world=WorldState(
            game_name="Resumed run",
            setting="harbor",
            genre="mystery",
            visual_style="ink",
        ),
        characters={"pc": pc},
        dialog_tree=DialogTree(
            nodes={"n0": node},
            root_id="n0",
            committed_path=["n0"],
        ),
        environments={},
        current_node_id="n0",
        on_stage=[],
    )
    # Persist via the production code path so the meta + game.json are
    # written exactly the way the load handler will read them.
    from lucidium.domain.settings import Settings as _Settings

    save_store.commit_save(game, _Settings(), root=tmp_app_data / "saves")

    # Slow-stall the LLM so the speculation task is observable as
    # in-flight when we look for it (otherwise it'd finish in the
    # same micro-yield and we'd see _speculative_tasks emptied).
    class _StallingLlm(CountingLlm):
        async def complete(self, prompt, *a, **kw):
            self.calls.append(prompt)
            for _ in range(50):
                await asyncio.sleep(0)

            async def gen():
                yield self._SPEC_FALLBACK

            return gen()

    llm = _StallingLlm(responses=[])
    session = Session(
        llm_client=llm,
        image_client=_NullImage(),
        saves_root=tmp_app_data / "saves",
    )
    ctx = HandlerContext(session=session)
    registry = build_default_registry()

    await _drain(
        registry.dispatch(
            Envelope(
                type=MessageType.c2s_saves_load,
                payload=C2SSavesLoad(save_id=game.id).model_dump(),
            ),
            ctx,
        )
    )

    # A speculation task should now be in flight for the continue
    # path on the resumed node.
    tasks = getattr(session, "_speculative_tasks", None) or {}
    assert tasks, (
        "saves_load handler should have spawned a continue-path "
        "speculation; the player's next Continue would otherwise "
        "have to wait through a synchronous LLM round-trip."
    )
    keys = list(tasks.keys())
    assert any(k.startswith("n0|") and k.endswith("__continue__") for k in keys), (
        f"expected n0|__continue__ speculation key, saw {keys}"
    )

    # Also verify an LLM call DID get fired (proving the
    # speculation actually started, not just registered).
    # Yield a few times so the task at least kicks off.
    for _ in range(5):
        await asyncio.sleep(0)
    assert len(llm.calls) >= 1, "speculation task ran but the LLM was never called"

    # Cancel the stalled task to keep test teardown tidy.
    for task in tasks.values():
        if not task.done():
            task.cancel()
    for task in tasks.values():
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
