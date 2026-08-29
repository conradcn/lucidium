from __future__ import annotations

from lucidium.domain.dialog import (
    DialogNode,
    DialogNodeState,
    DialogTree,
    premise_hash,
    world_snapshot_vector,
)
from lucidium.domain.game import Game
from lucidium.domain.world import WorldState
from lucidium.orchestration.cost import (
    CostDelta,
    apply_delta,
    estimate_image_call,
    estimate_llm_call,
    estimate_tokens,
    fold_into_game,
)


def _ph() -> str:
    return premise_hash(
        parent_id=None,
        chosen_option_id=None,
        snapshot_vector=world_snapshot_vector(
            on_stage_character_ids=[], character_attributes={}, world_fields={}
        ),
    )


def _make_game() -> Game:
    node = DialogNode(premise_hash=_ph(), state=DialogNodeState.committed)
    return Game(
        world=WorldState(game_name="t", setting="s", genre="g", visual_style="v"),
        dialog_tree=DialogTree(nodes={node.id: node}, root_id=node.id, committed_path=[node.id]),
        current_node_id=node.id,
    )


def test_estimate_tokens_minimum_one() -> None:
    assert estimate_tokens("a") == 1


def test_estimate_tokens_scales_with_chars() -> None:
    assert estimate_tokens("a" * 80) == 20


def test_estimate_llm_call_records_call_count_and_latency() -> None:
    delta = estimate_llm_call(prompt_chars=400, reply_chars=120, latency_ms=750)
    assert delta.llm_calls == 1
    assert delta.tokens_in == 100
    assert delta.tokens_out == 30
    assert delta.latency_ms == 750


def test_apply_delta_accumulates() -> None:
    game = _make_game()
    accumulated = apply_delta(
        apply_delta(game.cost_telemetry, CostDelta(tokens_in=10, llm_calls=1, latency_ms=200)),
        estimate_image_call(latency_ms=400),
    )
    assert accumulated.llm_calls == 1
    assert accumulated.image_calls == 1
    assert accumulated.tokens_in == 10
    assert accumulated.latency_ms_total == 600


def test_fold_into_game_returns_new_instance() -> None:
    game = _make_game()
    updated = fold_into_game(game, CostDelta(llm_calls=1, tokens_in=4, latency_ms=10))
    assert updated.cost_telemetry.llm_calls == 1
    assert game.cost_telemetry.llm_calls == 0  # original unchanged
