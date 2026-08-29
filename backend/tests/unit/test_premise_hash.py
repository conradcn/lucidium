from __future__ import annotations

from lucidium.domain.dialog import premise_hash, world_snapshot_vector


def test_premise_hash_is_stable_across_runs() -> None:
    snapshot = world_snapshot_vector(
        on_stage_character_ids=["b", "a"],
        character_attributes={"a": {"pose": "standing"}, "b": {"pose": "sitting"}},
        world_fields={"plot": "rising", "setting": "harbor"},
    )
    first = premise_hash(parent_id="N1", chosen_option_id="O1", snapshot_vector=snapshot)
    second = premise_hash(parent_id="N1", chosen_option_id="O1", snapshot_vector=snapshot)
    assert first == second


def test_premise_hash_changes_with_attributes() -> None:
    snapshot_a = world_snapshot_vector(
        on_stage_character_ids=["a"],
        character_attributes={"a": {"pose": "standing"}},
        world_fields={},
    )
    snapshot_b = world_snapshot_vector(
        on_stage_character_ids=["a"],
        character_attributes={"a": {"pose": "kneeling"}},
        world_fields={},
    )
    assert premise_hash(
        parent_id="N1", chosen_option_id=None, snapshot_vector=snapshot_a
    ) != premise_hash(parent_id="N1", chosen_option_id=None, snapshot_vector=snapshot_b)


def test_premise_hash_changes_with_chosen_option() -> None:
    snapshot = world_snapshot_vector(
        on_stage_character_ids=[],
        character_attributes={},
        world_fields={},
    )
    assert premise_hash(
        parent_id="N1", chosen_option_id="O1", snapshot_vector=snapshot
    ) != premise_hash(parent_id="N1", chosen_option_id="O2", snapshot_vector=snapshot)


def test_premise_hash_is_order_independent_for_on_stage() -> None:
    snapshot_a = world_snapshot_vector(
        on_stage_character_ids=["a", "b"],
        character_attributes={"a": {}, "b": {}},
        world_fields={},
    )
    snapshot_b = world_snapshot_vector(
        on_stage_character_ids=["b", "a"],
        character_attributes={"a": {}, "b": {}},
        world_fields={},
    )
    assert premise_hash(
        parent_id=None, chosen_option_id=None, snapshot_vector=snapshot_a
    ) == premise_hash(parent_id=None, chosen_option_id=None, snapshot_vector=snapshot_b)
