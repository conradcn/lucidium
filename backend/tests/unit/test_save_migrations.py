"""Save schema versioning: refusal of future saves, and the migration hook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lucidium.api.errors import ErrorCode, SaveVersionError, from_exception
from lucidium.config import GAME_SCHEMA_VERSION
from lucidium.domain.dialog import DialogNode, DialogTree, premise_hash, world_snapshot_vector
from lucidium.domain.game import Game
from lucidium.domain.settings import Settings
from lucidium.domain.world import WorldState
from lucidium.persistence import save_migrations
from lucidium.persistence.save_store import _save_dir, commit_save, load_save


def _make_game(name: str = "Embers") -> Game:
    snapshot = world_snapshot_vector(
        on_stage_character_ids=[],
        character_attributes={},
        world_fields={"plot": "opening"},
    )
    node = DialogNode(
        text="The harbor wakes.",
        premise_hash=premise_hash(parent_id=None, chosen_option_id=None, snapshot_vector=snapshot),
    )
    return Game(
        world=WorldState(game_name=name, setting="harbor", genre="mystery", visual_style="ink"),
        dialog_tree=DialogTree(nodes={node.id: node}, root_id=node.id, committed_path=[node.id]),
        current_node_id=node.id,
    )


def _stamp_version(save_dir: Path, game: Game, version: int) -> None:
    """Write a ``game.json`` carrying an arbitrary ``schema_version``.

    Goes through ``json`` rather than the model so we can produce a
    document the current build would refuse to build in the first place.
    """
    document = json.loads(game.model_dump_json())
    document["schema_version"] = version
    (save_dir / "game.json").write_text(json.dumps(document, indent=2), encoding="utf-8")


def test_save_from_newer_build_is_refused_with_a_recoverable_error(
    tmp_app_data: Path,
) -> None:
    game = _make_game("From the future")
    commit_save(game, Settings(), name="Future")
    _stamp_version(_save_dir(game.id), game, 99)

    with pytest.raises(SaveVersionError) as excinfo:
        load_save(game.id)

    structured = from_exception(excinfo.value)
    assert structured.recoverable is True, (
        "nothing is broken and nothing was written — the player just needs "
        "a newer build, which is the definition of recoverable"
    )
    assert structured.code is ErrorCode.schema_error
    assert "newer version of Lucidium" in structured.message
    assert "99" in structured.message, "the message must name the save's version"

    # The refusal must not have touched the file: a downgrade-and-recommit
    # is exactly the silent data loss this check exists to prevent.
    on_disk = json.loads((_save_dir(game.id) / "game.json").read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == 99


def test_current_version_save_round_trips(tmp_app_data: Path) -> None:
    game = _make_game("Embers")
    commit_save(game, Settings(), name="Run 1")

    on_disk = json.loads((_save_dir(game.id) / "game.json").read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == GAME_SCHEMA_VERSION

    loaded = load_save(game.id)
    assert loaded.id == game.id
    assert loaded.schema_version == GAME_SCHEMA_VERSION
    assert loaded.world.game_name == "Embers"


def test_registered_migration_runs_exactly_once_on_load(
    tmp_app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no-op registered for the current version is invoked once per load.

    This is the seam the next real migration plugs into, so the contract
    worth pinning is "called, and called once" — a chain that re-ran a
    step would double-apply any rewrite it performs.
    """
    calls: list[int] = []

    def _noop(data: dict[str, Any]) -> dict[str, Any]:
        calls.append(1)
        return data

    monkeypatch.setitem(save_migrations.MIGRATIONS, GAME_SCHEMA_VERSION, _noop)

    game = _make_game("Migrated")
    commit_save(game, Settings(), name="Run 1")
    loaded = load_save(game.id)

    assert len(calls) == 1
    assert loaded.world.game_name == "Migrated"


def test_older_save_is_upgraded_through_the_chain(
    tmp_app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An out-of-date save is rewritten before validation, not rejected.

    Simulated by registering a step from ``current - 1`` that repairs a
    field the "old" document is missing, which is the shape every real
    migration will take.
    """
    older = GAME_SCHEMA_VERSION - 1
    seen: list[int] = []

    def _upgrade(data: dict[str, Any]) -> dict[str, Any]:
        seen.append(older)
        data["world"]["game_name"] = "Upgraded"
        return data

    monkeypatch.setitem(save_migrations.MIGRATIONS, older, _upgrade)

    game = _make_game("Stale")
    commit_save(game, Settings(), name="Run 1")
    _stamp_version(_save_dir(game.id), game, older)

    loaded = load_save(game.id)
    assert seen == [older]
    assert loaded.world.game_name == "Upgraded"
    assert loaded.schema_version == GAME_SCHEMA_VERSION, (
        "the chain must restamp the payload at the version it reached"
    )


def test_unmigratable_validation_failure_is_user_actionable(tmp_app_data: Path) -> None:
    """``extra="forbid"`` breakage reaches the player as an actionable error.

    Before this branch existed it surfaced as ``code=internal``,
    ``recoverable=False`` plus verbatim pydantic text quoting on-disk
    values.
    """
    from pydantic import ValidationError

    game = _make_game("Embers")
    commit_save(game, Settings(), name="Run 1")
    document = json.loads(game.model_dump_json())
    document["weather_system"] = {"rain": True}
    (_save_dir(game.id) / "game.json").write_text(json.dumps(document, indent=2), encoding="utf-8")

    with pytest.raises(ValidationError) as excinfo:
        load_save(game.id)

    structured = from_exception(excinfo.value)
    assert structured.code is ErrorCode.schema_error
    assert structured.recoverable is True
    assert "weather_system" in structured.message
    assert "different version" in structured.message
