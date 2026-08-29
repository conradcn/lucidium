from __future__ import annotations

import json
from pathlib import Path

import pytest

from lucidium.config import GAME_SCHEMA_VERSION
from lucidium.domain.dialog import DialogNode, DialogTree, premise_hash, world_snapshot_vector
from lucidium.domain.game import Game
from lucidium.domain.settings import Settings
from lucidium.domain.world import WorldState
from lucidium.persistence.save_store import (
    commit_save,
    delete_save,
    list_saves,
    load_save,
    most_recent_save_id,
    rename_save,
)


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


def test_commit_load_round_trip(tmp_app_data: Path) -> None:
    game = _make_game()
    settings = Settings()
    meta = commit_save(game, settings, name="Run 1")
    assert meta.name == "Run 1"
    assert meta.summary == "The harbor wakes."

    loaded = load_save(game.id)
    assert loaded.id == game.id
    assert loaded.world.game_name == "Embers"


def test_list_saves_orders_by_recency(tmp_app_data: Path) -> None:
    a = _make_game("Alpha")
    b = _make_game("Beta")
    settings = Settings()
    commit_save(a, settings, name="A")
    commit_save(b, settings, name="B")
    summaries = list_saves()
    assert [s.name for s in summaries[:2]] == ["B", "A"]
    assert most_recent_save_id() == b.id


def test_rename_and_delete(tmp_app_data: Path) -> None:
    game = _make_game()
    commit_save(game, Settings(), name="Original")
    updated = rename_save(game.id, "Renamed")
    assert updated.name == "Renamed"
    assert list_saves()[0].name == "Renamed"
    delete_save(game.id)
    assert list_saves() == []


def test_untitled_meta_is_recovered_from_world_game_name(
    tmp_app_data: Path,
) -> None:
    """Older saves committed before ``Session.commit`` threaded
    ``world.game_name`` as the default landed on the "Untitled"
    placeholder. The load-game UI must still show their real name —
    ``list_saves`` recovers it from the game.json's
    ``world.game_name`` so the player isn't staring at five
    "Untitled" rows."""
    game = _make_game(name="Embers of the Veil")
    # Simulate the legacy commit path — caller passes no name AND
    # the old precedence didn't fall through to the world name.
    commit_save(game, Settings())
    # Hand-edit the meta to the literal placeholder, mimicking what
    # the historical save_store would have written.
    from lucidium.persistence.save_store import _meta_path

    meta_file = _meta_path(game.id)
    meta_text = meta_file.read_text(encoding="utf-8").replace(
        '"name": "Embers of the Veil"', '"name": "Untitled"'
    )
    meta_file.write_text(meta_text, encoding="utf-8")

    # ``list_saves`` should surface the world's game name even
    # though the meta on disk still says "Untitled".
    summaries = list_saves()
    assert len(summaries) == 1
    assert summaries[0].name == "Embers of the Veil"


def test_commit_upgrades_untitled_meta_on_next_commit(
    tmp_app_data: Path,
) -> None:
    """Once the player turns again, ``commit_save`` rewrites the
    meta with the real game name — the fix is permanent, not just
    cosmetic at list time."""
    game = _make_game(name="Embers of the Veil")
    commit_save(game, Settings(), name="Untitled")
    # Re-commit with the world's game name as the default.
    meta = commit_save(game, Settings(), name="Embers of the Veil")
    assert meta.name == "Embers of the Veil"


def test_player_rename_survives_commit_with_default(
    tmp_app_data: Path,
) -> None:
    """A name the player chose explicitly (anything other than the
    "Untitled" placeholder) wins over the fallback default. Without
    this the fallback would clobber a player rename on every
    subsequent commit."""
    game = _make_game(name="Embers of the Veil")
    commit_save(game, Settings(), name="Untitled")
    rename_save(game.id, "My run with Mira")
    # Subsequent commit passes the world name as default — must NOT
    # overwrite the player's rename.
    meta = commit_save(game, Settings(), name="Embers of the Veil")
    assert meta.name == "My run with Mira"


def test_corrupt_meta_does_not_hide_intact_saves(tmp_app_data: Path) -> None:
    """One unparseable ``meta.json`` used to make ``list_saves`` raise,
    which took every intact save with it — and with no save_id to hand
    out, Delete/Rename/Continue all died too. The corrupt directory is
    now listed as a placeholder entry the player can delete."""
    from lucidium.persistence.save_store import _meta_path

    good_a = _make_game("Alpha")
    good_b = _make_game("Beta")
    broken = _make_game("Broken")
    for game, name in ((good_a, "A"), (good_b, "B"), (broken, "Doomed")):
        commit_save(game, Settings(), name=name)
    _meta_path(broken.id).write_text('{"bogus":1}', encoding="utf-8")

    summaries = list_saves()
    assert len(summaries) == 3
    by_id = {s.id: s for s in summaries}
    assert by_id[good_a.id].name == "A"
    assert by_id[good_b.id].name == "B"
    assert not by_id[good_a.id].corrupt
    assert not by_id[good_b.id].corrupt
    assert by_id[broken.id].corrupt is True

    # Continue must skip the corrupt entry rather than try to load it.
    assert most_recent_save_id() in {good_a.id, good_b.id}

    # The listed id is the one the delete path takes.
    delete_save(broken.id)
    assert {s.id for s in list_saves()} == {good_a.id, good_b.id}


def test_commit_over_corrupt_meta_rewrites_it(tmp_app_data: Path) -> None:
    """An unparseable meta counts as absent, so saving keeps working."""
    from lucidium.persistence.save_store import _meta_path

    game = _make_game("Embers of the Veil")
    commit_save(game, Settings(), name="Run 1")
    _meta_path(game.id).write_text('{"bogus":1}', encoding="utf-8")

    meta = commit_save(game, Settings(), name="Embers of the Veil")
    assert meta.name == "Embers of the Veil"
    assert list_saves()[0].corrupt is False


def _legacy_placeholder_meta(save_id: str, name: str = "Untitled") -> None:
    """Rewrite a meta the way a pre-``name_recovered`` build wrote it.

    Saves committed by older builds have no ``name_recovered`` key at
    all, which is exactly the state that must still trigger recovery.
    """
    from lucidium.persistence.save_store import _meta_path

    meta_file = _meta_path(save_id)
    document = json.loads(meta_file.read_text(encoding="utf-8"))
    document["name"] = name
    document.pop("name_recovered", None)
    meta_file.write_text(json.dumps(document, indent=2), encoding="utf-8")


def test_recovered_name_is_written_back_to_meta(tmp_app_data: Path) -> None:
    """Recovery parses the whole game.json, so it must happen once.

    The derived name is persisted into ``meta.json``; the second
    listing reads it straight out of the meta with no game.json parse.
    """
    game = _make_game(name="Embers of the Veil")
    commit_save(game, Settings())
    _legacy_placeholder_meta(game.id)

    assert list_saves()[0].name == "Embers of the Veil"

    from lucidium.persistence.save_store import _meta_path

    document = json.loads(_meta_path(game.id).read_text(encoding="utf-8"))
    assert document["name"] == "Embers of the Veil"
    assert document["name_recovered"] is True

    # Deleting game.json proves the second listing never touches it.
    from lucidium.persistence.save_store import _save_dir

    (_save_dir(game.id) / "game.json").unlink()
    assert list_saves()[0].name == "Embers of the Veil"


def test_unrecoverable_placeholder_is_not_reparsed(tmp_app_data: Path) -> None:
    """The pathological case: ``world.game_name`` is empty, so recovery
    derives nothing and the save stays "Untitled". It used to re-parse
    the full game.json on *every* listing to rediscover that — full
    cost, zero benefit, forever. One attempt is enough."""
    game = _make_game(name="")
    commit_save(game, Settings())
    _legacy_placeholder_meta(game.id)

    assert list_saves()[0].name == "Untitled"

    from lucidium.persistence.save_store import _meta_path

    document = json.loads(_meta_path(game.id).read_text(encoding="utf-8"))
    assert document["name"] == "Untitled", "the placeholder must be kept as-is"
    assert document["name_recovered"] is True, "the dead end must be recorded"

    from lucidium.persistence.save_store import _save_dir

    (_save_dir(game.id) / "game.json").unlink()
    assert list_saves()[0].name == "Untitled"


def test_commit_settles_recovery_when_world_has_no_name(tmp_app_data: Path) -> None:
    """``commit_save`` holds the live Game, so it can answer for free
    the question recovery would re-parse a save to ask."""
    unnamed = _make_game(name="")
    assert commit_save(unnamed, Settings()).name_recovered is True

    # ...but a placeholder meta over a world that DOES have a name must
    # stay recoverable: commit_save itself never falls back to it.
    named = _make_game(name="Embers of the Veil")
    assert commit_save(named, Settings()).name_recovered is False
    assert list_saves()[0].name == "Embers of the Veil"


def _make_big_game(name: str, nodes: int = 1000) -> Game:
    """A ~1000-node save, the size the listing benchmark cares about."""
    game = _make_game(name=name)
    root = game.dialog_tree.nodes[game.dialog_tree.root_id]
    for step in range(nodes - 1):
        node = DialogNode(
            text=f"Node {step}: " + "the harbor wakes and the lamps gutter. " * 4,
            parent_id=root.id,
            premise_hash=premise_hash(
                parent_id=root.id,
                chosen_option_id=None,
                snapshot_vector=world_snapshot_vector(
                    on_stage_character_ids=[],
                    character_attributes={},
                    world_fields={"plot": f"beat-{step}"},
                ),
            ),
        )
        game.dialog_tree.nodes[node.id] = node
    return game


def _timed(call):
    """Steady-state wall time in ms, after one warm-up call.

    Writing eight ~1.4 MB saves leaves the OS flushing and (on Windows)
    the AV scanner walking the directory, which shows up as 60-250 ms of
    noise on the first couple of reads regardless of what the code does.
    The warm-up call takes that out so the number reflects the work
    ``list_saves`` actually performs.
    """
    import time

    call()
    start = time.perf_counter()
    result = call()
    return result, (time.perf_counter() - start) * 1000


@pytest.fixture()
def count_game_parses(monkeypatch: pytest.MonkeyPatch):
    """Count full ``game.json`` parses — the cost this task is about.

    Timings guard the headline number but vary by machine; the parse
    count is the exact, machine-independent statement of the fix.
    """
    calls = {"n": 0}
    original = Game.model_validate_json

    def counted(*args: object, **kwargs: object):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(Game, "model_validate_json", counted)
    return calls


def test_list_saves_does_not_reparse_unrecoverable_placeholders(
    tmp_app_data: Path, count_game_parses
) -> None:
    """Regression benchmark for the pathological placeholder case.

    Eight 1000-node saves whose ``world.game_name`` is empty, so the
    meta stays "Untitled". This used to re-parse all eight game.json
    files (~14 ms each, ~114 ms total) on *every* listing to rederive
    the same empty string. ``commit_save`` now settles the question
    while it still has the live Game, so the listing parses nothing.
    """
    for _ in range(8):
        commit_save(_make_big_game(""), Settings())

    count_game_parses["n"] = 0
    summaries, elapsed_ms = _timed(list_saves)

    assert len(summaries) == 8
    assert all(s.name == "Untitled" for s in summaries)
    assert count_game_parses["n"] == 0, (
        f"{count_game_parses['n']} game.json parses to rederive a name that is empty by definition"
    )
    # Generous bound: this guards an order-of-magnitude regression, not
    # a specific machine's timings. Was ~114 ms.
    assert elapsed_ms < 30, f"list_saves took {elapsed_ms:.1f} ms"

    count_game_parses["n"] = 0
    _, recent_ms = _timed(most_recent_save_id)
    assert count_game_parses["n"] == 0
    assert recent_ms < 30, f"most_recent_save_id took {recent_ms:.1f} ms"


def test_legacy_placeholder_saves_are_recovered_once(tmp_app_data: Path, count_game_parses) -> None:
    """Legacy saves must still pay the recovery parse — but only once.

    These metas predate ``name_recovered``, so the first listing does
    parse all eight game.json files to dig out the real names. It then
    writes them back, and every later listing reads metas only.
    """
    for index in range(8):
        game = _make_big_game(f"Saga {index}")
        commit_save(game, Settings())
        _legacy_placeholder_meta(game.id)

    count_game_parses["n"] = 0
    first = list_saves()
    assert count_game_parses["n"] == 8, "the one-off recovery parse"
    assert {s.name for s in first} == {f"Saga {i}" for i in range(8)}

    count_game_parses["n"] = 0
    second, second_ms = _timed(list_saves)
    assert count_game_parses["n"] == 0, "the recovered names are not sticking"
    assert [s.name for s in first] == [s.name for s in second]
    assert second_ms < 30, f"cached list_saves took {second_ms:.1f} ms"


# --------------------------------------------------------------------------
# Schema-version and partial-write handling.
#
# The three ``xfail(strict=True)`` tests below encode behaviour the save
# layer does NOT have yet. They are strict so that whoever fixes the
# production code is told to drop the marker rather than leaving a test
# that silently stops asserting anything. See the module-level notes in
# each test for what fails today.
# --------------------------------------------------------------------------


def _write_game_json(save_dir: Path, game: Game, **overrides: object) -> None:
    """Write a hand-doctored ``game.json`` into ``save_dir``.

    Goes through ``json`` rather than the model so we can produce
    documents the current model would refuse to *build* — an unknown
    field, or a ``schema_version`` from a future release.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    document = json.loads(game.model_dump_json())
    document.update(overrides)
    (save_dir / "game.json").write_text(json.dumps(document, indent=2), encoding="utf-8")


def test_save_with_game_json_but_no_meta_is_still_reachable(
    tmp_app_data: Path,
) -> None:
    """A half-committed save must not vanish from the load screen."""
    from lucidium.persistence.save_store import _meta_path, _save_dir

    intact = _make_game("Alpha")
    commit_save(intact, Settings(), name="A")

    partial = _make_game("Interrupted")
    commit_save(partial, Settings(), name="Interrupted")
    _meta_path(partial.id).unlink()
    assert (_save_dir(partial.id) / "game.json").exists()

    summaries = list_saves()
    by_id = {s.id: s for s in summaries}
    assert partial.id in by_id, (
        "a save directory holding a game.json but no meta.json was dropped "
        "from list_saves entirely — it can never be deleted or resumed"
    )
    assert by_id[intact.id].name == "A"
    # Continue must not land on the half-written entry.
    assert most_recent_save_id() == intact.id


def test_meta_less_save_is_reconstructed_from_game_json(tmp_app_data: Path) -> None:
    """A crashed first commit leaves game.json + images/ and nothing else.

    The directory is self-describing: the listing must derive a usable id
    and a real display name from the game file, and the save must load.
    """
    from lucidium.persistence.save_store import _meta_path, _save_dir

    game = _make_game("Embers of the Veil")
    commit_save(game, Settings(), name="Embers of the Veil")
    _meta_path(game.id).unlink()
    assert (_save_dir(game.id) / "images").is_dir()

    entry = next(s for s in list_saves() if s.id == game.id)
    assert entry.name == "Embers of the Veil"
    assert entry.corrupt is False
    assert entry.schema_version == GAME_SCHEMA_VERSION

    loaded = load_save(entry.id)
    assert loaded.id == game.id
    assert loaded.world.game_name == "Embers of the Veil"


def test_meta_less_save_without_world_name_gets_a_placeholder(
    tmp_app_data: Path,
) -> None:
    from lucidium.persistence.save_store import _meta_path

    game = _make_game(name="")
    commit_save(game, Settings(), name="")
    _meta_path(game.id).unlink()

    entry = next(s for s in list_saves() if s.id == game.id)
    assert entry.name.strip()  # something the load screen can render
    assert load_save(entry.id).id == game.id


def test_directory_with_unreadable_game_and_no_meta_is_skipped(
    tmp_app_data: Path,
) -> None:
    from lucidium.config import saves_dir

    stray = saves_dir() / "not-a-save"
    stray.mkdir(parents=True)
    (stray / "game.json").write_text("{ truncated", encoding="utf-8")

    assert [s.id for s in list_saves()] == []


def test_future_schema_version_is_rejected_not_silently_downgraded(
    tmp_app_data: Path,
) -> None:
    from lucidium.persistence.save_store import _save_dir

    game = _make_game("From the future")
    commit_save(game, Settings(), name="Future")
    future_version = GAME_SCHEMA_VERSION + 1
    _write_game_json(_save_dir(game.id), game, schema_version=future_version)

    with pytest.raises(ValueError) as excinfo:
        load_save(game.id)
    assert str(future_version) in str(excinfo.value), (
        "the error must name the version so the UI can explain itself"
    )

    # And the on-disk save must survive the refusal untouched — no
    # downgrade-and-recommit.
    on_disk = json.loads((_save_dir(game.id) / "game.json").read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == future_version


def test_unknown_field_in_game_json_is_rejected(tmp_app_data: Path) -> None:
    """Documents the ``extra="forbid"`` contract on ``Game``.

    This is the flip side of the version test above: because ``Game``
    forbids extras, a save written by a newer build that added ANY field
    fails to load with a bare Pydantic error that says nothing about
    versions. That is why the future-version check has to happen first —
    it is the only place a useful message can come from.
    """
    from pydantic import ValidationError

    from lucidium.persistence.save_store import _save_dir

    game = _make_game("Embers")
    commit_save(game, Settings(), name="Run 1")
    _write_game_json(_save_dir(game.id), game, weather_system={"rain": True})

    with pytest.raises(ValidationError) as excinfo:
        load_save(game.id)
    assert "weather_system" in str(excinfo.value)
