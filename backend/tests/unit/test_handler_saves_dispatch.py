"""``c2s/saves/*`` driven through ``HandlerRegistry.dispatch``.

The saves handlers are the only ones on the unauthenticated socket that
reach ``shutil.rmtree`` and ``atomic_write_text``, and the confinement
that keeps them honest is split across two layers:

  * the wire schema (``_SAVE_ID_PATTERN``), enforced inside ``dispatch``;
  * ``save_store._save_dir``, enforced under the handler.

``tests/unit/test_handler_path_confinement.py`` covers each layer in
isolation. These tests cover the composition — a raw payload dict going
in the front door, which is the shape an attacker actually controls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lucidium.api.errors import NotFoundError, SchemaError
from lucidium.api.handlers import HandlerContext
from lucidium.api.messages import MessageType
from lucidium.domain.settings import Settings
from lucidium.persistence import save_store

from .handler_harness import (
    dispatch,
    make_game,
    make_registry,
    make_session,
    types_of,
)

# Payloads a hostile (or merely buggy) renderer could put on the wire.
# ``..``-relative, absolute POSIX, absolute Windows, and UNC.
ESCAPING_SAVE_IDS = [
    "../../..",
    "..",
    "../sibling",
    "/etc",
    "C:/Windows",
    "C:\\Windows\\System32",
    "\\\\server\\share",
    "sub/dir",
    "",
]


def _seed_save(session, *, name: str) -> str:
    """Commit one real save into the session's saves root, return its id."""
    game, *_ = make_game()
    game = game.model_copy(update={"world": game.world.model_copy(update={"game_name": name})})
    save_store.commit_save(game, Settings(), name=name, root=session.saves_root)
    return game.id


# ---------------------------------------------------------------------------
# c2s/saves/list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_saves_list_on_an_empty_root_returns_an_empty_list(
    tmp_app_data: Path,
) -> None:
    session = make_session(tmp_app_data)
    messages = await dispatch(
        make_registry(), HandlerContext(session=session), MessageType.c2s_saves_list
    )

    assert types_of(messages) == [MessageType.s2c_saves_list]
    assert messages[0][1].saves == []


@pytest.mark.asyncio
async def test_saves_list_reports_committed_saves(tmp_app_data: Path) -> None:
    session = make_session(tmp_app_data)
    save_id = _seed_save(session, name="Run 1")

    messages = await dispatch(
        make_registry(), HandlerContext(session=session), MessageType.c2s_saves_list
    )

    summaries = messages[0][1].saves
    assert [s.id for s in summaries] == [save_id]
    assert summaries[0].name == "Run 1"
    assert summaries[0].corrupt is False


# ---------------------------------------------------------------------------
# c2s/saves/rename
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_saves_rename_renames_and_echoes_the_refreshed_list(
    tmp_app_data: Path,
) -> None:
    session = make_session(tmp_app_data)
    save_id = _seed_save(session, name="Run 1")

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_saves_rename,
        {"save_id": save_id, "new_name": "Second attempt"},
    )

    assert types_of(messages) == [MessageType.s2c_saves_list]
    assert [s.name for s in messages[0][1].saves] == ["Second attempt"]
    # And it landed on disk, not just in the echo.
    assert save_store.list_saves(session.saves_root)[0].name == "Second attempt"


@pytest.mark.asyncio
@pytest.mark.parametrize("save_id", ESCAPING_SAVE_IDS)
async def test_saves_rename_rejects_path_shaped_ids(tmp_app_data: Path, save_id: str) -> None:
    session = make_session(tmp_app_data)
    with pytest.raises(SchemaError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_saves_rename,
            {"save_id": save_id, "new_name": "pwned"},
        )


# ---------------------------------------------------------------------------
# c2s/saves/delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_saves_delete_removes_the_directory_and_echoes_the_list(
    tmp_app_data: Path,
) -> None:
    session = make_session(tmp_app_data)
    save_id = _seed_save(session, name="Run 1")
    assert (session.saves_root / save_id).is_dir()

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_saves_delete,
        {"save_id": save_id},
    )

    assert types_of(messages) == [MessageType.s2c_saves_list]
    assert messages[0][1].saves == []
    assert not (session.saves_root / save_id).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("save_id", ESCAPING_SAVE_IDS)
async def test_saves_delete_rejects_path_shaped_ids_before_rmtree(
    tmp_app_data: Path, save_id: str
) -> None:
    """The abuse case this whole file exists for.

    ``saves_delete`` is a direct line from an unauthenticated socket
    frame to ``shutil.rmtree``. ``dispatch`` must refuse the frame at
    validation — before the handler runs at all — and the sibling
    directory outside the saves root must still be there afterwards.
    """
    session = make_session(tmp_app_data)
    victim = tmp_app_data / "not-a-save"
    victim.mkdir()
    (victim / "precious.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(SchemaError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_saves_delete,
            {"save_id": save_id},
        )

    assert (victim / "precious.txt").read_text(encoding="utf-8") == "keep me"


@pytest.mark.asyncio
async def test_saves_delete_of_an_absolute_path_leaves_that_path_alone(
    tmp_app_data: Path,
) -> None:
    """``Path("saves") / "C:/Windows"`` is ``C:/Windows`` — the join
    silently discards the base when the right operand is absolute. This
    pins the behaviour with a real absolute path we own."""
    session = make_session(tmp_app_data)
    outside = tmp_app_data / "outside"
    outside.mkdir()
    (outside / "file.bin").write_bytes(b"x")

    with pytest.raises(SchemaError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_saves_delete,
            {"save_id": str(outside)},
        )

    assert (outside / "file.bin").exists()


# ---------------------------------------------------------------------------
# c2s/saves/load — same id surface, so the same abuse cases apply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_saves_load_installs_the_game(tmp_app_data: Path) -> None:
    session = make_session(tmp_app_data)
    save_id = _seed_save(session, name="Run 1")

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_saves_load,
        {"save_id": save_id},
    )

    assert types_of(messages) == [MessageType.s2c_state_full]
    assert session.game is not None
    assert session.game.id == save_id
    assert session.game.world.game_name == "Run 1"


@pytest.mark.asyncio
@pytest.mark.parametrize("save_id", ESCAPING_SAVE_IDS)
async def test_saves_load_rejects_path_shaped_ids(tmp_app_data: Path, save_id: str) -> None:
    session = make_session(tmp_app_data)
    with pytest.raises(SchemaError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_saves_load,
            {"save_id": save_id},
        )
    assert session.game is None


@pytest.mark.asyncio
async def test_saves_load_of_an_unknown_but_well_formed_id_fails_cleanly(
    tmp_app_data: Path,
) -> None:
    """A schema-legal id that names no save must not install anything.

    It surfaces as an OSError rather than a ``LucidiumError`` today; the
    assertion is deliberately loose about which, and strict about the
    session being left untouched.
    """
    session = make_session(tmp_app_data)
    with pytest.raises((OSError, NotFoundError, SchemaError)):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_saves_load,
            {"save_id": "no-such-save"},
        )
    assert session.game is None
