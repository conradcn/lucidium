"""``c2s/music/inventory`` and ``c2s/app/exit`` through dispatch.

``music/inventory`` is the Settings screen's "Test connection" button. It
is deliberately non-raising: every failure mode comes back as
``ok=false`` plus a human-readable string, because the renderer degrades
to a free-text model input rather than showing an error banner. That
contract is only meaningful if the failure paths are pinned, so each of
them is covered here.

``app/exit`` is the last thing that runs before the window closes, and
its whole job is to flush the live game to disk first. The test that
matters is the one asserting the save landed.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from lucidium.api.handlers import HandlerContext
from lucidium.api.messages import MessageType
from lucidium.domain.settings import MusicSettings, Settings
from lucidium.persistence import save_store

from .handler_harness import (
    dispatch,
    make_game,
    make_registry,
    make_session,
    types_of,
)

MUSIC_URL = "http://music.invalid:7860"
INVENTORY_URL = f"{MUSIC_URL}/v1/model_inventory"

# The shape an ACE-Step v1.0 server actually returns.
INVENTORY_BODY = {
    "code": 200,
    "data": {
        "models": [
            {"name": "ace-step-v1-3.5b", "is_default": False, "is_loaded": True},
            {"name": "ace-step-v1-tiny", "is_default": True, "is_loaded": False},
        ],
        "default_model": "ace-step-v1-tiny",
        "lm_models": [{"name": "should-not-appear", "is_loaded": False}],
        "loaded_lm_model": "should-not-appear",
    },
}


def _music_settings() -> Settings:
    return Settings(music=MusicSettings(base_url=MUSIC_URL))


# ---------------------------------------------------------------------------
# c2s/music/inventory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_music_inventory_reports_the_dit_models_default_first(
    tmp_app_data: Path, http_router: respx.MockRouter
) -> None:
    http_router.get(INVENTORY_URL).mock(return_value=httpx.Response(200, json=INVENTORY_BODY))
    session = make_session(tmp_app_data, settings=_music_settings())

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_music_inventory,
        {},
    )

    assert types_of(messages) == [MessageType.s2c_music_inventory]
    reply = messages[0][1]
    assert reply.ok is True
    assert reply.error == ""
    assert reply.base_url == MUSIC_URL
    # Default is pulled to the head; the LM list is excluded entirely.
    assert reply.models == ["ace-step-v1-tiny", "ace-step-v1-3.5b"]
    assert "should-not-appear" not in reply.models


@pytest.mark.asyncio
async def test_music_inventory_probes_the_url_from_the_payload_not_settings(
    tmp_app_data: Path, http_router: respx.MockRouter
) -> None:
    """The Settings screen probes a URL the player is still editing, so
    the wire value has to win over the stored one — WITHOUT committing
    it."""
    probe_url = "http://being-edited.invalid:9999"
    http_router.get(f"{probe_url}/v1/model_inventory").mock(
        return_value=httpx.Response(200, json=INVENTORY_BODY)
    )
    session = make_session(tmp_app_data, settings=_music_settings())

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_music_inventory,
        {"base_url": probe_url},
    )

    assert messages[0][1].base_url == probe_url
    assert messages[0][1].ok is True
    # Nothing was persisted — the stored setting is untouched.
    assert session.settings.music.base_url == MUSIC_URL


@pytest.mark.asyncio
async def test_music_inventory_reports_an_unreachable_server_without_raising(
    tmp_app_data: Path, http_router: respx.MockRouter
) -> None:
    http_router.get(INVENTORY_URL).mock(side_effect=httpx.ConnectError("connection refused"))
    session = make_session(tmp_app_data, settings=_music_settings())

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_music_inventory,
        {},
    )

    reply = messages[0][1]
    assert reply.ok is False
    assert reply.models == []
    assert "unreachable" in reply.error


@pytest.mark.asyncio
async def test_music_inventory_reports_a_4xx_as_a_server_error(
    tmp_app_data: Path, http_router: respx.MockRouter
) -> None:
    http_router.get(INVENTORY_URL).mock(return_value=httpx.Response(404, text="no such endpoint"))
    session = make_session(tmp_app_data, settings=_music_settings())

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_music_inventory,
        {},
    )

    reply = messages[0][1]
    assert reply.ok is False
    assert reply.error.startswith("server error")


@pytest.mark.asyncio
async def test_music_inventory_reports_a_5xx_as_unreachable(
    tmp_app_data: Path, http_router: respx.MockRouter
) -> None:
    http_router.get(INVENTORY_URL).mock(return_value=httpx.Response(500, text="boom"))
    session = make_session(tmp_app_data, settings=_music_settings())

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_music_inventory,
        {},
    )

    assert messages[0][1].ok is False
    assert "unreachable" in messages[0][1].error


@pytest.mark.asyncio
async def test_music_inventory_survives_a_non_json_body(
    tmp_app_data: Path, http_router: respx.MockRouter
) -> None:
    """A reverse proxy serving an HTML error page is a real failure mode
    on a locally-run server; it must not escape as an exception."""
    http_router.get(INVENTORY_URL).mock(return_value=httpx.Response(200, text="<html>nope</html>"))
    session = make_session(tmp_app_data, settings=_music_settings())

    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_music_inventory,
        {},
    )

    assert messages[0][1].ok is False
    assert messages[0][1].models == []


# ---------------------------------------------------------------------------
# c2s/app/exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_app_exit_flushes_the_live_game_to_disk(tmp_app_data: Path) -> None:
    session = make_session(tmp_app_data)
    game, *_ = make_game()
    session.install_game(game)
    assert not (session.saves_root / game.id).exists()

    messages = await dispatch(
        make_registry(), HandlerContext(session=session), MessageType.c2s_app_exit
    )

    # The reply is an empty patch — an acknowledgement, not state.
    assert types_of(messages) == [MessageType.s2c_state_patch]
    assert messages[0][1].ops == []
    # The commit really happened, and the save reloads.
    assert (session.saves_root / game.id / "game.json").is_file()
    assert save_store.load_save(game.id, root=session.saves_root).id == game.id


@pytest.mark.asyncio
async def test_app_exit_is_a_no_op_with_no_game_installed(tmp_app_data: Path) -> None:
    """Quitting from the title screen must not raise or write a save."""
    session = make_session(tmp_app_data)
    assert session.game is None

    messages = await dispatch(
        make_registry(), HandlerContext(session=session), MessageType.c2s_app_exit
    )

    assert types_of(messages) == [MessageType.s2c_state_patch]
    assert list(session.saves_root.iterdir()) == []
