from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import InvalidStatus

from lucidium.api.handlers import build_default_registry
from lucidium.api.messages import (
    Envelope,
    MessageType,
)
from lucidium.api.ws_server import run_server

TEST_TOKEN = "test-token-not-a-real-secret"


async def _start_server(tmp_app_data: Path) -> tuple[asyncio.Task, int]:
    """Start a real WS server bound to an ephemeral port; return its task + port.

    The server requires the per-launch bearer token on every handshake;
    tests pass a fixed one in and append it to the URL via ``_ws_url``.
    """
    ready = asyncio.Event()
    captured: dict[str, int] = {}

    # Patch ``_announce_port`` so we capture the port instead of writing to stdout.
    import lucidium.api.ws_server as ws_module

    original = ws_module._announce_port

    def capture(port: int) -> None:
        captured["port"] = port

    ws_module._announce_port = capture
    try:
        task = asyncio.create_task(
            run_server(
                registry=build_default_registry(),
                port=0,
                ready=ready,
                auth_token=TEST_TOKEN,
            )
        )
        await asyncio.wait_for(ready.wait(), timeout=5.0)
        return task, captured["port"]
    finally:
        ws_module._announce_port = original


def _ws_url(port: int, token: str | None = TEST_TOKEN) -> str:
    base = f"ws://127.0.0.1:{port}"
    return f"{base}/?token={token}" if token is not None else base


@pytest.mark.asyncio
async def test_hello_handshake_round_trip(tmp_app_data: Path) -> None:
    task, port = await _start_server(tmp_app_data)
    try:
        async with ws_connect(_ws_url(port)) as connection:
            envelope = Envelope(type=MessageType.c2s_hello, payload={"protocol_version": 1})
            await connection.send(envelope.model_dump_json())
            raw = await asyncio.wait_for(connection.recv(), timeout=5.0)
            response = json.loads(raw)
            assert response["type"] == MessageType.s2c_hello.value
            assert response["payload"]["has_save"] is False
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, BaseException):
            pass


@pytest.mark.asyncio
async def test_invalid_envelope_yields_schema_error(tmp_app_data: Path) -> None:
    task, port = await _start_server(tmp_app_data)
    try:
        async with ws_connect(_ws_url(port)) as connection:
            await connection.send("{not even json")
            raw = await asyncio.wait_for(connection.recv(), timeout=5.0)
            response = json.loads(raw)
            assert response["type"] == MessageType.s2c_error.value
            assert response["payload"]["recoverable"] is True
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, BaseException):
            pass


@pytest.mark.asyncio
async def test_handshake_without_token_is_rejected(tmp_app_data: Path) -> None:
    """The socket is authenticated: no token, no connection.

    Anything else on the machine — another local process, a page in the
    player's browser — can find the port, but cannot produce the token,
    and so cannot reach a single handler.
    """
    task, port = await _start_server(tmp_app_data)
    try:
        with pytest.raises(InvalidStatus) as excinfo:
            async with ws_connect(_ws_url(port, token=None)):
                pass  # pragma: no cover -- the handshake must not succeed
        assert excinfo.value.response.status_code == 401

        with pytest.raises(InvalidStatus) as excinfo:
            async with ws_connect(_ws_url(port, token="wrong-token")):
                pass  # pragma: no cover -- the handshake must not succeed
        assert excinfo.value.response.status_code == 401
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, BaseException):
            pass


@pytest.mark.asyncio
async def test_handshake_from_a_web_origin_is_rejected(tmp_app_data: Path) -> None:
    """Even WITH a token, a browser origin we don't ship is refused.

    Defence in depth against a drive-by page: a browser can't forge
    ``Origin``, so an ordinary web page is turned away before the token
    check even matters.
    """
    task, port = await _start_server(tmp_app_data)
    try:
        with pytest.raises(InvalidStatus) as excinfo:
            async with ws_connect(
                _ws_url(port),
                origin="https://evil.example",  # type: ignore[arg-type]
            ):
                pass  # pragma: no cover -- the handshake must not succeed
        assert excinfo.value.response.status_code == 403
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, BaseException):
            pass
