"""Liveness guarantees for the WebSocket connection loop.

Two failure modes this pins down, both of which leave the socket open and
so look like a frozen/broken UI rather than a crash:

  * head-of-line blocking — a long-running handler (a multi-GB model
    download, a torch-overlay install) must not stop the read loop from
    picking up the NEXT inbound frame, or ``c2s/app_exit`` never lands and
    the player's only recourse is killing the process;
  * a dying drain task — one unserialisable payload must not take every
    subsequent server push down with it.

Plus the one failure mode that leaves the socket *shut*: a session that
can't even be constructed (a settings.json the schema rejects, a provider
client that won't open) must still explain itself with an ``s2c/error``
before the connection goes away, or the renderer only sees an opaque
close code and reconnect-loops forever.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from lucidium.api import ws_server
from lucidium.api.messages import (
    Envelope,
    MessageType,
    PatchOp,
    S2CNotice,
    S2CStatePatch,
)
from lucidium.api.ws_server import _offer, _serve_connection


class FakeConnection:
    """Async-iterable stand-in for a ``ServerConnection``.

    Inbound frames are pushed with :meth:`feed`; the iteration ends when
    :meth:`close` is called, which is what unwinds ``_serve_connection``.
    """

    def __init__(self) -> None:
        self._inbound: asyncio.Queue[str | None] = asyncio.Queue()
        self.sent: list[str] = []
        self.sent_event = asyncio.Event()

    def feed(self, envelope: Envelope) -> None:
        self._inbound.put_nowait(envelope.model_dump_json())

    def close(self) -> None:
        self._inbound.put_nowait(None)

    async def send(self, raw: str) -> None:
        self.sent.append(raw)
        self.sent_event.set()

    async def __aiter__(self):
        while True:
            raw = await self._inbound.get()
            if raw is None:
                return
            yield raw

    # Suppression rationale: the per-call ``timeout`` is load-bearing — tests tune
    # how long they wait for a frame, and it maps onto ``asyncio.wait_for`` on
    # the event itself rather than cancelling a surrounding scope.
    async def next_sent(self, *, timeout: float = 2.0) -> dict:  # noqa: ASYNC109
        """Await the next frame written to the socket, as parsed JSON."""
        seen = len(self.sent)
        while len(self.sent) == seen:
            self.sent_event.clear()
            if len(self.sent) > seen:
                break
            await asyncio.wait_for(self.sent_event.wait(), timeout=timeout)
        return json.loads(self.sent[seen])


class FakeSession:
    def __init__(self, *, emit) -> None:
        self.emit = emit

    async def aclose(self) -> None:
        return None


class _Registry:
    """Minimal ``HandlerRegistry`` shape: ``dispatch`` -> async iterator."""

    def __init__(self, dispatch) -> None:
        self._dispatch = dispatch

    def dispatch(self, envelope, ctx):
        return self._dispatch(envelope, ctx)


def _hello(version: int) -> Envelope:
    return Envelope(type=MessageType.c2s_hello, payload={"protocol_version": version})


async def _serve(connection: FakeConnection, registry: _Registry):
    sessions: list[FakeSession] = []

    def factory(*, emit):
        session = FakeSession(emit=emit)
        sessions.append(session)
        return session

    task = asyncio.create_task(_serve_connection(connection, registry, session_factory=factory))
    # Let the read loop and drain task start before the test drives them.
    await asyncio.sleep(0)
    return task, sessions


@pytest.mark.asyncio
async def test_slow_handler_does_not_delay_the_next_message() -> None:
    """A handler that blocks for a long time must not stall the read loop.

    This is the model-download case: the second frame is the one that
    carries ``c2s/app_exit``.
    """
    release = asyncio.Event()

    async def dispatch(envelope, ctx):
        if envelope.payload["protocol_version"] == 1:
            await release.wait()  # the "6.9 GB download"
            yield MessageType.s2c_notice, S2CNotice(title="slow", body="done")
        else:
            yield MessageType.s2c_notice, S2CNotice(title="fast", body="done")

    connection = FakeConnection()
    task, _ = await _serve(connection, _Registry(dispatch))
    try:
        connection.feed(_hello(1))  # slow
        connection.feed(_hello(2))  # fast — must not queue behind the slow one

        first = await connection.next_sent()
        assert first["payload"]["title"] == "fast"

        release.set()
        second = await connection.next_sent()
        assert second["payload"]["title"] == "slow"
    finally:
        release.set()
        connection.close()
        await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_unserialisable_payload_is_logged_and_draining_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One bad ``PatchOp.value`` must cost exactly one frame, not the session.

    ``PatchOp.value`` is typed ``Any``, so a non-JSON-native object gets
    all the way to ``model_dump_json`` inside the drain task's try block.
    """

    async def dispatch(envelope, ctx):
        return
        yield  # pragma: no cover -- makes this an async generator

    connection = FakeConnection()
    task, sessions = await _serve(connection, _Registry(dispatch))
    try:
        # Trigger session creation and grab its ``emit`` bridge.
        emit = sessions[0].emit

        with caplog.at_level(logging.ERROR, logger=ws_server.__name__):
            emit(
                MessageType.s2c_state_patch,
                S2CStatePatch(ops=[PatchOp(op="replace", path="/x", value=object())]),
            )
            emit(MessageType.s2c_notice, S2CNotice(title="after", body="still alive"))

            follow_up = await connection.next_sent()

        assert follow_up["payload"]["title"] == "after"
        assert any(
            MessageType.s2c_state_patch.value in record.getMessage()
            for record in caplog.records
            if record.levelno >= logging.ERROR
        ), caplog.text
    finally:
        connection.close()
        await asyncio.wait_for(task, timeout=2.0)


class ClosableConnection(FakeConnection):
    """``FakeConnection`` whose ``close`` is the awaitable server-side one.

    ``_serve_connection`` closes the socket itself on the
    session-construction failure path, so the fake needs the async
    spelling rather than the test-driver's synchronous one.
    """

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def close(self) -> None:  # type: ignore[override]
        self.closed = True
        self._inbound.put_nowait(None)


@pytest.mark.asyncio
async def test_session_construction_failure_sends_error_frame(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A raising ``session_factory`` owes the client an ``s2c/error``.

    Session construction reads ``settings.json``; a known key holding a
    value the schema rejects used to raise here, propagate past
    ``run_server.handler`` (which catches only ``ConnectionClosed``) and
    kill the socket silently — leaving the Settings screen, the only
    in-app fix, unreachable.
    """

    async def dispatch(envelope, ctx):  # pragma: no cover -- never reached
        return
        yield

    def factory(*, emit):
        raise RuntimeError("settings.json is unusable")

    connection = ClosableConnection()
    with caplog.at_level(logging.ERROR, logger=ws_server.__name__):
        await asyncio.wait_for(
            _serve_connection(connection, _Registry(dispatch), session_factory=factory),
            timeout=2.0,
        )

    assert len(connection.sent) == 1, connection.sent
    frame = json.loads(connection.sent[0])
    assert frame["type"] == MessageType.s2c_error.value
    assert connection.closed


def test_full_outbox_drops_instead_of_growing(caplog: pytest.LogCaptureFixture) -> None:
    """The documented backpressure drop is real, not aspirational.

    Before the queues were given a ``maxsize`` this was unreachable: an
    unbounded queue never raises ``QueueFull``, so a peer that stopped
    reading grew the process instead.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    notice = S2CNotice(title="t", body="b")

    with caplog.at_level(logging.WARNING, logger=ws_server.__name__):
        for _ in range(10):
            _offer(queue, MessageType.s2c_notice, notice)

    assert queue.qsize() == 2
    assert "outbox full" in caplog.text


def test_outbox_is_bounded() -> None:
    assert ws_server.OUTBOX_MAXSIZE > 0
    assert ws_server.MAX_INFLIGHT_DISPATCHES > 0
