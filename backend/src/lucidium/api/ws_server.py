"""Single WebSocket endpoint.

Each connection gets its own ``Session`` (the engine is single-player and
single-window, so in production there is at most one). Validates every
inbound envelope against ``messages.py``, emits ``s2c/error`` for any
failure rather than dropping the connection.
"""

from __future__ import annotations

import asyncio
import hmac
import http
import json
import logging
import secrets
from collections.abc import AsyncIterator, Callable
from typing import Protocol
from urllib.parse import parse_qs, urlsplit

import websockets
from pydantic import BaseModel
from websockets.asyncio.server import ServerConnection, serve

# ``Request``/``Response`` are re-exported by ``websockets.asyncio.server``
# at runtime but are not in its ``__all__``, so importing them from there
# trips mypy's implicit-reexport check. ``websockets.http11`` is where they
# are defined and it does list them in ``__all__`` — it is also the module
# the websockets docs name when typing a ``process_request`` callback.
from websockets.http11 import Request, Response

from ..config import (
    PROTOCOL_VERSION,
    WS_ALLOWED_ORIGINS,
    WS_HOST,
    WS_PORT_ANNOUNCEMENT_PREFIX,
    WS_PORT_AUTO,
    WS_TOKEN_ANNOUNCEMENT_PREFIX,
)
from ..orchestration.session import OutboundEmit, Session
from .errors import LucidiumError, from_exception
from .handlers import HandlerContext, HandlerRegistry, build_default_registry
from .messages import (
    Envelope,
    MessageType,
    S2CError,
)

_log = logging.getLogger(__name__)

# Live connections, for SERVER-LEVEL broadcasts that aren't tied to one
# request's handler-yield stream. The per-connection ``outbox`` queue (set
# up in ``_serve_connection``) is the right place to push from a background
# task: writing to ``connection.send`` directly from another task would
# race the drain loop's writes onto the same socket. So a broadcaster
# enqueues onto every connection's outbox and the existing drain loop
# serialises it.
#
# In production this is at most one entry (single-player, single-window),
# but auto-provision starts before/independently of any connection, so the
# set may legitimately be empty when a broadcast fires — that's fine, the
# UI just learns the state on its next ``c2s/torch_overlay/status`` poll.
_OUTBOXES: set[asyncio.Queue[tuple[MessageType, BaseModel]]] = set()

# Outbox depth. Bounded on purpose: an unbounded queue turns a renderer
# that has stopped reading (hung, backgrounded, mid-crash) into unbounded
# process memory, and it makes the drop-on-backpressure contract that
# ``broadcast`` documents unreachable. Deep enough that a normal turn's
# burst of patches/notices never touches the ceiling.
OUTBOX_MAXSIZE = 1024

# Ceiling on concurrently-dispatching inbound messages. Dispatch is
# per-message (see ``inflight`` below) so the read loop keeps draining
# the socket, but without a cap a client could spawn a task per frame
# faster than handlers retire them.
MAX_INFLIGHT_DISPATCHES = 32


def _offer(
    outbox: asyncio.Queue[tuple[MessageType, BaseModel]],
    message_type: MessageType,
    payload: BaseModel,
) -> None:
    """Enqueue best-effort, dropping (loudly) when the outbox is full.

    A full outbox means the peer isn't reading; blocking here would stall
    whichever task is producing — including the read loop's dispatches —
    so the frame is dropped instead."""
    try:
        outbox.put_nowait((message_type, payload))
    except asyncio.QueueFull:
        _log.warning("outbox full; dropping %s", message_type.value)


def broadcast(message_type: MessageType, payload: BaseModel) -> None:
    """Enqueue a message onto every live connection's outbox.

    Thread-unsafe by design (asyncio.Queue.put_nowait must run on the loop
    thread); callers on a worker thread hop back via
    ``loop.call_soon_threadsafe``. Best-effort: a full/closed outbox just
    drops the frame (these are advisory progress/status notices)."""
    for outbox in list(_OUTBOXES):
        try:
            outbox.put_nowait((message_type, payload))
        except asyncio.QueueFull:
            pass


def _encode(message_type: MessageType, payload: BaseModel) -> str:
    """Serialise ``payload`` into an ``Envelope``-shaped JSON string.

    Deliberately NOT ``Envelope(payload=payload.model_dump(...)).
    model_dump_json()``: that walks the payload three times — dump to
    plain dicts, re-validate the whole thing through ``Envelope``, then
    serialise — which measured 2.3x a plain dump on a 1000-node
    ``s2c/state_full`` (the biggest message the protocol sends). The
    payload is already a validated model, so re-validating it buys
    nothing. Splicing the payload's own ``model_dump_json`` into the
    envelope literal serialises it exactly once.

    The envelope's own fields are a str-enum and an int, so they're
    formatted here rather than round-tripped through pydantic. The
    output must stay byte-identical to what ``Envelope`` would emit —
    ``tests/unit/test_ws_encode.py`` pins that.
    """
    return (
        f'{{"type":{json.dumps(message_type.value)},'
        f'"payload":{payload.model_dump_json()},'
        f'"protocol_version":{PROTOCOL_VERSION}}}'
    )


async def _send_error(connection: ServerConnection, exc: BaseException) -> None:
    structured = from_exception(exc)
    error_payload = S2CError(
        code=structured.code,
        message=structured.message,
        recoverable=structured.recoverable,
    )
    await connection.send(_encode(MessageType.s2c_error, error_payload))


class SessionFactory(Protocol):
    """Builds the per-connection :class:`Session`.

    Only ``emit`` is supplied by the server; every other ``Session``
    argument is optional, so ``Session`` itself satisfies this — as do
    the partially-applied factories the tests inject.
    """

    def __call__(self, *, emit: OutboundEmit) -> Session: ...


async def _serve_connection(
    connection: ServerConnection,
    registry: HandlerRegistry,
    *,
    session_factory: SessionFactory,
) -> None:
    # Background tasks (async asset generation, speculation completion)
    # push messages back to the renderer through ``session.emit``. We
    # bridge that to the WebSocket via an asyncio queue + a drain task
    # so handler-yielded messages and background-pushed messages
    # serialise cleanly onto the same connection.
    outbox: asyncio.Queue[tuple[MessageType, BaseModel]] = asyncio.Queue(maxsize=OUTBOX_MAXSIZE)

    def emit(message_type: MessageType, payload: BaseModel) -> None:
        _offer(outbox, message_type, payload)

    # Session construction touches the world: it loads settings off
    # disk, builds provider clients, opens the save store. Letting a
    # failure here propagate would reach ``run_server.handler``, which
    # only catches ``ConnectionClosed`` — so the socket dies with an
    # internal error code and the renderer just sees a reconnect loop
    # with no reason attached. Tell the client what broke first.
    try:
        session = session_factory(emit=emit)
    except Exception as exc:
        _log.exception("session construction failed")
        try:
            await _send_error(connection, exc)
        except websockets.ConnectionClosed:
            pass
        await connection.close()
        return
    ctx = HandlerContext(session=session)

    async def drain_outbox() -> None:
        while True:
            message_type, payload = await outbox.get()
            try:
                await connection.send(_encode(message_type, payload))
            except websockets.ConnectionClosed:
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                # Encoding runs inside this try, and ``model_dump_json``
                # raises for any non-JSON-native value that reached a
                # ``PatchOp.value`` (typed ``Any``). Letting that kill the
                # drain task silently loses EVERY subsequent server push
                # for a connection that is otherwise perfectly healthy —
                # patches, image-ready, notices, progress. Drop the one
                # bad frame and keep draining.
                _log.error("failed to send %s; dropping frame", message_type.value, exc_info=True)

    def enqueue_error(exc: BaseException) -> None:
        structured = from_exception(exc)
        _offer(
            outbox,
            MessageType.s2c_error,
            S2CError(
                code=structured.code,
                message=structured.message,
                recoverable=structured.recoverable,
            ),
        )

    # In-flight handler dispatches. Each inbound envelope runs in its
    # OWN task rather than inline in the read loop, because the read
    # loop is the only thing pulling frames off the socket: dispatching
    # inline meant a ``c2s/play/cancel`` sent during a 30 s generation
    # wasn't even READ until that generation finished, which makes a
    # cancel affordance impossible by construction. Handlers that
    # mutate shared state already serialise on ``session._play_lock``,
    # so concurrent dispatch doesn't reopen the races that lock closed.
    #
    # All handler output now goes through ``outbox`` instead of
    # ``connection.send``: with several dispatches live at once,
    # writing to the socket from each of them would interleave frames
    # mid-write. The drain task is the single writer.
    inflight: set[asyncio.Task[None]] = set()
    # FIFO, so messages that queue behind the cap still start in the
    # order they arrived on the wire.
    dispatch_slots = asyncio.Semaphore(MAX_INFLIGHT_DISPATCHES)

    async def run_dispatch(envelope: Envelope) -> None:
        try:
            async with dispatch_slots:
                outbound: AsyncIterator[tuple[MessageType, BaseModel]]
                outbound = registry.dispatch(envelope, ctx)
                async for message_type, payload in outbound:
                    _offer(outbox, message_type, payload)
        except asyncio.CancelledError:
            # Either a ``c2s/play/cancel`` or connection teardown. Both
            # are deliberate; the client isn't owed an error frame.
            raise
        except LucidiumError as exc:
            _log.info("handler error: %s", exc.message)
            enqueue_error(exc)
        except Exception as exc:
            _log.exception("unexpected handler failure")
            enqueue_error(exc)

    drain_task = asyncio.create_task(drain_outbox())

    def _report_drain_exit(task: asyncio.Task[None]) -> None:
        # Nothing awaits the drain task on the happy path, so an
        # exception escaping it would otherwise surface only as a
        # "Task exception was never retrieved" at GC time, long after
        # the pushes went missing.
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            _log.error("outbox drain task exited unexpectedly", exc_info=exc)

    drain_task.add_done_callback(_report_drain_exit)
    # Register for server-level broadcasts (e.g. auto-provision progress)
    # for the life of the connection.
    _OUTBOXES.add(outbox)
    try:
        async for raw in connection:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            try:
                envelope = Envelope.model_validate_json(raw)
            except Exception as exc:
                _log.info("rejected envelope: %s", exc)
                from .errors import SchemaError as _SchemaError

                await _send_error(connection, _SchemaError(f"invalid envelope: {exc}"))
                continue

            task = asyncio.create_task(run_dispatch(envelope))
            inflight.add(task)
            task.add_done_callback(inflight.discard)
    finally:
        _OUTBOXES.discard(outbox)
        # Stop the handlers still running for this (now dead) connection
        # BEFORE tearing the session down, so nothing is mid-mutation
        # when the session closes its provider clients.
        for task in list(inflight):
            task.cancel()
        if inflight:
            await asyncio.gather(*list(inflight), return_exceptions=True)
        # Cancel every background task the session spawned and close its
        # provider clients. Without this, closing the window mid-turn
        # left the LLM stream, speculation, summarizer, music and render
        # pump running against an orphaned session — spending real money
        # and GPU on output nobody would ever see.
        try:
            await session.aclose()
        except Exception:
            _log.warning("session teardown raised; continuing", exc_info=True)
        drain_task.cancel()
        try:
            await drain_task
        except (asyncio.CancelledError, websockets.ConnectionClosed):
            pass


def _announce_port(port: int) -> None:
    print(f"{WS_PORT_ANNOUNCEMENT_PREFIX}{port}", flush=True)


def _announce_token(token: str) -> None:
    print(f"{WS_TOKEN_ANNOUNCEMENT_PREFIX}{token}", flush=True)


def generate_auth_token() -> str:
    """Mint the per-launch bearer token.

    Fresh on every process start, so a token that leaks (a pasted log,
    a screenshot of the console) is worthless the moment the player
    restarts the app."""
    return secrets.token_urlsafe(32)


def _supplied_token(request: Request) -> str:
    """Pull the caller's token out of the handshake.

    Two accepted spellings, because the two kinds of client can't use
    the same one:

      * ``?token=<t>`` on the request path — the ONLY option for a
        browser/Electron renderer, since the WebSocket API gives no way
        to set request headers.
      * ``Authorization: Bearer <t>`` — for non-browser clients (tests,
        scripts) that would rather not put the secret in a URL.
    """
    try:
        query = parse_qs(urlsplit(request.path).query)
    except ValueError:
        query = {}
    values = query.get("token") or []
    if values and values[0]:
        return values[0]
    try:
        header = request.headers.get("Authorization") or ""
    except Exception:
        return ""
    prefix = "Bearer "
    return header[len(prefix) :] if header.startswith(prefix) else ""


def _make_process_request(
    expected_token: str,
) -> Callable[[ServerConnection, Request], Response | None]:
    """Build the handshake gate that requires the per-launch token.

    Runs BEFORE the connection is upgraded, so an unauthenticated peer
    never reaches ``_serve_connection`` — it gets a plain 401 and the
    socket closes. ``hmac.compare_digest`` keeps the comparison
    constant-time; the token is high-entropy enough that a timing
    oracle wouldn't help anyway, but there's no reason to leave one.
    """

    def process_request(connection: ServerConnection, request: Request) -> Response | None:
        if hmac.compare_digest(_supplied_token(request), expected_token):
            return None
        _log.warning("rejected websocket handshake: missing/invalid auth token")
        return connection.respond(http.HTTPStatus.UNAUTHORIZED, "unauthorized\n")

    return process_request


async def run_server(
    *,
    registry: HandlerRegistry | None = None,
    session_factory: SessionFactory | None = None,
    host: str = WS_HOST,
    port: int = WS_PORT_AUTO,
    ready: asyncio.Event | None = None,
    auth_token: str | None = None,
) -> None:
    """Bind, announce, and serve until cancelled.

    ``auth_token`` is the per-launch bearer token every client must
    present (see :func:`_make_process_request`). Callers that don't
    supply one get a freshly generated token, which — like the port —
    is announced on stdout for the parent process to pick up.
    """
    registry = registry or build_default_registry()
    session_factory = session_factory or Session
    token = auth_token or generate_auth_token()

    async def handler(connection: ServerConnection) -> None:
        try:
            await _serve_connection(connection, registry, session_factory=session_factory)
        except websockets.ConnectionClosed:
            return

    async with serve(
        handler,
        host,
        port,
        origins=list(WS_ALLOWED_ORIGINS),  # type: ignore[arg-type]
        process_request=_make_process_request(token),
    ) as server:
        _announce_token(token)
        bound_port = _resolve_bound_port(server, fallback=port)
        _announce_port(bound_port)
        if ready is not None:
            ready.set()
        await asyncio.Future()


def _resolve_bound_port(server: object, *, fallback: int) -> int:
    sockets = getattr(server, "sockets", None) or []
    for sock in sockets:
        try:
            return int(sock.getsockname()[1])
        except (OSError, IndexError, TypeError):
            continue
    return fallback
