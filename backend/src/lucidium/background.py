"""Retention for fire-and-forget asyncio tasks.

``loop.create_task`` hands the loop only a WEAK reference to the task.
A caller that drops the returned handle lets the garbage collector
reap the task mid-flight, and the coroutine simply never finishes —
silently, with no traceback anywhere. That bit us before (see the
``_dispose_image_client`` comment in ``orchestration/session.py``),
so every unawaited task now goes through :func:`spawn`, which parks a
strong reference in a module-level set until the task completes and
logs whatever the task raised instead of losing it at GC time.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

_log = logging.getLogger(__name__)

# Strong references to in-flight fire-and-forget tasks. Entries are
# discarded by the done callback, so this never grows without bound.
_TASKS: set[asyncio.Task[Any]] = set()


def spawn(
    coro: Coroutine[Any, Any, Any],
    *,
    label: str,
    loop: asyncio.AbstractEventLoop | None = None,
) -> asyncio.Task[Any]:
    """Schedule ``coro`` as a retained background task.

    ``label`` names the task in logs — it's the only breadcrumb left
    when the task fails, so make it specific.
    """
    task = (loop or asyncio.get_event_loop()).create_task(coro, name=label)
    _TASKS.add(task)
    task.add_done_callback(lambda t: _finish(t, label))
    return task


def _finish(task: asyncio.Task[Any], label: str) -> None:
    _TASKS.discard(task)
    if task.cancelled():
        _log.debug("background task %s cancelled", label)
        return
    exc = task.exception()
    if exc is not None:
        _log.error("background task %s failed", label, exc_info=exc)


def pending_count() -> int:
    """Number of retained tasks still in flight. For tests/diagnostics."""
    return len(_TASKS)
