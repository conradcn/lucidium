"""``_run_cancellable`` must not return while its worker thread still runs.

The embedded image client holds ``gpu_lock`` and the per-pipeline
inference lock across ``_run_cancellable``. If it unwinds while the
diffusion thread is still inside the pipeline — most dangerously in the
VAE decode after the last denoise step, where the abort callback can't
reach — the lock goes to the next render and two threads drive the same
pipeline's tensors at once. That is a native use-after-free in torch
(``c10.dll`` access violation), not a Python exception: the backend
process dies outright.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from lucidium.providers.embedded_image_client import _run_cancellable


def _uninterruptible(started: threading.Event, release: threading.Event, finished: threading.Event):
    """Stands in for the VAE decode: ignores ``abort`` entirely."""

    def _fn() -> str:
        started.set()
        release.wait(timeout=10)
        finished.set()
        return "png"

    return _fn


async def _wait_for(event: threading.Event) -> None:
    await asyncio.to_thread(event.wait, 10)


@pytest.mark.asyncio
async def test_single_cancel_waits_for_worker() -> None:
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    task = asyncio.create_task(_run_cancellable(_uninterruptible(started, release, finished)))
    await _wait_for(started)

    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done(), "unwound while the worker thread was still running"

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.asyncio
async def test_repeated_cancel_still_waits_for_worker() -> None:
    """Several teardown paths cancel the same session task (a superseded
    preview is cancelled, then the confirm/new-game sweep cancels it
    again because it isn't ``done()`` yet). The second cancel must not
    cut the wait short."""
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    task = asyncio.create_task(_run_cancellable(_uninterruptible(started, release, finished)))
    await _wait_for(started)

    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done(), "a repeated cancel released the lock mid-render"

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


@pytest.mark.asyncio
async def test_lock_is_not_handed_over_mid_render() -> None:
    """The failure as production sees it: a second render acquires the
    inference lock while the first render's thread is still running."""
    lock = asyncio.Lock()
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    overlap: list[bool] = []

    async def first() -> None:
        async with lock:
            await _run_cancellable(_uninterruptible(started, release, finished))

    async def second() -> None:
        async with lock:
            overlap.append(not finished.is_set())

    t1 = asyncio.create_task(first())
    await _wait_for(started)
    t2 = asyncio.create_task(second())
    await asyncio.sleep(0)

    t1.cancel()
    await asyncio.sleep(0.05)
    t1.cancel()
    await asyncio.sleep(0.2)
    release.set()

    await asyncio.gather(t1, t2, return_exceptions=True)
    assert overlap == [False], "second render entered the lock while the first thread was live"
