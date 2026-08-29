"""Fire-and-forget tasks must survive garbage collection.

The event loop keeps only a weak reference to a pending task, so a
``create_task`` whose handle is dropped can be reaped mid-flight — the
coroutine then never completes and nothing is logged. The
safety-retcon path in ``orchestration.assets`` is the worst case: the
UI has already told the player the history was rewritten.
"""

from __future__ import annotations

import asyncio
import gc
import subprocess
from pathlib import Path

import pytest

from lucidium import background
from lucidium.orchestration import assets

BACKEND_SRC = Path(__file__).resolve().parents[2] / "src"


class _FakeSession:
    emit = None
    game = None


@pytest.mark.asyncio
async def test_safety_retcon_task_survives_gc(monkeypatch: pytest.MonkeyPatch) -> None:
    ran = asyncio.Event()

    async def _slow_retcon(_ctx, _instructions, push_undo=True):
        await asyncio.sleep(0.05)
        ran.set()

    from lucidium.api import handlers

    monkeypatch.setattr(handlers, "apply_global_retcon", _slow_retcon)
    monkeypatch.setattr(handlers, "HandlerContext", lambda **kw: object())

    before = background.pending_count()
    assets._safety_retcon_age_correction(_FakeSession(), character_id="c1", name="Mira", old_age=15)
    assert background.pending_count() == before + 1

    # Drop everything reachable, then reap. An unretained task dies here.
    gc.collect()

    await asyncio.wait_for(ran.wait(), timeout=2.0)
    # Let the done callback run so the set drains again.
    await asyncio.sleep(0)
    assert background.pending_count() == before


@pytest.mark.asyncio
async def test_spawn_logs_task_exception(caplog: pytest.LogCaptureFixture) -> None:
    async def _boom() -> None:
        raise RuntimeError("kaboom")

    task = background.spawn(_boom(), label="unit-test")
    with caplog.at_level("ERROR", logger="lucidium.background"):
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
    assert any("unit-test" in r.getMessage() for r in caplog.records)
    assert background.pending_count() == 0


def test_no_unretained_create_task_sites() -> None:
    """Every ``create_task`` in the backend is awaited or retained."""
    out = subprocess.run(
        ["git", "grep", "-n", "create_task(", "--", "src"],
        cwd=BACKEND_SRC.parent,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    offenders = []
    for line in out.splitlines():
        _path, _lineno, code = line.split(":", 2)
        stripped = code.strip()
        if stripped.startswith(("#", '"', "`", "*")):
            continue  # docstring / comment reference
        if "=" in stripped.split("create_task(")[0]:
            continue  # assigned to a name
        if stripped.startswith(("await ", "return ", "tasks[", "yield ")):
            continue
        offenders.append(line)
    assert not offenders, f"unretained create_task sites: {offenders}"
