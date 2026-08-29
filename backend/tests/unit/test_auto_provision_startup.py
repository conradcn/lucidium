"""Startup wiring for GPU auto-provision: the setting gate + non-blocking
scheduling. NO real GPU / download / subprocess — everything is mocked.
"""

from __future__ import annotations

import asyncio

import pytest

from lucidium import app
from lucidium.domain.settings import ImageSettings, Settings
from lucidium.providers import gpu_provision


@pytest.fixture
def _log():
    import logging

    return logging.getLogger("test")


def test_setting_default_on() -> None:
    """The gate ships ON so a fresh install auto-provisions."""
    assert ImageSettings().auto_gpu_provision is True


def test_disabled_setting_skips_provision(monkeypatch, _log) -> None:
    """auto_gpu_provision=False -> no plan computed, no provision run."""
    settings = Settings()
    settings.image.auto_gpu_provision = False
    monkeypatch.setattr("lucidium.persistence.settings_store.load_settings", lambda: settings)

    called = {"plan": False, "run": False}
    monkeypatch.setattr(
        gpu_provision, "plan_auto_provision", lambda **k: called.__setitem__("plan", True)
    )
    monkeypatch.setattr(
        gpu_provision, "run_auto_provision", lambda *a, **k: called.__setitem__("run", True)
    )

    asyncio.run(app._auto_provision_torch_overlay(_log))
    assert called == {"plan": False, "run": False}


def test_noop_plan_skips_run(monkeypatch, _log) -> None:
    """Enabled, but the plan is a no-op (no GPU / already correct) -> the
    expensive run is never scheduled."""
    monkeypatch.setattr("lucidium.persistence.settings_store.load_settings", lambda: Settings())
    monkeypatch.setattr(
        gpu_provision,
        "plan_auto_provision",
        lambda **k: gpu_provision.ProvisionPlan(action="noop", flavor=None, reason="no gpu"),
    )
    ran = {"run": False}
    monkeypatch.setattr(
        gpu_provision, "run_auto_provision", lambda *a, **k: ran.__setitem__("run", True)
    )

    asyncio.run(app._auto_provision_torch_overlay(_log))
    assert ran["run"] is False


def test_install_plan_runs_provision(monkeypatch, _log) -> None:
    """Enabled + an install plan -> run_auto_provision is invoked with the
    plan and a progress + status callback wired in."""
    monkeypatch.setattr("lucidium.persistence.settings_store.load_settings", lambda: Settings())
    plan = gpu_provision.ProvisionPlan(action="install", flavor="cuda", reason="gpu")
    monkeypatch.setattr(gpu_provision, "plan_auto_provision", lambda **k: plan)

    seen: dict = {}

    def fake_run(p, *, on_progress=None, broadcast_status=None, run_selftest=None):
        seen["plan"] = p
        seen["has_progress"] = on_progress is not None
        seen["has_status"] = broadcast_status is not None
        return gpu_provision.ProvisionOutcome(plan=p, activated=True)

    monkeypatch.setattr(gpu_provision, "run_auto_provision", fake_run)

    asyncio.run(app._auto_provision_torch_overlay(_log))
    assert seen["plan"] is plan
    assert seen["has_progress"] is True
    assert seen["has_status"] is True


def test_unexpected_failure_never_raises(monkeypatch, _log) -> None:
    """The whole task is best-effort: a blowup in setup is swallowed."""

    def boom():
        raise RuntimeError("settings exploded")

    monkeypatch.setattr("lucidium.persistence.settings_store.load_settings", boom)
    # Must not raise.
    asyncio.run(app._auto_provision_torch_overlay(_log))


def test_broadcast_enqueues_to_outboxes() -> None:
    """ws_server.broadcast puts the message on every registered outbox and
    drops it cleanly when there are none."""
    from lucidium.api import ws_server
    from lucidium.api.messages import MessageType, S2CTorchOverlayProgress

    async def run() -> None:
        q: asyncio.Queue = asyncio.Queue()
        ws_server._OUTBOXES.add(q)
        try:
            ws_server.broadcast(
                MessageType.s2c_torch_overlay_progress,
                S2CTorchOverlayProgress(flavor="cuda", stage="downloading"),
            )
            mt, payload = q.get_nowait()
            assert mt == MessageType.s2c_torch_overlay_progress
            assert payload.flavor == "cuda"
        finally:
            ws_server._OUTBOXES.discard(q)
        # No registered outboxes -> no error.
        ws_server.broadcast(
            MessageType.s2c_torch_overlay_progress,
            S2CTorchOverlayProgress(flavor="cpu", stage="downloading"),
        )

    asyncio.run(run())
