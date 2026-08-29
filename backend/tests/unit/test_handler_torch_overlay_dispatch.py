"""``c2s/torch_overlay/status`` and ``c2s/torch_overlay/install`` through dispatch.

``status`` is a pure probe: it must be reachable on any machine, with or
without a GPU, without touching the network. ``install`` is the frozen
build's only way to acquire torch at all, so the shape under test is the
streaming contract — a ``resolving`` frame, zero or more ``downloading``
frames from a worker thread, then a terminal ``installed`` + ``status``
pair (or an error that surfaces as a ``LucidiumError``).

The install itself is monkeypatched: the real one fetches hundreds of
megabytes of wheels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lucidium.api.errors import ProviderUnreachableError, ProviderValidationError
from lucidium.api.handlers import HandlerContext
from lucidium.api.messages import MessageType
from lucidium.providers import torch_overlay

from .handler_harness import dispatch, make_registry, make_session, types_of


@pytest.fixture
def stub_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "recommended": "cuda",
        "installed": ["cpu"],
        "active": "cpu",
        "runtime_dir": str(tmp_path / "runtime"),
    }
    monkeypatch.setattr(torch_overlay, "status", lambda: dict(snapshot))
    return snapshot


# ---------------------------------------------------------------------------
# c2s/torch_overlay/status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_reports_the_probe_snapshot(
    tmp_app_data: Path, stub_status: dict[str, Any]
) -> None:
    session = make_session(tmp_app_data)
    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_torch_overlay_status,
        {},
    )

    assert types_of(messages) == [MessageType.s2c_torch_overlay_status]
    reply = messages[0][1]
    assert reply.recommended == "cuda"
    assert reply.installed == ["cpu"]
    assert reply.active == "cpu"
    assert reply.runtime_dir == stub_status["runtime_dir"]


@pytest.mark.asyncio
async def test_status_tolerates_a_machine_with_nothing_installed(
    tmp_app_data: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``active`` is ``None`` before any flavor is activated — the reply
    model has to accept that rather than coercing it to a string."""
    monkeypatch.setattr(
        torch_overlay,
        "status",
        lambda: {
            "recommended": "cpu",
            "installed": [],
            "active": None,
            "runtime_dir": str(tmp_path / "runtime"),
        },
    )
    session = make_session(tmp_app_data)
    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_torch_overlay_status,
        {},
    )

    reply = messages[0][1]
    assert reply.installed == []
    assert reply.active is None


@pytest.mark.asyncio
async def test_status_runs_against_the_real_probe_without_network(
    tmp_app_data: Path,
) -> None:
    """No stub: the live ``torch_overlay.status()`` must be safe to call
    on the test machine. The offline gate fails the test if it isn't."""
    session = make_session(tmp_app_data)
    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_torch_overlay_status,
        {},
    )

    reply = messages[0][1]
    assert reply.recommended
    assert isinstance(reply.installed, list)


# ---------------------------------------------------------------------------
# c2s/torch_overlay/install
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_streams_progress_then_a_terminal_status(
    tmp_app_data: Path,
    stub_status: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: list[tuple[str, bool]] = []

    def _fake_install(flavor: str, *, on_progress=None, activate: bool = True, **_kw: Any):
        seen.append((flavor, activate))
        if on_progress is not None:
            on_progress(10, 100)
            on_progress(100, 100)
        return torch_overlay.InstallResult(
            flavor=flavor,
            overlay_dir=tmp_path / "runtime" / flavor,
            skipped=False,
            activated=activate,
        )

    monkeypatch.setattr(torch_overlay, "install_flavor", _fake_install)

    session = make_session(tmp_app_data)
    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_torch_overlay_install,
        {"flavor": "cuda", "activate": True},
    )

    assert seen == [("cuda", True)]
    kinds = types_of(messages)
    assert kinds[0] == MessageType.s2c_torch_overlay_progress
    assert kinds[-1] == MessageType.s2c_torch_overlay_status
    stages = [m[1].stage for m in messages if m[0] == MessageType.s2c_torch_overlay_progress]
    assert stages[0] == "resolving"
    assert stages[-1] == "installed"
    assert "downloading" in stages
    # The byte counts from the worker thread made it onto the wire.
    downloads = [
        m[1]
        for m in messages
        if m[0] == MessageType.s2c_torch_overlay_progress and m[1].stage == "downloading"
    ]
    assert [(d.bytes_done, d.bytes_total) for d in downloads] == [(10, 100), (100, 100)]
    # Terminal status tells the UI a relaunch is needed.
    assert messages[-1][1].activated is True


@pytest.mark.asyncio
async def test_install_without_activate_reports_activated_false(
    tmp_app_data: Path,
    stub_status: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        torch_overlay,
        "install_flavor",
        lambda flavor, *, on_progress=None, activate=True, **_kw: torch_overlay.InstallResult(
            flavor=flavor,
            overlay_dir=tmp_path / flavor,
            skipped=True,
            activated=activate,
        ),
    )

    session = make_session(tmp_app_data)
    messages = await dispatch(
        make_registry(),
        HandlerContext(session=session),
        MessageType.c2s_torch_overlay_install,
        {"flavor": "cpu", "activate": False},
    )

    assert messages[-1][0] == MessageType.s2c_torch_overlay_status
    assert messages[-1][1].activated is False


@pytest.mark.asyncio
async def test_install_relays_a_transport_failure_as_provider_unreachable(
    tmp_app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: Any, **_kw: Any):
        raise OSError("wheel index unreachable")

    monkeypatch.setattr(torch_overlay, "install_flavor", _boom)

    session = make_session(tmp_app_data)
    with pytest.raises(ProviderUnreachableError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_torch_overlay_install,
            {"flavor": "cuda"},
        )


@pytest.mark.asyncio
async def test_install_preserves_a_lucidium_error_from_the_worker(
    tmp_app_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``LucidiumError`` already carries its own error code (e.g. "no
    compatible wheel for this flavor"), so the handler must re-raise it
    rather than flattening it into provider-unreachable."""

    def _boom(*_a: Any, **_kw: Any):
        raise ProviderValidationError("no compatible wheel for flavor 'rocm'")

    monkeypatch.setattr(torch_overlay, "install_flavor", _boom)

    session = make_session(tmp_app_data)
    with pytest.raises(ProviderValidationError):
        await dispatch(
            make_registry(),
            HandlerContext(session=session),
            MessageType.c2s_torch_overlay_install,
            {"flavor": "rocm"},
        )
