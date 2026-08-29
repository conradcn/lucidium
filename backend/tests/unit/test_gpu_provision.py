"""Auto-provision decision table + self-heal fallback + the setting gate.

Pure-Python, NO real GPU / download / torch / subprocess. Every install
goes through a fake ``install_flavor`` and every self-test through a fake
``run_selftest``, so the policy logic is exercised in isolation:

  * ``plan_auto_provision`` across the (active, recommended, installed)
    matrix.
  * ``run_auto_provision`` install / self-heal happy + failure paths,
    including the reinstall-then-give-up chain (a broken GPU overlay is
    reported, never silently downgraded to CPU).
  * The best-effort guarantee: a raising install never propagates.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from lucidium.providers import gpu_provision
from lucidium.providers import torch_overlay as ov

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    """Stands in for torch_overlay.InstallResult."""

    flavor: str
    skipped: bool = False
    activated: bool = True


class _FakeInstaller:
    """Records install_flavor calls; returns a configurable result and can
    be made to raise for a chosen flavor."""

    def __init__(self, *, raise_on: set[str] | None = None, activated: bool = True):
        self.calls: list[tuple[str, dict]] = []
        self.raise_on = raise_on or set()
        self.activated = activated

    def __call__(self, flavor, *, on_progress=None, activate=True, force=False):
        self.calls.append(
            (
                flavor,
                {"activate": activate, "force": force, "has_progress": on_progress is not None},
            )
        )
        if flavor in self.raise_on:
            raise RuntimeError(f"boom installing {flavor}")
        return _FakeResult(flavor=flavor, activated=self.activated and activate)


def _selftest_returning(*results):
    """Build a fake run_selftest that yields the given dicts in order, then
    repeats the last one. Ignores the overlay arg."""
    seq = list(results)
    state = {"i": 0}

    def _fake(overlay_dir, **_kw):
        i = min(state["i"], len(seq) - 1)
        state["i"] += 1
        return seq[i]

    return _fake


# ---------------------------------------------------------------------------
# plan_auto_provision — the decision table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("recommended", "active", "installed", "expect_action", "expect_installed_flag"),
    [
        # No GPU detected -> never touch anything.
        ("cpu", None, [], "noop", False),
        ("cpu", "cpu", ["cpu"], "noop", False),
        ("cpu", "cuda", ["cuda", "cpu"], "noop", False),  # don't downgrade a GPU overlay
        # GPU recommended, nothing/CPU active, not installed -> download+install.
        ("cuda", None, [], "install", False),
        ("cuda", "cpu", ["cpu"], "install", False),
        ("directml", "cpu", ["cpu"], "install", False),
        # GPU recommended, already on disk but not active -> re-activate only.
        ("cuda", "cpu", ["cpu", "cuda"], "install", True),
        # GPU recommended AND already active -> self-heal (verify it computes).
        ("cuda", "cuda", ["cuda"], "selfheal", False),
        ("rocm", "rocm", ["rocm", "cpu"], "selfheal", False),
    ],
)
def test_plan_decision_table(
    monkeypatch, recommended, active, installed, expect_action, expect_installed_flag
) -> None:
    # ``active=None`` in the table means "no overlay is active", but that is
    # also plan_auto_provision's "not supplied -> probe the live system"
    # sentinel. Without this the row would read the DEVELOPER'S real GPU and
    # pass or fail depending on the host. Pin the probes to the table row.
    monkeypatch.setattr(ov, "recommend_flavor", lambda: recommended)
    monkeypatch.setattr(ov, "active_flavor", lambda: active)
    monkeypatch.setattr(ov, "installed_flavors", lambda: list(installed))

    plan = gpu_provision.plan_auto_provision(
        recommended=recommended, active=active, installed=installed
    )
    assert plan.action == expect_action
    if expect_action == "install":
        assert plan.flavor == recommended
        assert plan.already_installed is expect_installed_flag
    elif expect_action == "selfheal":
        assert plan.flavor == recommended
    else:
        assert plan.flavor is None


def test_plan_uses_live_probes_by_default(monkeypatch) -> None:
    """With no explicit triple, the plan pulls torch_overlay's live
    helpers — patch them and confirm they drive the result."""
    monkeypatch.setattr(ov, "recommend_flavor", lambda: "cuda")
    monkeypatch.setattr(ov, "active_flavor", lambda: "cpu")
    monkeypatch.setattr(ov, "installed_flavors", lambda: ["cpu"])
    plan = gpu_provision.plan_auto_provision()
    assert plan.action == "install"
    assert plan.flavor == "cuda"
    assert plan.already_installed is False


# ---------------------------------------------------------------------------
# run_auto_provision — install path
# ---------------------------------------------------------------------------


def test_run_noop_does_nothing() -> None:
    installer = _FakeInstaller()
    plan = gpu_provision.ProvisionPlan(action="noop", flavor=None, reason="x")
    outcome = gpu_provision.run_auto_provision(
        plan, install_flavor=installer, run_selftest=_selftest_returning({"ok": True})
    )
    assert installer.calls == []
    assert outcome.activated is False


def test_run_install_downloads_and_broadcasts() -> None:
    installer = _FakeInstaller(activated=True)
    statuses: list[tuple[dict, bool]] = []
    plan = gpu_provision.ProvisionPlan(action="install", flavor="cuda", reason="x")

    outcome = gpu_provision.run_auto_provision(
        plan,
        install_flavor=installer,
        run_selftest=_selftest_returning({"ok": True}),
        broadcast_status=lambda snap, act: statuses.append((snap, act)),
    )

    assert installer.calls[0][0] == "cuda"
    assert installer.calls[0][1]["activate"] is True
    assert installer.calls[0][1]["force"] is False
    assert outcome.activated is True
    # Terminal status broadcast carries the relaunch-needed (activated) flag.
    assert statuses and statuses[-1][1] is True


def test_run_install_failure_is_best_effort() -> None:
    """A raising install must NOT propagate — the app keeps running."""
    installer = _FakeInstaller(raise_on={"cuda"})
    plan = gpu_provision.ProvisionPlan(action="install", flavor="cuda", reason="x")
    outcome = gpu_provision.run_auto_provision(plan, install_flavor=installer)
    assert outcome.activated is False
    assert outcome.error is not None and "boom" in outcome.error


# ---------------------------------------------------------------------------
# run_auto_provision — self-heal path
# ---------------------------------------------------------------------------


def _active_overlay(monkeypatch, path="/fake/overlay"):
    """Make active_overlay_path() resolve to a fake path so self-heal runs
    the self-test instead of treating the overlay as missing."""
    from pathlib import Path

    monkeypatch.setattr(ov, "active_overlay_path", lambda: Path(path) if path else None)


def test_selfheal_passes_no_repair(monkeypatch) -> None:
    _active_overlay(monkeypatch)
    installer = _FakeInstaller()
    plan = gpu_provision.ProvisionPlan(action="selfheal", flavor="cuda", reason="x")
    outcome = gpu_provision.run_auto_provision(
        plan,
        install_flavor=installer,
        run_selftest=_selftest_returning({"ok": True, "device": "cuda"}),
    )
    assert outcome.selftest_ok is True
    assert outcome.repair is None
    assert installer.calls == []  # healthy -> nothing reinstalled


def test_selfheal_reinstall_repairs(monkeypatch) -> None:
    """Failing self-test -> force-reinstall -> re-test passes."""
    _active_overlay(monkeypatch)
    installer = _FakeInstaller(activated=True)
    plan = gpu_provision.ProvisionPlan(action="selfheal", flavor="cuda", reason="x")
    # First self-test fails; the post-reinstall re-test passes.
    selftest = _selftest_returning(
        {"ok": False, "stage": "import-torch", "error": "ImportError"},
        {"ok": True, "device": "cuda"},
    )
    outcome = gpu_provision.run_auto_provision(
        plan, install_flavor=installer, run_selftest=selftest
    )
    assert outcome.repair == "reinstall"
    assert outcome.selftest_ok is True
    # Reinstall used force=True.
    assert installer.calls[0] == ("cuda", {"activate": True, "force": True, "has_progress": False})
    # No CPU fallback needed.
    assert all(c[0] != "cpu" for c in installer.calls)


def test_selfheal_reinstall_still_broken_does_not_force_cpu(monkeypatch) -> None:
    """Self-test fails AND the reinstall still fails self-test -> the active
    GPU overlay is left alone. We deliberately do NOT force the CPU overlay
    active: that silently funnels the player onto minutes-per-image CPU
    renders. The failure is surfaced through the broadcast status instead."""
    _active_overlay(monkeypatch)
    installer = _FakeInstaller(activated=True)
    plan = gpu_provision.ProvisionPlan(action="selfheal", flavor="cuda", reason="x")
    selftest = _selftest_returning({"ok": False, "error": "device==cpu"})
    statuses: list[tuple[dict, bool]] = []
    outcome = gpu_provision.run_auto_provision(
        plan,
        install_flavor=installer,
        run_selftest=selftest,
        broadcast_status=lambda snap, act: statuses.append((snap, act)),
    )
    assert outcome.repair == "reinstall"
    assert outcome.selftest_ok is False
    # Exactly one reinstall of the GPU flavor, and no CPU downgrade.
    assert [c[0] for c in installer.calls] == ["cuda"]
    assert statuses  # a terminal status was broadcast so the UI can report it


def test_selfheal_reinstall_raises_is_best_effort(monkeypatch) -> None:
    """Reinstall of the GPU flavor RAISES -> swallowed, active overlay left
    as-is, and the error is reported on the outcome rather than propagated."""
    _active_overlay(monkeypatch)
    installer = _FakeInstaller(raise_on={"cuda"})
    plan = gpu_provision.ProvisionPlan(action="selfheal", flavor="cuda", reason="x")
    selftest = _selftest_returning({"ok": False, "error": "broken"})
    outcome = gpu_provision.run_auto_provision(
        plan, install_flavor=installer, run_selftest=selftest
    )
    assert outcome.repair is None
    assert outcome.error is not None and "boom" in outcome.error
    assert [c[0] for c in installer.calls] == ["cuda"]


def test_selfheal_missing_overlay_dir_repairs(monkeypatch) -> None:
    """Pointer names an active flavor whose dir vanished -> straight to
    repair (reinstall), no self-test of a non-existent dir. The reinstall
    recreates the dir, so the post-reinstall re-test runs and passes."""
    from pathlib import Path

    # active_overlay_path() returns None at first (dir gone), then a real
    # path once the reinstall has recreated it.
    state = {"i": 0}

    def _path():
        state["i"] += 1
        return None if state["i"] == 1 else Path("/fake/overlay")

    monkeypatch.setattr(ov, "active_overlay_path", _path)
    installer = _FakeInstaller(activated=True)
    plan = gpu_provision.ProvisionPlan(action="selfheal", flavor="cuda", reason="x")
    selftest = _selftest_returning({"ok": True, "device": "cuda"})
    outcome = gpu_provision.run_auto_provision(
        plan, install_flavor=installer, run_selftest=selftest
    )
    assert outcome.repair == "reinstall"
    assert outcome.selftest_ok is True
    assert installer.calls[0][0] == "cuda"
    assert installer.calls[0][1]["force"] is True
