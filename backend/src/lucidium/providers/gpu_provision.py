"""Automatic torch-overlay provisioning + self-heal.

The product wants torch to "just work": on first launch the bundled CPU
overlay is seeded (see ``torch_overlay.seed_bundled_overlay``) so the app
renders OFFLINE immediately, but a player with a real GPU shouldn't have
to find a button and consent to a multi-GB download — the app should
detect the GPU, fetch the right torch flavor itself, and repair a broken
or mismatched torch environment so image-gen always lands on the fastest
device that machine has.

This module is the policy layer over ``torch_overlay``'s mechanics. It
has two jobs:

  1. DECIDE (:func:`plan_auto_provision`) — compare the recommended
     flavor against what's active / installed and produce a plan: install
     a better flavor, self-heal a broken active one, or do nothing.
  2. EXECUTE (:func:`run_auto_provision`) — carry the plan out, wiring
     download progress + a terminal status onto the EXISTING
     ``s2c/torch_overlay/{progress,status}`` websocket channel (so the UI
     surfaces auto-provisioning with the exact code path manual installs
     already use), and self-healing if the active GPU overlay fails its
     self-test.

THE RELAUNCH SUBTLETY (why this can't take effect immediately): torch is
already on ``sys.path`` and (after first render) already imported in THIS
process. Activating a freshly-installed overlay only rewrites the
``active_overlay`` pointer the entry shim reads at the NEXT process
launch. So a successful auto-install does NOT swap torch under the running
app — it primes the next launch. We signal that to the UI with the same
``activated`` flag a manual install uses, and the UI prompts a relaunch.

CONSENT / OBSERVABILITY TRADEOFF: auto-downloading gigabytes without a
click is a deliberate product call (the owner wants zero-friction GPU
acceleration). We mitigate the surprise three ways: (a) it's gated behind
``ImageSettings.auto_gpu_provision`` (default on, but a single toggle
off); (b) every decision and byte-milestone is logged AND streamed over
the existing progress channel, so it's never silent; (c) it is strictly
best-effort — any network / disk / resolution failure is caught, logged,
and the app keeps running on whatever overlay already works (CPU at
worst). We NEVER leave the app unable to render because an auto-provision
attempt failed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from . import torch_overlay
from .torch_overlay import InstallResult, ProgressCallback

_log = logging.getLogger(__name__)


# A plan action. Kept as a small closed set so the decision table is
# exhaustively testable:
#   * ``noop``    — the active overlay already matches the recommendation
#                   (or no GPU is recommended); nothing to do.
#   * ``install`` — a better (GPU) flavor is recommended and isn't active;
#                   download+activate it (idempotent if already installed,
#                   in which case we just re-activate without downloading).
#   * ``selfheal``— the active overlay IS the recommended GPU flavor, but
#                   we still want to verify it actually computes; the
#                   executor self-tests it and repairs on failure.
PlanAction = Literal["noop", "install", "selfheal"]


@dataclass(frozen=True)
class ProvisionPlan:
    """What :func:`run_auto_provision` should do, decided up front from
    cheap (network-free) signals so the decision is unit-testable without
    spawning anything.

    ``action`` is the verb; ``flavor`` is the flavor it targets (the
    recommended one for ``install`` / ``selfheal``, ``None`` for ``noop``).
    ``reason`` is a human-readable WHY for the log line and the UI tooltip.
    ``already_installed`` is True when an ``install`` plan only needs to
    re-activate an on-disk overlay (no download) — lets the executor skip
    the progress theatre.
    """

    action: PlanAction
    flavor: str | None
    reason: str
    already_installed: bool = False


def plan_auto_provision(
    *,
    recommended: str | None = None,
    active: str | None = None,
    installed: list[str] | None = None,
) -> ProvisionPlan:
    """Decide the auto-provision action from the (recommended, active,
    installed) triple. PURE + network-free so every branch is testable.

    Signals default to the live ``torch_overlay`` probes; tests inject
    explicit values to sweep the decision table.

    Decision table (active × recommended → action):

      * recommended == cpu (no GPU detected)
            → NOOP. Never auto-download the CPU overlay — it's the seeded
              floor that already runs everywhere. A box with no GPU has
              nothing to upgrade to.
      * recommended is a GPU flavor, active == recommended
            → SELFHEAL. The right flavor is already active; we still
              verify it actually computes (a driver/runtime mismatch can
              leave a GPU torch silently falling back to CPU). The
              executor self-tests and repairs only on failure.
      * recommended is a GPU flavor, active != recommended,
        recommended already installed on disk
            → INSTALL (already_installed=True). No download — just flip
              the pointer so the next launch uses the better flavor.
      * recommended is a GPU flavor, active != recommended, NOT installed
            → INSTALL. Download + unpack + activate the recommended flavor.

    NOTE the asymmetry: a CPU recommendation never downgrades an active
    GPU overlay. If the player physically removed a GPU, the active GPU
    torch will simply fail its lazy import at render time and the existing
    diagnostics surface it; auto-provision's job is to ADD acceleration,
    not to tear it down. (We'd rather a stale-but-present GPU overlay than
    auto-clobbering it on a transient probe miss.)
    """
    if recommended is None:
        recommended = torch_overlay.recommend_flavor()
    if active is None:
        active = torch_overlay.active_flavor()
    if installed is None:
        installed = torch_overlay.installed_flavors()

    # No GPU detected: the seeded CPU floor is all there is. Nothing to do.
    if recommended == "cpu":
        return ProvisionPlan(
            action="noop",
            flavor=None,
            reason="no GPU detected; CPU overlay is the floor (nothing to provision)",
        )

    # Recommended GPU flavor is already the active one: verify it works.
    if active == recommended:
        return ProvisionPlan(
            action="selfheal",
            flavor=recommended,
            reason=(
                f"active overlay already matches recommendation ({recommended}); "
                "self-testing to confirm it computes on the GPU"
            ),
        )

    # Recommended GPU flavor differs from active. Install it — cheaply
    # (re-activate) if it's already unpacked on disk, else download it.
    if recommended in installed:
        return ProvisionPlan(
            action="install",
            flavor=recommended,
            reason=(
                f"recommended {recommended} already installed but not active "
                f"(active={active!r}); re-activating for next launch"
            ),
            already_installed=True,
        )
    return ProvisionPlan(
        action="install",
        flavor=recommended,
        reason=(
            f"GPU detected: recommended {recommended} not installed "
            f"(active={active!r}); downloading + activating"
        ),
    )


# A terminal-status broadcaster. Given the post-action overlay status
# snapshot plus whether a relaunch is needed, push an
# ``s2c/torch_overlay/status`` to connected clients. Injected by the
# caller (app/ws layer) so this module stays free of the api package.
StatusBroadcaster = Callable[[dict[str, object], bool], None]


class FlavorInstaller(Protocol):
    """The ``torch_overlay.install_flavor`` seam, injectable for tests."""

    def __call__(
        self,
        flavor: str,
        *,
        on_progress: ProgressCallback | None = ...,
        activate: bool = ...,
        force: bool = ...,
    ) -> InstallResult: ...


# ``overlay dir -> gpu_selftest.run_selftest`` result dict.
SelftestRunner = Callable[[Path], dict[str, object]]


@dataclass
class ProvisionOutcome:
    """What :func:`run_auto_provision` actually did. Returned for logging
    / tests; the UI learns the same facts via the broadcast callbacks."""

    plan: ProvisionPlan
    # Did we change the active pointer (=> a relaunch picks up new torch)?
    activated: bool = False
    # For self-heal: did the active overlay pass its device self-test?
    selftest_ok: bool | None = None
    # What repair (if any) the self-heal path took: ``reinstall`` (tried
    # reinstalling the recommended flavor) or ``None`` (no repair needed).
    # The legacy ``cpu_fallback`` branch was removed — we no longer flip
    # the active overlay to CPU on GPU failure, since that silently dumped
    # the player onto minutes-per-image CPU-SDXL renders.
    repair: str | None = None
    error: str | None = None


def run_auto_provision(
    plan: ProvisionPlan,
    *,
    on_progress: ProgressCallback | None = None,
    broadcast_status: StatusBroadcaster | None = None,
    install_flavor: FlavorInstaller | None = None,
    run_selftest: SelftestRunner | None = None,
) -> ProvisionOutcome:
    """Execute a :class:`ProvisionPlan`. BEST-EFFORT: never raises — any
    failure is caught, logged, recorded in the outcome, and the app keeps
    running on whatever overlay already works.

    ``on_progress`` / ``broadcast_status`` are wired by the caller to the
    EXISTING ``s2c/torch_overlay/{progress,status}`` websocket messages so
    auto-provisioning shows up in the UI through the same channel a manual
    install uses. ``install_flavor`` / ``run_selftest`` are injectable
    seams (default to the real ones); tests pass fakes so nothing is
    downloaded and no subprocess is spawned.

    INSTALL path: calls ``install_flavor(recommended, on_progress=...,
    activate=True)``. On success it broadcasts a terminal status carrying
    ``activated`` — the relaunch-needed signal the UI already consumes
    (torch can't hot-swap into the running process; see module docstring).

    SELFHEAL path: runs the device self-test on the ACTIVE overlay. If it
    passes, done. If it FAILS (import error, device==cpu while a GPU is
    recommended, or a compute mismatch), repair in order:
      1. Reinstall the recommended flavor (force=True) and re-self-test it.
      2. If that still fails, force the CPU overlay active so the app
         renders SOMETHING on the next launch rather than crashing on a
         broken GPU torch import.
    """
    _install = install_flavor or torch_overlay.install_flavor
    from .gpu_selftest import run_selftest as _real_selftest

    _selftest = run_selftest or _real_selftest

    outcome = ProvisionOutcome(plan=plan)

    if plan.action == "noop" or plan.flavor is None:
        _log.info("auto-provision: no-op (%s)", plan.reason)
        return outcome

    if plan.action == "install":
        return _run_install(plan, outcome, _install, on_progress, broadcast_status)

    if plan.action == "selfheal":
        return _run_selfheal(plan, outcome, _install, _selftest, on_progress, broadcast_status)

    return outcome


def _broadcast_terminal(broadcast_status: StatusBroadcaster | None, activated: bool) -> None:
    """Push a terminal ``s2c/torch_overlay/status`` snapshot. Best-effort:
    a broadcast failure must never turn a successful install into a
    reported failure."""
    if broadcast_status is None:
        return
    try:
        broadcast_status(torch_overlay.status(), activated)
    except Exception:
        _log.debug("auto-provision: status broadcast failed", exc_info=True)


def _run_install(
    plan: ProvisionPlan,
    outcome: ProvisionOutcome,
    install_flavor: FlavorInstaller,
    on_progress: ProgressCallback | None,
    broadcast_status: StatusBroadcaster | None,
) -> ProvisionOutcome:
    assert plan.flavor is not None
    _log.info(
        "auto-provision: installing %s (%s)%s",
        plan.flavor,
        plan.reason,
        " [already on disk; re-activating only]" if plan.already_installed else "",
    )
    try:
        result = install_flavor(plan.flavor, on_progress=on_progress, activate=True)
    except Exception as exc:
        # The CRITICAL invariant: a failed auto-download leaves the app
        # exactly as it was (still on the seeded CPU / prior overlay). Log
        # loudly so it's diagnosable, but never propagate.
        outcome.error = f"{type(exc).__name__}: {exc}"
        _log.warning(
            "auto-provision: install of %s failed (app keeps running on the current overlay): %s",
            plan.flavor,
            outcome.error,
        )
        return outcome

    outcome.activated = bool(getattr(result, "activated", False))
    _log.info(
        "auto-provision: %s ready (skipped=%s activated=%s); takes effect on next launch",
        plan.flavor,
        getattr(result, "skipped", "?"),
        outcome.activated,
    )
    _broadcast_terminal(broadcast_status, outcome.activated)
    return outcome


def _run_selfheal(
    plan: ProvisionPlan,
    outcome: ProvisionOutcome,
    install_flavor: FlavorInstaller,
    run_selftest: SelftestRunner,
    on_progress: ProgressCallback | None,
    broadcast_status: StatusBroadcaster | None,
) -> ProvisionOutcome:
    assert plan.flavor is not None
    overlay = torch_overlay.active_overlay_path()
    if overlay is None:
        # The pointer says this flavor is active but its dir vanished. Treat
        # exactly like a failed self-test: reinstall, else fall back to CPU.
        _log.warning(
            "auto-provision: active overlay %s has no on-disk dir; repairing",
            plan.flavor,
        )
        return _repair(plan, outcome, install_flavor, run_selftest, on_progress, broadcast_status)

    _log.info("auto-provision: self-testing active overlay %s", plan.flavor)
    res = run_selftest(overlay)
    outcome.selftest_ok = bool(res.get("ok"))
    if outcome.selftest_ok:
        _log.info(
            "auto-provision: active %s self-test PASSED (device=%s, torch=%s)",
            plan.flavor,
            res.get("device"),
            res.get("torch_version"),
        )
        return outcome

    _log.warning(
        "auto-provision: active %s self-test FAILED at stage=%s (device=%s, error=%s); repairing",
        plan.flavor,
        res.get("stage"),
        res.get("device"),
        res.get("error"),
    )
    return _repair(plan, outcome, install_flavor, run_selftest, on_progress, broadcast_status)


def _repair(
    plan: ProvisionPlan,
    outcome: ProvisionOutcome,
    install_flavor: FlavorInstaller,
    run_selftest: SelftestRunner,
    on_progress: ProgressCallback | None,
    broadcast_status: StatusBroadcaster | None,
) -> ProvisionOutcome:
    """Self-heal: reinstall the recommended GPU flavor and re-self-test.

    We intentionally do NOT fall back to the CPU overlay on failure.
    The old behaviour ("force CPU active so the app renders SOMETHING")
    just funnelled the player onto minutes-per-image CPU-SDXL renders
    with no signal that the GPU path had broken. The product call now is:
    surface the GPU failure honestly via the broadcast status, leave the
    active overlay alone, and let the GPU progress / status UI tell the
    player to fix the install (or wait for the next launch to retry)."""
    assert plan.flavor is not None

    try:
        _log.info("auto-provision: repair — reinstalling %s", plan.flavor)
        result = install_flavor(plan.flavor, on_progress=on_progress, activate=True, force=True)
        outcome.repair = "reinstall"
        outcome.activated = bool(getattr(result, "activated", False))
        overlay = torch_overlay.active_overlay_path()
        if overlay is not None:
            recheck = run_selftest(overlay)
            outcome.selftest_ok = bool(recheck.get("ok"))
            if outcome.selftest_ok:
                _log.info(
                    "auto-provision: reinstall of %s repaired the overlay",
                    plan.flavor,
                )
            else:
                _log.warning(
                    "auto-provision: reinstalled %s STILL fails self-test "
                    "(error=%s); leaving active overlay as-is — image-gen "
                    "will surface the failure via the GPU progress UI",
                    plan.flavor,
                    recheck.get("error"),
                )
        _broadcast_terminal(broadcast_status, outcome.activated)
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
        _log.warning(
            "auto-provision: reinstall of %s failed (%s); leaving active overlay as-is",
            plan.flavor,
            outcome.error,
        )
    return outcome
