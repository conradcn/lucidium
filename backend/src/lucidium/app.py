"""Process entrypoint.

Wires up logging, picks the provider implementations from settings or
the offline gate, and starts the WebSocket server. Reachable via the
``lucidium`` console script declared in ``pyproject.toml``.
"""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import sys
from pathlib import Path

from .api.handlers import HandlerRegistry, build_default_registry
from .api.ws_server import generate_auth_token, run_server
from .config import app_data_dir, is_offline


def _configure_parent_death_signal() -> None:
    """On Linux, ask the kernel to send us SIGTERM the moment our
    parent (the Electron main process) dies. Closes the
    "orphaned-backend" bug where killing or crashing the frontend
    leaves the Python sidecar running forever — still holding the
    SDXL pipeline in VRAM, the LLM HTTP client open, etc. The normal
    happy-path shutdown already works via the Electron-side SIGTERM
    in ``main.ts::shutdownBackend``; PDEATHSIG only fires on the
    UNHAPPY path (SIGKILL on the renderer, segfault, terminal close
    of the launcher).

    Linux only — ``PR_SET_PDEATHSIG`` is a Linux-specific prctl. On
    macOS there's no equivalent kernel facility (a launchd-style
    parent watcher would be the substitute); on Windows the Electron
    runtime tears the child down through its Job Object handling.
    Best-effort: a prctl failure (rare — only on a heavily-sandboxed
    libc) just means we fall back to the previous behaviour, which
    is what we'd ship if this function didn't exist.
    """
    if sys.platform != "linux":
        return
    try:
        import ctypes
        import signal as _signal

        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        rc = libc.prctl(PR_SET_PDEATHSIG, _signal.SIGTERM, 0, 0, 0)
        if rc != 0:
            # Don't log here — logging isn't configured yet at this
            # point in main() and a stray write would compete with
            # basicConfig's handler install. Failure is non-fatal.
            return
    except Exception:
        return


def _configure_amd_rocm_env() -> None:
    """Set ``HSA_OVERRIDE_GFX_VERSION`` for unofficially-supported AMD
    cards on Linux, BEFORE torch is ever imported.

    ROCm only ships kernels for a handful of officially-supported
    gfx targets. Consumer Navi 2x cards like the Radeon 6700 XT
    (gfx1031) are NOT on that list, so ROCm refuses to use them
    unless ``HSA_OVERRIDE_GFX_VERSION`` masks the arch to the nearest
    supported one — 10.3.0 for the gfx103x family. Without this, a
    ROCm torch reports ``torch.cuda.is_available() == False`` on these
    cards and image-gen silently falls back to CPU.

    This MUST run before the first torch import: ROCm reads the env
    var when its HSA runtime initialises (on torch import), so setting
    it afterwards has no effect. Hence it's the very first thing
    ``main`` does.

    Guards, narrowest-first:
      * Linux only — the override is a ROCm/HSA concept; on Windows
        AMD goes through DirectML and on macOS through MPS.
      * AMD GPU present only — reuse the embedded client's vendor
        probe (sysfs PCI ID on Linux, no subprocess) so we don't set
        a ROCm-specific var on an NVIDIA/Intel box.
      * Never clobber a user-provided value — a power user with a
        different card (e.g. gfx1030 needing 10.3.0 already, or a
        gfx900 needing 9.0.0) may have set it deliberately; respect it.
    """
    if sys.platform != "linux":
        return
    if "HSA_OVERRIDE_GFX_VERSION" in os.environ:
        return
    try:
        # Import here (not at module top) so the probe only runs on
        # the Linux path and a failure to import the providers package
        # can't take down startup before logging is even configured.
        from .providers.embedded_image_client import _has_amd_gpu
    except Exception:
        return
    try:
        if not _has_amd_gpu():
            return
    except Exception:
        return
    # gfx1031 (Radeon 6700/6750 XT) masks to the gfx103x baseline.
    # This is the value the broadest set of unofficial Navi 2x cards
    # need; users on other archs override it themselves (guarded above).
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"


def _configure_rembg_home() -> None:
    """Point ``U2NET_HOME`` at the bundled U2NET ONNX directory
    when we're running from a PyInstaller-frozen build, so rembg
    NEVER tries to download the ~170 MB model from
    ``github.com/danielgatis/rembg/releases`` at runtime.

    PyInstaller's bootloader sets ``sys.frozen`` and exposes the
    extraction root via ``sys._MEIPASS`` (one-file) or via the
    directory containing the exe (one-folder). We probe both
    layouts; the bundled tree puts the ONNX file at
    ``u2net_models/u2net.onnx`` so a direct path join finds it.
    Dev runs (no ``sys.frozen``) fall through to rembg's own
    ``~/.u2net`` default — that's the cache the spec also reads
    from at package time.
    """
    if not getattr(sys, "frozen", False):
        return
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "u2net_models")
    exe_dir = Path(sys.executable).resolve().parent
    candidates.append(exe_dir / "_internal" / "u2net_models")
    candidates.append(exe_dir / "u2net_models")
    for cand in candidates:
        if cand.is_dir() and (cand / "u2net.onnx").exists():
            os.environ["U2NET_HOME"] = str(cand)
            return


def _configure_logging() -> None:
    """Wire two handlers:

    * stderr — picked up by the Electron main process and tagged
      as ``[backend]`` in the renderer's console.
    * ``%APPDATA%/Lucidium/session.log`` — rotating file handler
      kept for one previous run, so a player who hits a bug can
      attach the log without the file growing unbounded across
      sessions. The directory is created if missing; if creation
      fails (permissions, read-only volume) we silently fall back
      to stderr-only — the engine still runs, the player just
      can't grab a file copy of the log on this launch.
    """
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)
    handlers.append(stream_handler)

    try:
        log_dir = app_data_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "session.log"
        # Rotate at 8 MB and keep one backup. A typical bug-report
        # session is well under 1 MB, but a long auto-research run
        # can drown a log; 8 MB is a generous floor that keeps
        # everything from the run available for inspection.
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=8 * 1024 * 1024,
            backupCount=1,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        handlers.append(file_handler)
    except Exception:
        # Don't print here; basicConfig hasn't been called yet so a
        # stray stream write would compete with the structured
        # handler we're about to install.
        pass

    logging.basicConfig(level=logging.INFO, handlers=handlers)


def _log_compute_device_diagnostics(_log: logging.Logger) -> None:
    """Print a clear startup banner about which device SDXL will run on.

    Most common silent failure: a user does ``pip install torch``
    without the PyTorch accelerator index URL. PyPI's
    default torch wheel on Windows is the CPU-only build, and the
    engine then runs SDXL on the CPU at ~minutes per image instead of
    the sub-second renders a real GPU produces. The pipeline still
    "works", so without an explicit log line nobody notices until
    they're staring at a blank stage for ten minutes.
    """
    try:
        from .providers.embedded_image_client import detect_compute_device
    except Exception:
        return
    device, diagnostic = detect_compute_device()
    if device == "cuda":
        _log.info("compute device: cuda (image gen will run on GPU)")
    elif device == "xpu":
        _log.info("compute device: xpu (Intel Arc / oneAPI accelerator)")
    elif device == "mps":
        _log.info("compute device: mps (Apple Silicon GPU)")
    else:
        # Loud — surface in stderr so it shows up in Electron's
        # backend log AND in any terminal the engine launched from.
        _log.warning(
            "compute device: CPU — image-gen renders will be very slow. %s",
            diagnostic,
        )


async def _auto_provision_torch_overlay(_log: logging.Logger) -> None:
    """Detect the GPU and, if a better torch flavor than the active one is
    available, download + activate it itself — and self-heal a broken or
    mismatched active GPU overlay. NON-BLOCKING with respect to startup:
    scheduled as a background task AFTER the server is serving, and its
    multi-GB download runs on a worker thread so the event loop (and every
    websocket message) stays responsive throughout.

    Why a background task and not part of ``main`` setup: a GPU torch
    download can take minutes / gigabytes. Doing it inline would stall the
    server bind and leave the renderer staring at a dead socket. Instead
    we let the server come up first (the app is fully usable on the seeded
    CPU overlay), then provision in the background and stream progress over
    the existing ``s2c/torch_overlay/{progress,status}`` channel.

    Gated behind ``ImageSettings.auto_gpu_provision`` (default on). Skips
    entirely (no thread, no probe cost beyond the cheap plan) when the
    active overlay already matches the recommendation and needs no repair.
    Strictly best-effort — any failure is caught inside the provision
    layer and the app keeps running on whatever overlay already works.
    """
    try:
        from .persistence.settings_store import load_settings
        from .providers import gpu_provision
        from .providers.gpu_selftest import run_selftest

        settings = load_settings()
        if not settings.image.auto_gpu_provision:
            _log.info("auto-provision: disabled by setting; skipping")
            return

        # Apply the settings-driven stall bound before anything can
        # download. The install gate this download runs inside pauses
        # image generation, so an unbounded read would refuse every
        # render for the rest of the process's life.
        from .providers import torch_overlay

        stall = settings.image.torch_download_stall_timeout_s
        if stall and stall > 0:
            torch_overlay.configure_download_timeouts(stall_timeout_s=float(stall))
        else:
            _log.warning(
                "auto-provision: ignoring non-positive "
                "torch_download_stall_timeout_s=%r; using default %.0fs",
                stall,
                torch_overlay.DEFAULT_STALL_TIMEOUT_S,
            )

        # Decide cheaply (network-free) what to do. A plain CPU box or an
        # already-correct-and-healthy GPU overlay short-circuits here with
        # no thread spawned for the install case.
        plan = gpu_provision.plan_auto_provision()
        if plan.action == "noop":
            _log.info("auto-provision: %s", plan.reason)
            return
        _log.info("auto-provision: plan=%s flavor=%s — %s", plan.action, plan.flavor, plan.reason)

        loop = asyncio.get_running_loop()

        # Progress + terminal status are streamed over the SAME websocket
        # messages a manual install uses, so the UI surfaces auto-provision
        # identically. Both callbacks fire from the worker thread, so they
        # hop back onto the loop thread before touching the broadcaster
        # (which enqueues onto each connection's outbox — loop-affine).
        from .api import ws_server
        from .api.messages import (
            MessageType,
            S2CTorchOverlayProgress,
            S2CTorchOverlayStatus,
        )

        flavor = plan.flavor or ""

        def _on_progress(done: int, total: int | None) -> None:
            loop.call_soon_threadsafe(
                ws_server.broadcast,
                MessageType.s2c_torch_overlay_progress,
                S2CTorchOverlayProgress(
                    flavor=flavor,
                    stage="downloading",
                    bytes_done=done,
                    bytes_total=total,
                ),
            )

        def _broadcast_status(snapshot: dict[str, object], activated: bool) -> None:
            # ``torch_overlay.status()`` is typed ``dict[str, object]``, so
            # each field is narrowed before it reaches the wire model.
            installed_raw = snapshot.get("installed", [])
            installed = (
                [str(flavor) for flavor in installed_raw] if isinstance(installed_raw, list) else []
            )
            active_raw = snapshot.get("active")
            loop.call_soon_threadsafe(
                ws_server.broadcast,
                MessageType.s2c_torch_overlay_status,
                S2CTorchOverlayStatus(
                    recommended=str(snapshot.get("recommended", "")),
                    installed=installed,
                    active=str(active_raw) if active_raw is not None else None,
                    runtime_dir=str(snapshot.get("runtime_dir", "")),
                    activated=activated,
                ),
            )

        # Run the (blocking, network + subprocess) execution on a worker
        # thread so the event loop never stalls. The provision layer is
        # best-effort and never raises, so a bare await is safe.
        outcome = await asyncio.to_thread(
            gpu_provision.run_auto_provision,
            plan,
            on_progress=_on_progress,
            broadcast_status=_broadcast_status,
            run_selftest=run_selftest,
        )
        _log.info(
            "auto-provision: done (action=%s activated=%s selftest_ok=%s repair=%s error=%s)",
            outcome.plan.action,
            outcome.activated,
            outcome.selftest_ok,
            outcome.repair,
            outcome.error,
        )
    except Exception:
        _log.warning("auto-provision: unexpected failure; ignoring", exc_info=True)


async def _serve(registry: HandlerRegistry, _log: logging.Logger) -> None:
    """Bring the websocket server up, then schedule GPU auto-provisioning
    as a background task once the server is accepting connections.

    We gate the provision task on the server's ``ready`` event so the
    renderer can connect (and start receiving progress broadcasts) before
    any multi-GB download begins — the download must not delay the bind.
    """
    ready = asyncio.Event()
    # Mint the per-launch bearer token here, alongside the port: both are
    # handed to the renderer over the same stdout channel (the server
    # prints ``LUCIDIUM_WS_TOKEN=`` immediately before
    # ``LUCIDIUM_WS_PORT=``), and both die with this process.
    auth_token = generate_auth_token()
    server_task = asyncio.create_task(
        run_server(registry=registry, ready=ready, auth_token=auth_token)
    )
    await ready.wait()
    # Fire-and-forget: keep a reference so the task isn't GC'd mid-flight,
    # but don't await it — startup is complete and the app is usable now.
    provision_task = asyncio.create_task(_auto_provision_torch_overlay(_log))
    # Same deal for the compute-device banner. It runs ``import torch``
    # (measured ~1.5 s), and doing that before the bind meant the
    # renderer's splash sat there for a second and a half waiting on a
    # log line. It's diagnostics — it can land after the port does.
    diagnostics_task = asyncio.create_task(asyncio.to_thread(_log_compute_device_diagnostics, _log))
    try:
        await server_task
    finally:
        provision_task.cancel()
        diagnostics_task.cancel()


def main() -> None:
    # VERY FIRST — bind our lifetime to the Electron parent before any
    # heavyweight import has a chance to allocate. If the renderer dies
    # while we're mid-import, the kernel will SIGTERM us at the next
    # syscall instead of leaving an orphaned multi-GB process behind.
    _configure_parent_death_signal()
    # THEN the ROCm env override — also before anything that could
    # import torch. ROCm reads HSA_OVERRIDE_GFX_VERSION at HSA-runtime
    # init (torch import time), so the override must be set before
    # the first torch import.
    _configure_amd_rocm_env()
    _configure_logging()
    _configure_rembg_home()
    _log = logging.getLogger("lucidium")
    _log.info("starting backend (offline=%s)", is_offline())
    # NOT here: ``_log_compute_device_diagnostics`` imports torch, which
    # costs ~1.5 s and used to be paid before the websocket port was
    # announced. ``_serve`` runs it on a worker thread once the server
    # is bound.
    registry = build_default_registry()
    try:
        asyncio.run(_serve(registry, _log))
    except KeyboardInterrupt:
        _log.info("interrupted; shutting down")


if __name__ == "__main__":
    main()
