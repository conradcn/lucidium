#!/usr/bin/env python3
"""Lucidium GPU-acceleration smoke test (AMD / NVIDIA / Intel Arc).

WHAT THIS DOES, end to end, using the REAL product code paths:

  1. Detects the GPU vendor and the torch overlay "flavor" Lucidium
     would recommend for this machine (cuda / rocm / directml / xpu / cpu).
  2. Downloads + unpacks the matching torch build into an overlay dir
     via ``lucidium.providers.torch_overlay.install_flavor`` — the exact
     code the shipping app runs on first launch. This is a real network
     download (a few hundred MB for DirectML, ~2-3 GB for ROCm/CUDA).
  3. Launches a fresh subprocess with that overlay on ``sys.path`` (the
     way the packaged backend loads torch), imports torch FROM THE
     OVERLAY, runs Lucidium's ``detect_compute_device()``, and then does
     real GPU math — a 512x512 matmul and a conv2d — comparing the GPU
     result against a CPU reference. Those two ops are the building
     blocks of an SDXL render, so if they're correct and run on the GPU,
     image generation will work.
  4. (Optional) If you pass ``--checkpoint <path-to-SDXL.safetensors>``
     it also loads that checkpoint and does a tiny 2-step render to
     prove the full diffusers pipeline runs on the GPU.

It writes a single report file (``amd-smoke-report-<timestamp>.txt``)
next to itself. Send that file back to the developer.

Run it via the wrapper for your OS (they find the right Python):
    Windows:  scripts\amd-smoke.ps1
    Linux:    scripts/amd-smoke.sh
or directly with the project's venv Python:
    backend/.venv/Scripts/python.exe scripts/amd_smoke_test.py        (Windows)
    backend/.venv-linux/bin/python   scripts/amd_smoke_test.py        (Linux)

This script does NOT modify the app's own runtime dir by default — it
installs the overlay into ``scripts/amd-smoke-out/runtime`` so it can't
disturb a real install. Pass ``--use-app-runtime`` to populate the
real per-user runtime dir instead (so the app benefits afterwards).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# scripts/ lives at the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_SRC = REPO_ROOT / "backend" / "src"


# ---------------------------------------------------------------------------
# Subprocess entry: import torch FROM THE OVERLAY and exercise the device.
# This branch runs in a CHILD process the driver spawns; it must NOT import
# torch at module top level (the driver imports this same file). The actual
# device math now lives in ``lucidium.providers.gpu_selftest`` so the app's
# startup self-heal and this smoke test share ONE copy of the checker; this
# branch just delegates to it (after putting backend/src on sys.path so the
# import resolves from a non-editable checkout).
# ---------------------------------------------------------------------------
def run_checker() -> int:
    """Child process: delegate to the shared gpu_selftest checker.

    The shared module reads the overlay dir from ``SMOKE_OVERLAY`` (one of
    its accepted env vars) and an optional ``SMOKE_CHECKPOINT``, runs the
    same import + matmul + conv2d (+ optional render) sequence, and prints
    the ``SMOKE_RESULT_JSON=`` line this driver already parses.
    """
    if str(BACKEND_SRC) not in sys.path:
        sys.path.insert(0, str(BACKEND_SRC))
    from lucidium.providers.gpu_selftest import _run_checker

    return _run_checker()


# ---------------------------------------------------------------------------
# Driver: detect, install the overlay, spawn the checker, write the report.
# ---------------------------------------------------------------------------
class _Report:
    """Tees every line to stdout AND an in-memory buffer for the file."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.lines.append(msg)

    def section(self, title: str) -> None:
        self("")
        self("=" * 70)
        self(title)
        self("=" * 70)


def _human(nbytes: int) -> str:
    mb = nbytes / (1024 * 1024)
    return f"{mb:,.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description="Lucidium GPU smoke test")
    parser.add_argument(
        "--flavor",
        default=None,
        help="Override the auto-detected flavor (cuda/rocm/directml/xpu/cpu).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path to an SDXL .safetensors for a real 2-step render.",
    )
    parser.add_argument(
        "--use-app-runtime",
        action="store_true",
        help="Install into the real per-user runtime dir (the app will then "
        "use it). Default: an isolated dir under scripts/amd-smoke-out/.",
    )
    parser.add_argument(
        "--runtime-dir",
        default=None,
        help="Explicit runtime dir to install the overlay into.",
    )
    args = parser.parse_args()

    log = _Report()
    started = _dt.datetime.now()
    log.section("LUCIDIUM GPU SMOKE TEST")
    log(f"started:        {started.isoformat(timespec='seconds')}")
    log(f"script version: 1.0")

    # --- Section 1: environment -------------------------------------------
    log.section("1. ENVIRONMENT")
    log(f"os:             {platform.system()} {platform.release()} ({platform.version()})")
    log(f"platform:       {sys.platform}")
    log(f"machine:        {platform.machine()}")
    log(f"processor:      {platform.processor()}")
    log(f"python:         {sys.version.split()[0]} ({sys.executable})")

    # Make the product code importable from a non-editable checkout too.
    if str(BACKEND_SRC) not in sys.path:
        sys.path.insert(0, str(BACKEND_SRC))

    try:
        from lucidium.providers import torch_overlay
        from lucidium.providers.embedded_image_client import (
            _has_amd_gpu,
            _has_intel_arc_gpu,
            _has_nvidia_gpu,
        )
    except Exception as exc:  # noqa: BLE001
        log("")
        log(f"FATAL: could not import the Lucidium backend modules: {exc}")
        log("Make sure you ran this with the project's venv Python and that")
        log("the backend is set up (start.ps1 -Setup / start.sh).")
        _write_report(log, started, verdict="ERROR (import failed)")
        return 2

    # --- Section 2: GPU detection -----------------------------------------
    log.section("2. GPU DETECTION")
    nvidia, amd, arc = _has_nvidia_gpu(), _has_amd_gpu(), _has_intel_arc_gpu()
    log(f"NVIDIA present: {nvidia}")
    log(f"AMD present:    {amd}")
    log(f"Intel Arc:      {arc}")
    recommended = torch_overlay.recommend_flavor()
    log(f"recommended flavor: {recommended}")
    flavor = args.flavor or recommended
    if args.flavor:
        log(f"flavor OVERRIDE (--flavor): {flavor}")
    spec = torch_overlay.FLAVORS.get(flavor)
    log(f"flavor:         {flavor}  ({spec.description if spec else 'UNKNOWN'})")
    if flavor == "cpu":
        log("")
        log("NOTE: no discrete GPU was detected, so this would run CPU-only.")
        log("If you DO have a GPU, re-run with e.g. --flavor directml (Windows")
        log("AMD), --flavor rocm (Linux AMD), or --flavor cuda (NVIDIA).")

    # --- Section 3: install the overlay (REAL download) -------------------
    log.section("3. DOWNLOAD + UNPACK TORCH OVERLAY")
    if args.runtime_dir:
        rt = Path(args.runtime_dir).resolve()
    elif args.use_app_runtime:
        rt = None  # use torch_overlay's default per-user dir
    else:
        rt = (REPO_ROOT / "scripts" / "amd-smoke-out" / "runtime").resolve()

    if rt is not None:
        os.environ["LUCIDIUM_RUNTIME_DIR"] = str(rt)
        rt.mkdir(parents=True, exist_ok=True)
    log(f"runtime dir:    {torch_overlay.runtime_dir()}")
    log(f"index:          {spec.index_url if spec else '?'}")
    log(f"wheels:         {', '.join(spec.wheels) if spec else '?'}")
    log("downloading... (this can take several minutes / a few GB)")

    last_print = [0.0]

    def on_progress(done: int, total: int | None) -> None:
        now = time.time()
        if now - last_print[0] >= 1.0:  # throttle to ~1 line/sec
            tot = _human(total) if total else "?"
            log(f"  downloaded {_human(done)} / {tot}")
            last_print[0] = now

    install_ok = False
    overlay_dir = None
    try:
        t0 = time.perf_counter()
        res = torch_overlay.install_flavor(
            flavor, on_progress=on_progress, activate=True
        )
        dl_elapsed = time.perf_counter() - t0
        overlay_dir = res.overlay_dir
        log("")
        log(f"install:        {'reused cached' if res.skipped else 'downloaded'}")
        log(f"overlay dir:    {overlay_dir}")
        log(f"activated:      {res.activated}")
        log(f"elapsed:        {dl_elapsed:,.1f} s")
        install_ok = True
    except Exception as exc:  # noqa: BLE001
        import traceback

        log("")
        log(f"INSTALL FAILED: {type(exc).__name__}: {exc}")
        for line in traceback.format_exc().splitlines():
            log("  " + line)

    # --- Section 4: load overlay torch + exercise the GPU -----------------
    log.section("4. IMPORT OVERLAY TORCH + RUN GPU COMPUTE")
    checker: dict[str, object] = {}
    if install_ok and overlay_dir is not None:
        env = dict(os.environ)
        env["SMOKE_CHECKER"] = "1"
        env["SMOKE_OVERLAY"] = str(overlay_dir)
        if args.checkpoint:
            env["SMOKE_CHECKPOINT"] = str(Path(args.checkpoint).resolve())
        log(f"spawning checker subprocess with overlay on sys.path...")
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            env=env,
            capture_output=True,
            text=True,
        )
        for raw in proc.stdout.splitlines():
            if raw.startswith("SMOKE_RESULT_JSON="):
                try:
                    checker = json.loads(raw[len("SMOKE_RESULT_JSON=") :])
                except Exception:  # noqa: BLE001
                    pass
            else:
                log("  [checker] " + raw)
        if proc.stderr.strip():
            log("  [checker stderr]")
            for line in proc.stderr.splitlines():
                log("    " + line)

        if checker:
            log("")
            log(f"torch version:      {checker.get('torch_version')}")
            log(f"torch loaded from:  {checker.get('torch_file')}")
            log(f"torch FROM overlay: {checker.get('torch_from_overlay')}")
            log(f"detected device:    {checker.get('device')}  "
                f"(compute={checker.get('compute_device')})")
            if checker.get("device_diagnostic"):
                log(f"device diagnostic:  {checker.get('device_diagnostic')}")
            if checker.get("gpu_name"):
                log(f"GPU name:           {checker.get('gpu_name')}")
            if checker.get("torch_hip"):
                log(f"torch HIP version:  {checker.get('torch_hip')}")
            if checker.get("dml_device_count") is not None:
                log(f"DirectML devices:   {checker.get('dml_device_count')} "
                    f"{checker.get('dml_device_name', '')}")
            log(f"matmul ok:          {checker.get('matmul_ok')}  "
                f"(max err {checker.get('matmul_max_abs_err')}, "
                f"{checker.get('matmul_elapsed_s')} s)")
            log(f"conv2d ok:          {checker.get('conv_ok')}  "
                f"(max err {checker.get('conv_max_abs_err')})")
            render = checker.get("render")
            if isinstance(render, dict):
                log(f"render ok:          {render.get('ok')}  "
                    f"({render.get('elapsed_s', '?')} s, size {render.get('image_size')})")
                if not render.get("ok"):
                    log(f"  render error:     {render.get('error')}")
            if checker.get("error"):
                log("")
                log(f"CHECKER ERROR at stage '{checker.get('stage')}': "
                    f"{checker.get('error')}")
                for line in str(checker.get("traceback", "")).splitlines():
                    log("  " + line)
    else:
        log("skipped (overlay install did not succeed)")

    # --- Verdict ----------------------------------------------------------
    log.section("VERDICT")
    gpu_expected = flavor != "cpu"
    device_is_gpu = str(checker.get("device", "")) in ("cuda", "dml", "xpu", "mps")
    checks = {
        "overlay downloaded/unpacked": install_ok,
        "torch imported from overlay": bool(checker.get("torch_from_overlay")),
        "device is a GPU (not cpu fallback)": device_is_gpu if gpu_expected else True,
        "matmul correct on device": bool(checker.get("matmul_ok")),
        "conv2d correct on device": bool(checker.get("conv_ok")),
    }
    if args.checkpoint and isinstance(checker.get("render"), dict):
        checks["SDXL 2-step render"] = bool(checker["render"].get("ok"))  # type: ignore[index]

    for name, ok in checks.items():
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    overall = all(checks.values())
    verdict = "PASS" if overall else "FAIL"
    log("")
    log(f"OVERALL: {verdict}")
    if gpu_expected and not device_is_gpu:
        log("")
        log("The GPU flavor installed but torch still reports a CPU device.")
        log("This is the key failure mode to report — include this whole file.")

    _write_report(log, started, verdict=verdict)
    return 0 if overall else 1


def _write_report(log: _Report, started: _dt.datetime, *, verdict: str) -> None:
    stamp = started.strftime("%Y%m%d-%H%M%S")
    out = REPO_ROOT / "scripts" / f"amd-smoke-report-{stamp}.txt"
    log("")
    log(f"report written: {out}")
    log("")
    log(">>> Please send this report file back to the developer. <<<")
    try:
        out.write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"(could not write report file: {exc})", flush=True)


if __name__ == "__main__":
    if os.environ.get("SMOKE_CHECKER") == "1":
        raise SystemExit(run_checker())
    raise SystemExit(main())
