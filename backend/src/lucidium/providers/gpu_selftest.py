"""Out-of-process torch device self-test.

This is the single source of truth for the question "does THIS overlay's
torch actually load and compute on the GPU?". It exists because that
question can only be answered honestly in a CLEAN process:

  * torch can only be imported ONCE per process, and once imported its
    accelerator backends (CUDA / ROCm HSA runtime / DirectML) latch onto
    whatever was on ``sys.path`` at import time. The running app has
    *already* imported torch from the active overlay (first render), so we
    can't re-import a candidate overlay's torch in-process to test it.
  * Importing torch into the app's own process purely to self-test would
    drag a multi-hundred-MB native library into the address space of the
    server event loop — for a check we want to be cheap and side-effect
    free.

So the self-test ALWAYS runs in a freshly-spawned child interpreter with
the candidate overlay prepended to ``sys.path``. The child imports torch
from the overlay, runs Lucidium's own ``detect_compute_device()`` and a
512x512 matmul + a conv2d against a CPU reference (the two ops that make
up an SDXL render — if they're correct on the device, image-gen works),
and prints a single ``SMOKE_RESULT_JSON=<json>`` line. The parent parses
that line into a structured dict.

Two entry points:

  * :func:`run_selftest` — the PARENT side. Spawn the child, parse the
    result, never raise (failures land in the returned dict's ``error``).
  * :func:`_run_checker` — the CHILD side. Runnable as
    ``python -m lucidium.providers.gpu_selftest`` with the overlay dir in
    ``LUCIDIUM_GPU_SELFTEST_OVERLAY`` (or the legacy ``SMOKE_OVERLAY`` /
    ``LUCIDIUM_TORCH_OVERLAY``). ``scripts/amd_smoke_test.py`` delegates
    its checker branch here so there is only one copy of the device
    math.

The result dict keys (stable contract — the self-heal logic and the
smoke-test report both read them)::

    ok                 bool  -- overlay torch imported AND matmul AND conv ok
    stage              str   -- last stage reached (for crash localisation)
    torch_version      str
    torch_file         str   -- where torch was imported from
    torch_from_overlay bool  -- torch_file lives under the overlay dir
    device             str   -- Lucidium device sentinel (cuda/xpu/mps/dml/cpu)
    compute_device     str   -- the torch device arg actually used
    matmul_ok          bool
    conv_ok            bool
    error              str   -- present only on failure
    ... plus optional gpu_name / torch_hip / dml_device_count / etc.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

# Env var naming the overlay dir to test, read by the CHILD process. We
# accept the historical ``SMOKE_OVERLAY`` (the amd_smoke_test driver set
# it) and ``LUCIDIUM_TORCH_OVERLAY`` (the entry shim's dev/test override)
# as fallbacks so existing tooling keeps working, but the canonical name
# is the explicit one below.
SELFTEST_OVERLAY_ENV_VAR = "LUCIDIUM_GPU_SELFTEST_OVERLAY"
# Optional: a path to an SDXL .safetensors for a real 2-step render. Only
# the smoke-test driver sets this; the app's self-heal never does.
SELFTEST_CHECKPOINT_ENV_VAR = "SMOKE_CHECKPOINT"

# The line prefix the child prints its JSON result on. Kept identical to
# the smoke test's historical sentinel so any log scraping still matches.
RESULT_PREFIX = "SMOKE_RESULT_JSON="

# backend/src — prepended in the child so ``lucidium`` imports even when the
# child interpreter has no editable install (frozen / bare venv). Resolved
# from THIS module's location: providers/gpu_selftest.py -> .../src/.
_BACKEND_SRC = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# CHILD process: import overlay torch + exercise the device.
# Must NEVER import torch at module top level — this module is imported by
# the parent (and by app code transitively). Everything torch-touching is
# inside _run_checker(), AFTER the overlay is prepended to sys.path.
# ---------------------------------------------------------------------------
def _resolve_child_overlay() -> str | None:
    """Read the overlay-to-test path from the child's environment.

    Canonical var first, then the historical fallbacks so the existing
    smoke-test driver and the entry shim's dev override keep working.
    """
    for var in (
        SELFTEST_OVERLAY_ENV_VAR,
        "SMOKE_OVERLAY",
        "LUCIDIUM_TORCH_OVERLAY",
    ):
        val = os.environ.get(var)
        if val:
            return val
    return None


def _run_checker() -> int:
    """CHILD process body: prove the overlay torch loads + the GPU computes.

    Emits one ``SMOKE_RESULT_JSON=<json>`` line on stdout. NEVER raises out
    of here — failures are captured into the JSON so the parent reports
    them cleanly instead of seeing a bare non-zero exit with no detail.
    """
    result: dict[str, object] = {"ok": False, "stage": "start"}
    try:
        overlay = _resolve_child_overlay()
        if not overlay:
            raise RuntimeError(f"no overlay dir in env (set {SELFTEST_OVERLAY_ENV_VAR})")
        # Overlay FIRST so its torch shadows any torch already importable in
        # the venv; backend/src so ``lucidium`` resolves without an editable
        # install. Order mirrors the entry shim's sys.path prepend.
        sys.path.insert(0, overlay)
        sys.path.insert(1, str(_BACKEND_SRC))

        result["stage"] = "import-torch"
        import torch

        torch_file = getattr(torch, "__file__", "") or ""
        result["torch_version"] = getattr(torch, "__version__", "?")
        result["torch_file"] = torch_file
        result["torch_from_overlay"] = os.path.normcase(torch_file).startswith(
            os.path.normcase(os.path.abspath(overlay))
        )

        result["stage"] = "detect-device"
        from lucidium.providers.embedded_image_client import (
            detect_compute_device,
        )

        device, diag = detect_compute_device()
        result["device"] = device
        result["device_diagnostic"] = diag

        # Map Lucidium's device sentinel to a real torch device argument.
        # DirectML registers a "privateuseone" device exposed via the
        # separate torch_directml wheel; everything else is a plain string.
        if device == "dml":
            # torch_directml ships no type stubs and is only present on
            # Windows DirectML installs.
            import torch_directml  # type: ignore[import-not-found]

            dev: torch.device | str = torch_directml.device()
            result["dml_device_count"] = int(torch_directml.device_count())
            name = getattr(torch_directml, "device_name", None)
            if callable(name):
                try:
                    result["dml_device_name"] = name(0)
                except Exception:
                    pass
        elif device in ("cuda", "xpu", "mps"):
            dev = device
        else:
            dev = "cpu"
        result["compute_device"] = str(dev)

        # Backend identity details for the report.
        if device == "cuda":
            try:
                result["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
            # On a ROCm build torch.version.hip is set; on real CUDA, cuda.
            result["torch_hip"] = getattr(torch.version, "hip", None)
            result["torch_cuda"] = getattr(torch.version, "cuda", None)

        # --- Real GPU compute: matmul (core of attention / linear layers) ---
        result["stage"] = "matmul"
        torch.manual_seed(0)
        a_cpu = torch.randn(512, 512)
        b_cpu = torch.randn(512, 512)
        ref = a_cpu @ b_cpu

        t0 = time.perf_counter()
        a = a_cpu.to(dev)
        b = b_cpu.to(dev)
        out_dev = a @ b
        out = out_dev.to("cpu")
        if device == "cuda":
            torch.cuda.synchronize()
        result["matmul_elapsed_s"] = round(time.perf_counter() - t0, 4)
        mm_err = (out - ref).abs().max().item()
        result["matmul_max_abs_err"] = mm_err
        result["matmul_ok"] = bool(mm_err < 1e-2)

        # --- Real GPU compute: conv2d (core of the UNet / VAE) ---
        result["stage"] = "conv2d"
        import torch.nn.functional as F

        x = torch.randn(1, 3, 64, 64)
        w = torch.randn(8, 3, 3, 3)
        cref = F.conv2d(x, w)
        cout = F.conv2d(x.to(dev), w.to(dev)).to("cpu")
        conv_err = (cout - cref).abs().max().item()
        result["conv_max_abs_err"] = conv_err
        result["conv_ok"] = bool(conv_err < 1e-2)

        # --- Optional: a real tiny SDXL render if a checkpoint was given ---
        # Only the smoke-test driver sets SMOKE_CHECKPOINT; the app's
        # self-heal never does (it wants a fast, dependency-light check).
        ckpt = os.environ.get(SELFTEST_CHECKPOINT_ENV_VAR)
        if ckpt:
            result["stage"] = "render"
            render: dict[str, object] = {"checkpoint": ckpt}
            try:
                from diffusers import (
                    EulerAncestralDiscreteScheduler,
                    StableDiffusionXLPipeline,
                )

                dtype = torch.float16 if device != "cpu" else torch.float32
                pipe = StableDiffusionXLPipeline.from_single_file(ckpt, torch_dtype=dtype)
                # diffusers' ``from_config`` carries no annotations in the
                # installed package; the call is the documented API.
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(  # type: ignore[no-untyped-call]
                    pipe.scheduler.config
                )
                pipe = pipe.to(dev)
                r0 = time.perf_counter()
                image = pipe(
                    "a red cube on a wooden table",
                    num_inference_steps=2,
                    width=512,
                    height=512,
                ).images[0]
                if device == "cuda":
                    torch.cuda.synchronize()
                render["elapsed_s"] = round(time.perf_counter() - r0, 2)
                render["image_size"] = list(image.size)
                render["ok"] = True
            except Exception as exc:
                import traceback

                render["ok"] = False
                render["error"] = f"{type(exc).__name__}: {exc}"
                render["traceback"] = traceback.format_exc()
            result["render"] = render

        result["stage"] = "done"
        result["ok"] = bool(
            result.get("torch_from_overlay") and result.get("matmul_ok") and result.get("conv_ok")
        )
    except Exception as exc:
        import traceback

        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return 0 if result.get("ok") else 1


# ---------------------------------------------------------------------------
# PARENT process: spawn the child, parse the single JSON result line.
# ---------------------------------------------------------------------------
class _CompletedLike(Protocol):
    """The subset of :class:`subprocess.CompletedProcess` the parent reads.

    A Protocol (not the concrete class) because tests inject a fake runner;
    the attributes are still read defensively via ``getattr`` since a fake
    may omit them entirely.
    """

    @property
    def stdout(self) -> str: ...

    @property
    def stderr(self) -> str: ...

    @property
    def returncode(self) -> int: ...


# ``(argv, env, timeout) -> CompletedProcess-like`` — the injectable spawn seam.
SelftestRunner = Callable[[list[str], dict[str, str], float], _CompletedLike]


def run_selftest(
    overlay_dir: Path | str,
    *,
    python_executable: str | None = None,
    checkpoint: Path | str | None = None,
    timeout: float = 180.0,
    runner: SelftestRunner | None = None,
) -> dict[str, object]:
    """Run the device self-test against ``overlay_dir`` IN A SUBPROCESS.

    NEVER imports torch in the caller's process (see module docstring for
    why that's mandatory). Spawns a child interpreter running
    ``python -m lucidium.providers.gpu_selftest`` with the overlay dir in
    ``LUCIDIUM_GPU_SELFTEST_OVERLAY``; parses the child's single
    ``SMOKE_RESULT_JSON=`` line into the structured dict documented at the
    top of this module.

    Always returns a dict — it NEVER raises. A spawn failure, a timeout, a
    crashed child, or a missing result line all come back as
    ``{"ok": False, "error": ...}`` so the self-heal caller can branch on
    ``ok`` without a try/except. (The whole point of this seam is to
    contain torch's import-time fragility; letting it raise here would
    defeat that.)

    ``runner`` is an injectable seam for tests: a callable
    ``(argv, env, timeout) -> CompletedProcess-like`` (needs ``.stdout`` /
    ``.stderr`` / ``.returncode``). Defaults to :func:`subprocess.run`. The
    unit tests pass a fake so no child is ever spawned and no torch moves.
    """
    overlay = str(overlay_dir)
    python = python_executable or sys.executable

    env = dict(os.environ)
    env[SELFTEST_OVERLAY_ENV_VAR] = overlay
    if checkpoint is not None:
        env[SELFTEST_CHECKPOINT_ENV_VAR] = str(checkpoint)

    argv = [python, "-m", "lucidium.providers.gpu_selftest"]

    def _default_runner(
        argv: list[str], env: dict[str, str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    run = runner or _default_runner

    try:
        proc = run(argv, env, timeout)
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "stage": "spawn",
            "error": f"gpu self-test timed out after {timeout}s for {overlay}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "stage": "spawn",
            "error": f"failed to spawn gpu self-test: {type(exc).__name__}: {exc}",
        }

    parsed = _parse_result(getattr(proc, "stdout", "") or "")
    if parsed is None:
        stderr = (getattr(proc, "stderr", "") or "").strip()
        rc = getattr(proc, "returncode", "?")
        return {
            "ok": False,
            "stage": "parse",
            "error": (
                f"gpu self-test produced no result line (exit {rc})"
                + (f"; stderr: {stderr[:2000]}" if stderr else "")
            ),
        }
    return parsed


def _parse_result(stdout: str) -> dict[str, object] | None:
    """Pull the LAST ``SMOKE_RESULT_JSON=`` line out of child stdout and
    JSON-decode it. Returns ``None`` when no parseable line is present.

    Scans LAST-first so a stray earlier print (a warning, a torch import
    banner) before the real result line can't shadow it.
    """
    for raw in reversed(stdout.splitlines()):
        if raw.startswith(RESULT_PREFIX):
            try:
                obj = json.loads(raw[len(RESULT_PREFIX) :])
            except json.JSONDecodeError:
                return None
            return obj if isinstance(obj, dict) else None
    return None


if __name__ == "__main__":
    raise SystemExit(_run_checker())
