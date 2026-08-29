"""Out-of-process device self-test — PARENT side, fully mocked subprocess.

NO real torch / subprocess / GPU. We inject a fake ``runner`` so
``run_selftest`` never actually spawns a child; we only verify the
parent's contract: it wires the overlay env var, parses the child's
``SMOKE_RESULT_JSON=`` line, and NEVER raises (every failure mode comes
back as ``{"ok": False, "error": ...}``).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from lucidium.providers import gpu_selftest


@dataclass
class _FakeProc:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


def _runner(proc: _FakeProc):
    """Build a runner that records the argv/env it was handed and returns
    the given fake process."""
    seen: dict = {}

    def run(argv, env, timeout):
        seen["argv"] = argv
        seen["env"] = env
        seen["timeout"] = timeout
        return proc

    return run, seen


def test_parses_result_line_and_sets_env() -> None:
    payload = {"ok": True, "device": "cuda", "matmul_ok": True, "conv_ok": True}
    proc = _FakeProc(
        stdout="some torch banner\n" + gpu_selftest.RESULT_PREFIX + json.dumps(payload)
    )
    run, seen = _runner(proc)

    res = gpu_selftest.run_selftest("/overlays/cuda", runner=run)

    assert res["ok"] is True
    assert res["device"] == "cuda"
    # Overlay dir is wired into the canonical env var for the child.
    assert seen["env"][gpu_selftest.SELFTEST_OVERLAY_ENV_VAR] == "/overlays/cuda"
    # Invoked as a module so the child runs _run_checker.
    assert seen["argv"][1:] == ["-m", "lucidium.providers.gpu_selftest"]


def test_takes_last_result_line() -> None:
    """A stray earlier result-ish line must not shadow the real last one."""
    bad = gpu_selftest.RESULT_PREFIX + json.dumps({"ok": False})
    good = gpu_selftest.RESULT_PREFIX + json.dumps({"ok": True, "device": "xpu"})
    proc = _FakeProc(stdout=bad + "\n" + good)
    run, _ = _runner(proc)
    res = gpu_selftest.run_selftest("/o", runner=run)
    assert res["ok"] is True
    assert res["device"] == "xpu"


def test_no_result_line_is_error_not_raise() -> None:
    proc = _FakeProc(stdout="nothing useful here", stderr="boom traceback", returncode=1)
    run, _ = _runner(proc)
    res = gpu_selftest.run_selftest("/o", runner=run)
    assert res["ok"] is False
    assert "no result line" in res["error"]
    assert "boom traceback" in res["error"]


def test_unparseable_json_is_error() -> None:
    proc = _FakeProc(stdout=gpu_selftest.RESULT_PREFIX + "{not json")
    run, _ = _runner(proc)
    res = gpu_selftest.run_selftest("/o", runner=run)
    assert res["ok"] is False
    assert "no result line" in res["error"]


def test_timeout_is_caught() -> None:
    def run(argv, env, timeout):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)

    res = gpu_selftest.run_selftest("/o", runner=run, timeout=5.0)
    assert res["ok"] is False
    assert "timed out" in res["error"]


def test_spawn_failure_is_caught() -> None:
    def run(argv, env, timeout):
        raise OSError("no python")

    res = gpu_selftest.run_selftest("/o", runner=run)
    assert res["ok"] is False
    assert "failed to spawn" in res["error"]


def test_checkpoint_wired_into_env() -> None:
    proc = _FakeProc(stdout=gpu_selftest.RESULT_PREFIX + json.dumps({"ok": True}))
    run, seen = _runner(proc)
    gpu_selftest.run_selftest("/o", checkpoint="/models/sdxl.safetensors", runner=run)
    assert seen["env"][gpu_selftest.SELFTEST_CHECKPOINT_ENV_VAR] == "/models/sdxl.safetensors"
