"""End-to-end smoke tests for the ``start.ps1`` pipeline.

These spawn real subprocesses and connect to local ports. They are slow
(several seconds each) and so live outside the offline-test gate. Run
with::

    pwsh tasks.ps1 smoke
    # or, equivalently:
    backend\\.venv\\Scripts\\python.exe -m pytest tests\\smoke

Each test cleans up its own subprocess on exit; if it doesn't, port
5173 (Vite) or the chosen WS port can stay bound until the next reboot.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest


def _pwsh() -> str:
    return shutil.which("pwsh") or shutil.which("powershell") or "powershell"


def _wait_for_port(port: int, *, timeout: float) -> bool:
    """Poll until something accepts a TCP connection on ``port``.

    Tries both IPv4 and IPv6 loopback because Vite binds to ``[::1]``
    on Windows by default, which an IPv4-only probe never sees.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for host in ("127.0.0.1", "::1"):
            try:
                family = socket.AF_INET6 if ":" in host else socket.AF_INET
                with socket.socket(family, socket.SOCK_STREAM) as sock:
                    sock.settimeout(0.5)
                    sock.connect((host, port))
                    return True
            except (ConnectionRefusedError, OSError):
                continue
        time.sleep(0.25)
    return False


def _read_until(stream, prefix: str, *, timeout: float) -> str | None:
    """Block-read ``stream`` line-by-line until a line starts with ``prefix``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = stream.readline()
        if not line:
            time.sleep(0.05)
            continue
        decoded = line.decode("utf-8", errors="replace").rstrip()
        if decoded.startswith(prefix):
            return decoded
    return None


def _run_pwsh(
    script: Path,
    args: list[str],
    *,
    cwd: Path,
    timeout: float = 600.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_pwsh(), "-NoProfile", "-NonInteractive", "-File", str(script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# ----------------------------- Phase 1: Setup --------------------------------


def test_setup_completes_and_installs_lucidium(start_script: Path, repo_root: Path) -> None:
    """``start.ps1 -Setup`` must finish successfully and leave a venv with
    the ``lucidium`` package importable."""
    result = _run_pwsh(start_script, ["-Setup", "-SkipCodegen"], cwd=repo_root)
    assert result.returncode == 0, (
        f"start.ps1 -Setup failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    venv_python = repo_root / "backend" / ".venv" / "Scripts" / "python.exe"
    assert venv_python.exists(), f"venv python missing at {venv_python}"

    probe = subprocess.run(
        [str(venv_python), "-c", "import lucidium, pydantic; print(lucidium.__version__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, (
        f"venv cannot import lucidium/pydantic\n"
        f"stdout:\n{probe.stdout}\nstderr:\n{probe.stderr}"
    )
    assert probe.stdout.strip() != ""


# ----------------------------- Phase 2: Codegen ------------------------------


def test_committed_schemas_match_a_fresh_export(
    venv_python: Path, repo_root: Path, tmp_path: Path
) -> None:
    """``shared-schemas/`` must be byte-identical to what the exporter
    produces right now.

    The previous version of this test ran the exporter straight into
    ``repo_root/shared-schemas`` and then asserted the directory was
    non-empty — it regenerated the very artifacts it inspected, so drift
    was undetectable by construction (it would even have passed on a
    checkout with no schemas committed at all), and it dirtied the
    working tree on every run. Here the export goes to ``tmp_path`` and
    the committed tree is only ever read.
    """
    committed_dir = repo_root / "shared-schemas"
    assert committed_dir.is_dir(), (
        f"{committed_dir} is missing — the committed schemas are the contract "
        "under test, not something this test may generate"
    )

    export_dir = tmp_path / "shared-schemas"
    contract_dir = tmp_path / "contract-schemas"

    env = {
        **os.environ,
        "LUCIDIUM_SCHEMA_OUT_DIR": str(export_dir),
        "LUCIDIUM_CONTRACT_SCHEMA_OUT_DIR": str(contract_dir),
    }
    result = subprocess.run(
        [str(venv_python), str(repo_root / "scripts" / "codegen" / "export-schemas.py")],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, (
        f"schema export failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "wrote" in result.stdout

    fresh = {p.name: p for p in export_dir.glob("*.json")}
    assert len(fresh) >= 30, (
        f"expected at least 30 exported schema files, got {len(fresh)}; "
        "the exporter, not the committed tree, is broken"
    )
    committed = {p.name: p for p in committed_dir.glob("*.json")}

    problems: list[str] = []
    for name in sorted(set(fresh) | set(committed)):
        if name not in committed:
            problems.append(f"  + {name} (exported but not committed)")
            continue
        if name not in fresh:
            problems.append(f"  - {name} (committed but no longer exported)")
            continue
        fresh_text = fresh[name].read_text(encoding="utf-8")
        committed_text = committed[name].read_text(encoding="utf-8")
        if fresh_text != committed_text:
            diff = "\n".join(
                difflib.unified_diff(
                    committed_text.splitlines(),
                    fresh_text.splitlines(),
                    fromfile=f"committed/{name}",
                    tofile=f"exported/{name}",
                    lineterm="",
                )
            )
            problems.append(f"  ~ {name} differs:\n{textwrap.indent(diff, '      ')}")

    assert not problems, (
        "shared-schemas/ is out of date with the Pydantic models. Run\n"
        "    backend\\.venv\\Scripts\\python.exe scripts\\codegen\\export-schemas.py\n"
        "and commit the result.\n" + "\n".join(problems)
    )


# ----------------------------- Phase 3: Backend ------------------------------


@contextlib.contextmanager
def _spawn(args: list[str], **kwargs) -> Iterator[subprocess.Popen]:
    """Spawn a subprocess and ensure its tree is killed on exit (Windows)."""
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    proc = subprocess.Popen(args, creationflags=flags, **kwargs)
    try:
        yield proc
    finally:
        if proc.poll() is None:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            else:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def test_backend_serves_websocket_handshake(venv_python: Path, repo_root: Path) -> None:
    """Spawn the backend directly (the same way start.ps1 -Backend does)
    and complete a real ``c2s/hello`` -> ``s2c/hello`` round trip."""
    args = [str(venv_python), "-m", "lucidium.app"]
    with _spawn(args, cwd=str(repo_root / "backend"), stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        assert proc.stdout is not None
        # The backend announces the per-launch bearer token immediately
        # BEFORE the port (see app.py), and the handshake is rejected
        # with a bare 401 without it — so both lines must be read, in
        # that order.
        token_line = _read_until(proc.stdout, "LUCIDIUM_WS_TOKEN=", timeout=20.0)
        assert token_line is not None, (
            "backend did not announce its auth token on stdout within 20 s "
            f"(stderr tail: {_read_stderr_tail(proc)})"
        )
        token = token_line.split("=", 1)[1]

        line = _read_until(proc.stdout, "LUCIDIUM_WS_PORT=", timeout=20.0)
        assert line is not None, (
            "backend did not announce its port on stdout within 20 s "
            f"(stderr tail: {_read_stderr_tail(proc)})"
        )
        port = int(line.split("=", 1)[1])
        assert _wait_for_port(port, timeout=10.0), f"backend did not listen on {port}"

        result = asyncio.run(_hello_round_trip(port, token))

    assert result["type"] == "s2c/hello"
    payload = result["payload"]
    assert payload["protocol_version"] == 1
    assert isinstance(payload["has_save"], bool)


async def _hello_round_trip(port: int, token: str) -> dict:
    import websockets

    # ``?token=`` on the path rather than a header: that is the query
    # form the server accepts from browser-side clients, and it is what
    # the renderer uses.
    url = f"ws://127.0.0.1:{port}?token={token}"
    async with websockets.connect(url) as conn:
        envelope = {"type": "c2s/hello", "payload": {"protocol_version": 1}, "protocol_version": 1}
        await conn.send(json.dumps(envelope))
        raw = await asyncio.wait_for(conn.recv(), timeout=5.0)
        return json.loads(raw)


def _read_stderr_tail(proc: subprocess.Popen, *, lines: int = 10) -> str:
    if proc.stderr is None:
        return "(no stderr captured)"
    try:
        proc.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    chunks = []
    for _ in range(lines):
        line = proc.stderr.readline()
        if not line:
            break
        chunks.append(line.decode("utf-8", errors="replace"))
    return "".join(chunks)


# ----------------------------- Phase 4: Renderer -----------------------------


def test_renderer_serves_index_html(start_script: Path, repo_root: Path) -> None:
    """``start.ps1 -Renderer`` brings up Vite serving the renderer's HTML."""
    if not (repo_root / "frontend" / "node_modules").exists():
        pytest.skip("frontend/node_modules missing; run `.\\start.ps1 -Setup` first")

    args = [_pwsh(), "-NoProfile", "-NonInteractive", "-File", str(start_script), "-Renderer", "-NoSetup"]
    with _spawn(args, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        if not _wait_for_port(5173, timeout=60.0):
            stdout = _read_pipe_nonblocking(proc.stdout)
            stderr = _read_pipe_nonblocking(proc.stderr)
            pytest.fail(
                "vite did not bind to 5173 within 60 s\n"
                f"start.ps1 stdout:\n{stdout}\n\nstart.ps1 stderr:\n{stderr}"
            )

        # Vite advertises localhost; on Windows that's IPv6 by default.
        with urllib.request.urlopen("http://localhost:5173/", timeout=5.0) as response:
            assert response.status == 200
            body = response.read().decode("utf-8", errors="replace")
        assert "<title>Lucidium</title>" in body, "renderer HTML did not include the title tag"
        assert 'id="root"' in body


def _read_pipe_nonblocking(stream, *, max_bytes: int = 4096) -> str:
    """Best-effort read of whatever is currently buffered. Returns ''
    if the stream is unavailable or empty."""
    if stream is None:
        return ""
    try:
        # On Windows, stream is a regular pipe; we read what's available
        # by switching to a read with a tight timeout via os-level ops.
        import os
        try:
            data = os.read(stream.fileno(), max_bytes)
        except (OSError, ValueError):
            data = b""
        return data.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"(could not read stream: {exc})"


# ------------------------- Phase 5: Electron compile -------------------------


def test_electron_main_compiles(repo_root: Path) -> None:
    """``tsc -p tsconfig.electron.json`` must produce ``dist-electron/main.js``."""
    frontend = repo_root / "frontend"
    if not (frontend / "node_modules").exists():
        pytest.skip("frontend/node_modules missing; run `.\\start.ps1 -Setup` first")

    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if npx is None:
        pytest.skip("npx not on PATH")

    result = subprocess.run(
        [npx, "tsc", "-p", "tsconfig.electron.json"],
        cwd=str(frontend),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"tsc failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert (frontend / "dist-electron" / "main.js").exists()
    assert (frontend / "dist-electron" / "preload.js").exists()
