"""Pin: in a packaged build, rembg never reaches the internet
for its U2NET model. The startup hook ``_configure_rembg_home``
points ``U2NET_HOME`` at the bundled ``u2net_models`` directory,
which the spec populated by copying ``%USERPROFILE%/.u2net/
u2net.onnx`` at build time.

We can't reproduce a frozen-build environment under pytest, so
the test instead simulates one: stub ``sys.frozen``, point the
expected lookup directories at a tmp path that contains a
sentinel ``u2net.onnx``, and assert the env var lands.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_configure_rembg_home_dev_run_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``sys.frozen`` (i.e., dev run from the venv), the
    hook leaves ``U2NET_HOME`` alone — rembg falls back to its
    own ``~/.u2net`` cache the same way it always has."""
    from lucidium import app as app_module

    monkeypatch.delattr("sys.frozen", raising=False)
    monkeypatch.delenv("U2NET_HOME", raising=False)
    app_module._configure_rembg_home()
    assert os.environ.get("U2NET_HOME") in (None, "")


def test_configure_rembg_home_picks_meipass_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One-file PyInstaller layout: model lives under
    ``sys._MEIPASS/u2net_models/``. Hook must point
    ``U2NET_HOME`` at that directory."""
    from lucidium import app as app_module

    bundle = tmp_path / "meipass"
    (bundle / "u2net_models").mkdir(parents=True)
    (bundle / "u2net_models" / "u2net.onnx").write_bytes(b"fake")

    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys._MEIPASS", str(bundle), raising=False)
    monkeypatch.delenv("U2NET_HOME", raising=False)

    app_module._configure_rembg_home()
    assert os.environ.get("U2NET_HOME") == str(bundle / "u2net_models")


def test_configure_rembg_home_picks_internal_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One-folder PyInstaller layout (which we use): model lives
    under ``<exe-dir>/_internal/u2net_models/``. Hook must
    resolve to that path when ``sys._MEIPASS`` isn't set."""
    from lucidium import app as app_module

    exe_dir = tmp_path / "win-unpacked"
    (exe_dir / "_internal" / "u2net_models").mkdir(parents=True)
    (exe_dir / "_internal" / "u2net_models" / "u2net.onnx").write_bytes(b"fake")

    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.delattr("sys._MEIPASS", raising=False)
    monkeypatch.setattr(
        "sys.executable",
        str(exe_dir / "lucidium-backend.exe"),
        raising=False,
    )
    monkeypatch.delenv("U2NET_HOME", raising=False)

    app_module._configure_rembg_home()
    assert os.environ.get("U2NET_HOME") == str(
        exe_dir / "_internal" / "u2net_models",
    )


def test_configure_rembg_home_no_op_when_model_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the bundled directory exists but the .onnx isn't in
    it (build-time cache miss), don't override U2NET_HOME —
    rembg's own discovery logic will then surface the missing
    file as a clear download attempt the user can debug,
    rather than us silently pointing at an empty directory."""
    from lucidium import app as app_module

    exe_dir = tmp_path / "win-unpacked"
    (exe_dir / "_internal" / "u2net_models").mkdir(parents=True)
    # No u2net.onnx in there.

    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.delattr("sys._MEIPASS", raising=False)
    monkeypatch.setattr(
        "sys.executable",
        str(exe_dir / "lucidium-backend.exe"),
        raising=False,
    )
    monkeypatch.delenv("U2NET_HOME", raising=False)

    app_module._configure_rembg_home()
    assert os.environ.get("U2NET_HOME") in (None, "")
