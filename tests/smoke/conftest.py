"""Pytest configuration for the full-pipeline smoke tests.

These tests deliberately reach the network (localhost only) and run real
subprocesses. They intentionally do NOT inherit the offline-network gate
from ``backend/tests/conftest.py``; that gate is for unit/integration
tests of the backend in isolation.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = REPO_ROOT / "backend" / ".venv" / "Scripts" / "python.exe"
START_SCRIPT = REPO_ROOT / "start.ps1"


def _is_windows() -> bool:
    return sys.platform == "win32"


def _has_pwsh() -> bool:
    return shutil.which("pwsh") is not None or shutil.which("powershell") is not None


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def venv_python() -> Path:
    if not VENV_PYTHON.exists():
        pytest.skip(
            "backend/.venv not initialized; run `.\\start.ps1 -Setup` first."
        )
    return VENV_PYTHON


@pytest.fixture(scope="session")
def start_script() -> Path:
    if not START_SCRIPT.exists():
        pytest.skip(f"start.ps1 not found at {START_SCRIPT}")
    return START_SCRIPT


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip on platforms where ``start.ps1`` cannot run."""
    if _is_windows() and _has_pwsh():
        return
    skip_reason = (
        "smoke tests require Windows + PowerShell (pwsh or powershell)"
        if not _is_windows()
        else "neither pwsh nor powershell is on PATH"
    )
    skip_marker = pytest.mark.skip(reason=skip_reason)
    for item in items:
        item.add_marker(skip_marker)
