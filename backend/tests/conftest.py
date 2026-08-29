"""Shared pytest fixtures and the offline-network gate.

Constitution IV (Testability): no test may reach the network. ``respx``
is enabled autouse to intercept every outbound HTTP call; any call that
is not explicitly mocked fails the test. Tests that need to simulate
provider behavior must do so by routing through ``RecordedLlmClient`` /
``RecordedImageClient``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import respx

from lucidium.config import APP_DATA_OVERRIDE_ENV_VAR


@pytest.fixture(autouse=True)
def _deny_network(request: pytest.FixtureRequest) -> Iterator[respx.MockRouter | None]:
    # Tests marked ``live`` opt out of the network gate — they
    # drive the real LLM endpoint and ComfyUI deliberately. Tests
    # marked ``embedded_live`` also opt out: the embedded client
    # loads a local checkpoint (no network) but rembg lazy-downloads
    # its U2Net background-removal model on first use, and the
    # SDXL Turbo bootstrap fetches from HuggingFace if the models
    # directory is empty. Every other test stays sealed against
    # accidental egress.
    if request.node.get_closest_marker("live") is not None:
        yield None
        return
    if request.node.get_closest_marker("embedded_live") is not None:
        yield None
        return
    # ``assert_all_mocked`` (respx's default) is the deny rule: any
    # request that does not match a route registered by the test raises
    # ``AllMockedAssertionError`` instead of leaving the process. It is
    # used in preference to a catch-all ``router.route()`` because respx
    # resolves routes in registration order — a catch-all installed here
    # would shadow every route a test tried to add afterwards, making
    # the handful of handlers that legitimately speak HTTP
    # (``settings/validate_api_key``, ``music/inventory``) untestable.
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        yield router


@pytest.fixture
def http_router(_deny_network: respx.MockRouter | None) -> respx.MockRouter:
    """The live offline-gate router, for tests that need to allow ONE
    specific outbound call. Registering a route here narrows the gate for
    that URL only; everything else still raises."""
    if _deny_network is None:  # pragma: no cover - live/embedded_live only
        pytest.skip("network gate is disabled for live tests")
    return _deny_network


@pytest.fixture
def tmp_app_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(APP_DATA_OVERRIDE_ENV_VAR, str(tmp_path))
    return tmp_path


@pytest.fixture
def offline_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LUCIDIUM_OFFLINE", "1")
    yield


# Make the package importable from `src/` layout when pytest is invoked
# directly (CI uses `pip install -e .[dev]` which handles this, but the
# fallback keeps `pytest backend/tests` working from the repo root too).
def pytest_configure(config: pytest.Config) -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in os.sys.path:
        os.sys.path.insert(0, str(src))
