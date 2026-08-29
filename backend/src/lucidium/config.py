"""Single source of truth for default constants.

Constitution V (DRY): every numeric or string default that influences
runtime behavior MUST be declared here exactly once. Importers reference
these names; they never inline literals.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Final

GAME_SCHEMA_VERSION: Final[int] = 1
PROTOCOL_VERSION: Final[int] = 1

DEFAULT_LLM_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL: Final[str] = "deepseek/deepseek-v3.2-exp"
DEFAULT_LLM_TEMPERATURE: Final[float] = 0.8
DEFAULT_LLM_MAX_TOKENS: Final[int] = 1024

DEFAULT_IMAGE_BASE_URL: Final[str] = "http://127.0.0.1:8000"
DEFAULT_IMAGE_PORTRAIT_WORKFLOW: Final[str] = "portrait.workflow.json"
DEFAULT_IMAGE_BACKGROUND_WORKFLOW: Final[str] = "background.workflow.json"

DEFAULT_PROMPT_HISTORY_CLAMP_CHARS: Final[int] = 12_000
DEFAULT_TYPEWRITER_CHARS_PER_SEC: Final[int] = 60

DEFAULT_LLM_MAX_IN_FLIGHT: Final[int] = 4
DEFAULT_IMAGE_MAX_IN_FLIGHT: Final[int] = 2

LLM_RETRY_BUDGET: Final[int] = 3
IMAGE_RETRY_BUDGET: Final[int] = 2

# Wall-clock ceiling on acquiring the ``world_init`` LLM response when a
# player presses Begin. The underlying call stack multiplies retries —
# ``call_llm_json_with_retry`` (validation) wraps ``llm_text``
# (``LLM_RETRY_BUDGET``) wraps ``complete`` (``LLM_RETRY_BUDGET``), and
# each network attempt can wait out the httpx 300 s read timeout. Left
# unbounded, a wedged provider turns "new game" into a 15–30 min freeze.
# This deadline is one read-timeout window plus margin: a single healthy
# (or legitimately slow) world_init attempt always completes inside it,
# but the retry / prefetch-fallback stacking can no longer pile up. On
# expiry the confirm handler raises a recoverable ProviderUnreachableError
# so the player just presses Begin again instead of staring at a spinner.
WORLD_INIT_DEADLINE_S: Final[float] = 330.0


def bundled_placeholder_dir() -> Path:
    """Path to the bundled white-room + dream-guide placeholders (FR-022).

    Lives next to the ComfyUI workflows. In dev these are committed to
    the repository; in a packaged build they ship as ``extraResources``.
    """
    here = Path(__file__).resolve()
    backend_root = here.parents[2]  # backend/
    return backend_root / "workflows" / "placeholders"


def bundled_white_room_path() -> Path:
    return bundled_placeholder_dir() / "white_room.png"


def bundled_dream_guide_path() -> Path:
    return bundled_placeholder_dir() / "dream_guide.png"


WS_HOST: Final[str] = "127.0.0.1"
WS_PORT_AUTO: Final[int] = 0  # bind any free port; the chosen port is printed on stdout
WS_PORT_ANNOUNCEMENT_PREFIX: Final[str] = "LUCIDIUM_WS_PORT="
# The per-launch bearer token is announced on the SAME stdout channel as
# the port, immediately before it (the port line stays the readiness
# signal the Electron main process waits on). Only our own parent
# process can read this pipe, so anything else on the box — including a
# page in the player's browser — cannot learn the token and cannot
# complete a handshake. See ``api/ws_server.run_server``.
WS_TOKEN_ANNOUNCEMENT_PREFIX: Final[str] = "LUCIDIUM_WS_TOKEN="

# Origins we accept on the WebSocket handshake. A browser ALWAYS sends
# an ``Origin`` it cannot forge, so this list is what keeps a random
# page the player visits from even reaching the token check: the packaged
# renderer loads over ``file://`` and the dev renderer over the Vite dev
# server, and neither of those is an origin an attacker can obtain.
# ``None`` (no Origin header at all) is allowed because non-browser
# clients — the integration tests, scripts — don't send one; those are
# gated by the bearer token, which is the actual authentication.
WS_ALLOWED_ORIGINS: Final[tuple[object, ...]] = (
    None,
    "file://",
    re.compile(r"http://(127\.0\.0\.1|localhost)(:\d+)?"),
)

OFFLINE_ENV_VAR: Final[str] = "LUCIDIUM_OFFLINE"
APP_DATA_OVERRIDE_ENV_VAR: Final[str] = "LUCIDIUM_APP_DATA"

_APP_NAME: Final[str] = "Lucidium"


def app_data_dir() -> Path:
    """Return the per-user app-data directory.

    Honors ``LUCIDIUM_APP_DATA`` for tests. Otherwise:
      * Windows: ``%APPDATA%/Lucidium``
      * macOS:   ``~/Library/Application Support/Lucidium``
      * Linux:   ``$XDG_DATA_HOME/Lucidium`` or ``~/.local/share/Lucidium``
    """
    override = os.environ.get(APP_DATA_OVERRIDE_ENV_VAR)
    if override:
        return Path(override)

    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / _APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / _APP_NAME


def saves_dir() -> Path:
    return app_data_dir() / "saves"


def embedded_models_dir() -> Path:
    """Default directory for the embedded image-generation backend.

    Mirrors ``saves_dir`` but holds SDXL-family safetensors
    checkpoints. The user can override it via
    ``ImageSettings.embedded_models_dir`` — empty there falls back
    to this path so the bootstrap that downloads SDXL Turbo on
    first run has somewhere reliable to write.
    """
    return app_data_dir() / "models" / "image"


def settings_path() -> Path:
    return app_data_dir() / "settings.json"


def is_offline() -> bool:
    return os.environ.get(OFFLINE_ENV_VAR) == "1"
