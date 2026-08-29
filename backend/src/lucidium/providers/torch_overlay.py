"""Runtime torch-overlay manager.

Torch and torchvision are *excluded* from the PyInstaller freeze (see
``excludes`` in lucidium.spec) and are instead loaded at runtime from a
writable "overlay" directory placed on ``sys.path`` by
``lucidium_pyi_entry.py`` BEFORE any import that could transitively pull
in torch. This module is the production machinery that downloads,
installs, and selects that overlay.

WHY an overlay instead of freezing torch directly:
  * The correct torch build depends on the player's GPU (CUDA / ROCm /
    DirectML / Intel XPU / CPU). We can't know that at build time, and
    bundling every flavor would balloon the installer past 10 GiB.
  * The frozen exe has NO pip, so we can't ``pip install`` the right
    flavor after the fact. Instead we fetch the wheel files (a wheel is
    just a zip) from PyTorch's PEP 503 simple index and unpack them
    ourselves into the overlay dir.

Spike findings honored here:
  * The overlay must contain the *unpacked* contents of the torch wheel
    AND the torchvision wheel. ``torchgen`` ships inside the torch wheel
    (top-level ``torchgen/``), so unpacking torch provides it — we do not
    fetch it separately.
  * torch's pure-python deps (numpy / sympy / typing_extensions / etc.)
    stay FROZEN in the bundle. They are NOT placed in the overlay.
  * torch's ``__init__`` calls ``os.add_dll_directory(torch/lib)`` itself
    on Windows, so no manual DLL-directory handling is needed here.

HASH POLICY: every wheel we install MUST carry an index-advertised
``#sha256=`` fragment, and the downloaded bytes must match it. This is
the highest-privilege download in the product — the wheel is unpacked
onto ``sys.path`` and imported on the next launch — so it fails closed:
if a selected candidate has no expected hash, the install raises
``ProviderValidationError`` naming the index URL rather than trusting an
unverified wheel. Both configured indexes emit the fragment; an index
response without one means the markup changed or the response was
tampered with, and neither is a case we install through.

Layout under :func:`runtime_dir`::

    runtime/
      overlays/
        cuda/        # unpacked torch + torchvision wheels for this flavor
        directml/
        ...
      active_overlay        # one-line pointer file: the active flavor name

The ``active_overlay`` pointer file contains a single line: the flavor
name (e.g. ``cuda``). The entry shim reads it (or honors the
``LUCIDIUM_TORCH_OVERLAY`` env override) to decide which dir to prepend
to ``sys.path``.

NOTE ON IMPORTS: keep this module's *top-level* imports dependency-free
stdlib only (no torch, no lucidium app modules). The path-resolution
helpers at the top (:func:`runtime_dir`, :func:`active_overlay_path`,
etc.) are intentionally importable from the tiny PyInstaller entry shim,
which must not pull in anything heavy. Network / hash / zip machinery is
imported lazily inside the functions that need it.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Install-in-flight gate. Image-gen consults this and refuses to render
# while a torch overlay (GPU wheel, usually) is downloading + unpacking —
# the product call is "show a progress bar, don't quietly fall back to
# CPU-SDXL for the multi-minute window the GPU wheel takes to install."
# A counter (not a bool) so concurrent or nested installs are tolerated;
# the gate stays raised until every install has unwound.
# ---------------------------------------------------------------------------
import threading as _threading  # noqa: E402  (kept beside the gate it implements)

_install_in_flight_lock = _threading.Lock()
_install_in_flight_count = 0


def is_install_in_flight() -> bool:
    """True while at least one ``install_flavor`` call is mid-execution.

    Image-gen calls this before kicking off a render and refuses
    (raising ``ProviderUnreachableError`` tagged as ``torch_installing``)
    while a torch wheel is being downloaded or unpacked. Without the
    gate the renderer would fall back to whatever overlay is currently
    active — typically the seeded CPU torch — and the player would
    silently get minutes-per-image CPU-SDXL renders during the very
    window where the GPU wheel they actually wanted is being fetched.
    """
    with _install_in_flight_lock:
        return _install_in_flight_count > 0


def _enter_install() -> None:
    global _install_in_flight_count
    with _install_in_flight_lock:
        _install_in_flight_count += 1


def _exit_install() -> None:
    global _install_in_flight_count
    with _install_in_flight_lock:
        _install_in_flight_count = max(0, _install_in_flight_count - 1)


# ---------------------------------------------------------------------------
# Download timeouts. Both are wall-clock seconds and both are *safety*
# bounds, not performance tuning: whatever they're set to, the download
# must terminate so ``install_flavor``'s ``finally`` can lower the
# in-flight gate. Defaults can be overridden by env (useful for the
# frozen bundle / CI) or by the settings-driven ``configure_download_
# timeouts`` call the app makes before auto-provisioning.
# ---------------------------------------------------------------------------

_DOWNLOAD_CHUNK = 1024 * 256

# Connect + handshake budget. Matches embedded_models.download_model's 60s.
DEFAULT_CONNECT_TIMEOUT_S = 60.0
# Max time with ZERO bytes arriving before we call the transfer dead.
# Generous: a real multi-GB wheel on a slow link still delivers *some*
# bytes every few seconds, so only a black-holed connection trips this.
DEFAULT_STALL_TIMEOUT_S = 120.0


def _env_timeout(name: str, fallback: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return fallback
    try:
        value = float(raw)
    except ValueError:
        _log.warning("torch overlay: ignoring non-numeric %s=%r", name, raw)
        return fallback
    if value <= 0:
        _log.warning("torch overlay: ignoring non-positive %s=%r", name, raw)
        return fallback
    return value


_connect_timeout_s: float | None = None
_stall_timeout_s: float | None = None


def configure_download_timeouts(
    *, connect_timeout_s: float | None = None, stall_timeout_s: float | None = None
) -> None:
    """Override the download timeouts for this process (settings-driven).

    ``None`` leaves a value at its current setting; a non-positive value
    is rejected — "no timeout" is exactly the bug this exists to prevent.
    """
    global _connect_timeout_s, _stall_timeout_s
    if connect_timeout_s is not None:
        if connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be > 0")
        _connect_timeout_s = float(connect_timeout_s)
    if stall_timeout_s is not None:
        if stall_timeout_s <= 0:
            raise ValueError("stall_timeout_s must be > 0")
        _stall_timeout_s = float(stall_timeout_s)


def download_timeouts() -> tuple[float, float]:
    """``(connect_timeout_s, stall_timeout_s)`` in effect right now:
    explicit configuration > env override > built-in default."""
    connect = (
        _connect_timeout_s
        if _connect_timeout_s is not None
        else _env_timeout("LUCIDIUM_TORCH_CONNECT_TIMEOUT_S", DEFAULT_CONNECT_TIMEOUT_S)
    )
    stall = (
        _stall_timeout_s
        if _stall_timeout_s is not None
        else _env_timeout("LUCIDIUM_TORCH_STALL_TIMEOUT_S", DEFAULT_STALL_TIMEOUT_S)
    )
    return connect, stall


# Type of the injectable downloader seam. Given a URL and a destination
# Path, stream the bytes to that path, invoking ``on_progress`` with
# (bytes_done, total_bytes_or_None) as it goes. Tests pass a fake so no
# gigabytes ever move. ``total`` is ``None`` when the server doesn't
# send a Content-Length.
ProgressCallback = Callable[[int, "int | None"], None]
Downloader = Callable[[str, Path, "ProgressCallback | None"], None]


# ---------------------------------------------------------------------------
# Path resolution — DEPENDENCY-FREE. The PyInstaller entry shim duplicates
# this logic inline (it must not import this module to stay tiny), and a
# unit test asserts the two stay in agreement. If you change the layout
# here, change it in ``lucidium_pyi_entry.py`` too.
# ---------------------------------------------------------------------------

_APP_NAME = "Lucidium"

# Env override so a dev / test can point the runtime dir somewhere
# scratch without writing to the real per-user location. Distinct from
# ``LUCIDIUM_TORCH_OVERLAY`` (which names a *fully-resolved overlay dir*
# directly, bypassing the pointer file entirely).
RUNTIME_DIR_ENV_VAR = "LUCIDIUM_RUNTIME_DIR"

# Name of the pointer file naming the active flavor. One line of text.
ACTIVE_OVERLAY_POINTER = "active_overlay"


def runtime_dir() -> Path:
    """Resolve the writable per-user runtime directory.

    Honors ``LUCIDIUM_RUNTIME_DIR`` for tests / dev. Otherwise:
      * Windows: ``%LOCALAPPDATA%\\Lucidium\\runtime``
      * macOS:   ``~/Library/Application Support/Lucidium/runtime``
      * Linux:   ``${XDG_DATA_HOME:-~/.local/share}/lucidium/runtime``

    NOTE: this is deliberately a SEPARATE tree from
    ``config.app_data_dir()`` (saves / settings / models). The runtime
    overlay holds large, regenerable, machine-specific binaries (a
    multi-GiB unpacked torch). Keeping it out of the app-data tree means
    a "reset my settings" or save-folder sync never touches the GPU
    runtime, and a torch reinstall never risks a save. Windows uses
    LOCALAPPDATA (not the roaming APPDATA the app-data dir uses) for the
    same reason — these bytes must never roam to another machine.
    """
    override = os.environ.get(RUNTIME_DIR_ENV_VAR)
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / _APP_NAME / "runtime"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APP_NAME / "runtime"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    # Lower-case "lucidium" on Linux to match XDG / freedesktop convention
    # (app-id style), where the Windows/macOS dirs use the branded name.
    return Path(base) / "lucidium" / "runtime"


def overlays_dir() -> Path:
    """Directory holding one subdir per installed flavor."""
    return runtime_dir() / "overlays"


def overlay_dir_for(flavor: str) -> Path:
    """Resolved overlay dir for a given flavor (may not exist yet)."""
    return overlays_dir() / flavor


def _pointer_path() -> Path:
    return runtime_dir() / ACTIVE_OVERLAY_POINTER


def active_flavor() -> str | None:
    """Read the active-flavor name from the pointer file, or ``None``
    when no flavor has been activated yet (fresh install) or the
    pointer is missing / unreadable."""
    pointer = _pointer_path()
    try:
        text = pointer.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return text or None


def active_overlay_path() -> Path | None:
    """Return the resolved active overlay dir, or ``None`` when there is
    no active flavor or its overlay dir is missing on disk.

    Used by the entry shim (to decide what to prepend to ``sys.path``)
    and by diagnostics. Returns ``None`` rather than a stale path so the
    shim never prepends a half-deleted overlay.
    """
    flavor = active_flavor()
    if flavor is None:
        return None
    path = overlay_dir_for(flavor)
    if not path.is_dir():
        return None
    return path


def installed_flavors() -> list[str]:
    """List the flavors that have an overlay dir on disk. Sorted for a
    stable UI. A flavor is "installed" iff its dir exists AND is
    non-empty — a staging dir that was swapped in atomically always has
    contents, so an empty dir means a partially-cleaned-up install we
    shouldn't advertise."""
    root = overlays_dir()
    if not root.is_dir():
        return []
    out: list[str] = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and any(entry.iterdir()):
            out.append(entry.name)
    return out


# ---------------------------------------------------------------------------
# Bundled CPU overlay seeding (first-run, offline, no-download)
# ---------------------------------------------------------------------------
#
# The installer ships a pre-unpacked CPU torch+torchvision overlay INSIDE
# the frozen bundle (see lucidium.spec — it lands next to the frozen exe at
# ``bundled-overlay/cpu/`` via the PyInstaller ``datas`` mechanism). On the
# very first launch there is no active overlay yet, so we seed the runtime
# dir from this bundled copy and activate ``cpu`` — making image generation
# work OFFLINE, out of the box, with NO download. A user with a GPU can
# later download a faster flavor from the UI; the bundled CPU one is the
# universal floor.
#
# WHY a copy into the runtime dir instead of just pointing sys.path at the
# bundled dir directly:
#   * The bundled dir lives under the (read-only on some installs / wiped on
#     uninstall-reinstall) app resources. The runtime dir is the writable,
#     per-user, machine-local tree all the other flavors live in — keeping
#     ``cpu`` there means ``installed_flavors()`` / the flavor switcher / the
#     atomic-swap reinstall logic all treat it uniformly.
#   * It decouples the active overlay's lifetime from the app bundle's.

# Name of the bundled-overlay dir shipped beside the frozen exe.
BUNDLED_OVERLAY_DIRNAME = "bundled-overlay"
# The single flavor we pre-bake into the installer. CPU runs anywhere.
BUNDLED_FLAVOR = "cpu"


def bundled_overlay_source() -> Path | None:
    """Locate the pre-unpacked CPU overlay shipped inside the frozen build,
    or ``None`` when running unfrozen (dev) or when it isn't present.

    PyInstaller exposes the bundle root via ``sys._MEIPASS`` (one-file) or
    the directory containing the exe (one-folder, which is our layout). The
    spec bundles the overlay at ``<root>/bundled-overlay/cpu/``. We probe
    both layouts plus the ``_internal`` subdir PyInstaller 6 uses for
    one-folder data, mirroring ``app._configure_rembg_home``'s probing.

    Returns the flavor dir (``.../bundled-overlay/cpu``) iff it exists and is
    non-empty — an empty dir means a broken/partial bundle we must not seed
    from (we'd write an empty overlay and then fail torch import).
    """
    if not getattr(sys, "frozen", False):
        return None
    candidates: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / BUNDLED_OVERLAY_DIRNAME / BUNDLED_FLAVOR)
    exe_dir = Path(sys.executable).resolve().parent
    candidates.append(exe_dir / "_internal" / BUNDLED_OVERLAY_DIRNAME / BUNDLED_FLAVOR)
    candidates.append(exe_dir / BUNDLED_OVERLAY_DIRNAME / BUNDLED_FLAVOR)
    for cand in candidates:
        if cand.is_dir() and any(cand.iterdir()):
            return cand
    return None


def seed_bundled_overlay(*, source: Path | None = None) -> str | None:
    """First-run seed: if NO overlay is active, copy the bundled CPU overlay
    into ``overlays/cpu`` and activate it. Returns the activated flavor name
    (``"cpu"``) when it seeded, or ``None`` when it did nothing.

    Idempotent + cheap to call on every launch:
      * If an overlay is already active (pointer present AND its dir exists),
        do nothing — never clobber a GPU flavor the user downloaded.
      * If the bundled source isn't present (dev / non-frozen run, or a build
        that didn't bake one in), do nothing.
      * If ``overlays/cpu`` somehow already exists non-empty but isn't active,
        just activate it (no re-copy).

    The copy goes to a staging dir on the same volume and is atomically
    swapped into place, reusing :func:`_atomic_swap_dir`, so a crash mid-copy
    never leaves a half-populated overlay that torch would try to import.

    This MUST run BEFORE torch is first imported (the entry shim calls it,
    then prepends the freshly-seeded overlay to ``sys.path``). It is
    dependency-light (stdlib ``shutil`` only) so the import-light entry shim
    can safely trigger it.
    """
    import shutil
    import tempfile

    # Already have an active, on-disk overlay? Leave it alone.
    if active_overlay_path() is not None:
        return None

    src = source if source is not None else bundled_overlay_source()
    if src is None:
        return None

    final_dir = overlay_dir_for(BUNDLED_FLAVOR)
    if final_dir.is_dir() and any(final_dir.iterdir()):
        # Present but not active (e.g. pointer lost) — just re-activate.
        write_active_flavor(BUNDLED_FLAVOR)
        return BUNDLED_FLAVOR

    rt = runtime_dir()
    rt.mkdir(parents=True, exist_ok=True)
    _log.info("seeding bundled %s overlay from %s", BUNDLED_FLAVOR, src)
    # Stage on the same volume as the final dir so the swap is a rename.
    with tempfile.TemporaryDirectory(dir=str(rt), prefix=f".seed-{BUNDLED_FLAVOR}-") as staging_str:
        staging = Path(staging_str) / "overlay"
        # ``dirs_exist_ok`` is irrelevant (fresh staging) but keep the copy
        # robust to symlinks the wheel never contains.
        shutil.copytree(src, staging, symlinks=False)
        _atomic_swap_dir(staging, final_dir)

    write_active_flavor(BUNDLED_FLAVOR)
    _log.info("seeded + activated bundled %s overlay at %s", BUNDLED_FLAVOR, final_dir)
    return BUNDLED_FLAVOR


# ---------------------------------------------------------------------------
# Flavor recommendation + wheel matrix
# ---------------------------------------------------------------------------

# The valid overlay flavors. ``cpu`` is the universal fallback that runs
# anywhere (slowly). Order is the recommendation precedence reference.
FLAVORS_VALID: tuple[str, ...] = ("cuda", "rocm", "directml", "xpu", "cpu")


@dataclass(frozen=True)
class FlavorSpec:
    """How to install one flavor.

    ``index_url`` is the PyTorch PEP 503 *simple index* base for the
    torch/torchvision wheels. We append ``torch/`` (or
    ``torchvision/``) to reach each project's file list.

    ``wheels`` is the list of distribution names to fetch + unpack into
    the overlay. ``directml`` is special: torch and torchvision come
    from the CPU index (the public DirectML stack runs the CPU build of
    torch and registers the DML device through the separate
    ``torch-directml`` wheel), so it ALSO lists ``torch-directml`` and
    pulls that one from PyPI rather than the PyTorch index.
    """

    index_url: str
    wheels: tuple[str, ...]
    # Distributions in ``wheels`` that come from PyPI's simple index
    # rather than ``index_url`` (currently only ``torch-directml``).
    pypi_wheels: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


# PyTorch publishes per-accelerator wheels under download.pytorch.org.
# Pins (cu130 / rocm7.2) mirror the strings the embedded_image_client
# diagnostics already hand users, so the overlay installs exactly what
# the "reinstall torch" advice elsewhere names.
_PYTORCH_INDEX = "https://download.pytorch.org/whl"
_PYPI_INDEX = "https://pypi.org/simple"

FLAVORS: dict[str, FlavorSpec] = {
    "cuda": FlavorSpec(
        # CUDA 13.0, not the newer cu132: both ship the same torch
        # (2.13.0), but the 13.2 wheels want a newer NVIDIA driver. The
        # overlay downloads onto whatever machine the player owns, so
        # the pin tracks the widest-supported driver floor that still
        # carries current torch.
        index_url=f"{_PYTORCH_INDEX}/cu130",
        wheels=("torch", "torchvision"),
        description="NVIDIA CUDA 13.0",
    ),
    "rocm": FlavorSpec(
        index_url=f"{_PYTORCH_INDEX}/rocm7.2",
        wheels=("torch", "torchvision"),
        description="AMD ROCm 7.2 (Linux)",
    ),
    "xpu": FlavorSpec(
        index_url=f"{_PYTORCH_INDEX}/xpu",
        wheels=("torch", "torchvision"),
        description="Intel Arc / oneAPI XPU",
    ),
    "cpu": FlavorSpec(
        index_url=f"{_PYTORCH_INDEX}/cpu",
        wheels=("torch", "torchvision"),
        description="CPU-only (no GPU acceleration)",
    ),
    "directml": FlavorSpec(
        # torch / torchvision are the plain CPU builds; the DML device
        # is registered by the separate ``torch-directml`` wheel from
        # PyPI. Mirrors how ``detect_compute_device`` treats DirectML as
        # CPU-torch + a privateuseone device.
        index_url=f"{_PYTORCH_INDEX}/cpu",
        wheels=("torch", "torchvision", "torch-directml"),
        pypi_wheels=("torch-directml",),
        description="DirectML (Windows AMD / any DX12 GPU)",
    ),
}


def recommend_flavor(
    *,
    platform: str | None = None,
    has_nvidia: Callable[[], bool] | None = None,
    has_amd: Callable[[], bool] | None = None,
    has_intel_arc: Callable[[], bool] | None = None,
) -> str:
    """Recommend the overlay flavor for THIS machine.

    Rules (vendor precedence NVIDIA → AMD → Intel Arc → CPU):
      * NVIDIA present (any OS)            → ``cuda``
      * AMD present + Linux               → ``rocm``
      * AMD present + Windows             → ``directml``
      * Intel Arc present (any OS)        → ``xpu``
      * otherwise                         → ``cpu``

    Defaults pull the vendor probes from ``embedded_image_client`` so the
    overlay and the live device-detection agree on what hardware is
    present. The probes / platform are injectable so tests can sweep
    every (platform, vendor) combination without real hardware.

    macOS note: there is no Apple-Silicon overlay flavor. MPS torch is
    the standard PyPI macOS wheel and (in this milestone) is expected to
    ship via the platform's default install path rather than the
    overlay; with no NVIDIA/AMD/Arc match a Mac recommends ``cpu`` here.
    """
    if platform is None:
        platform = sys.platform
    if has_nvidia is None or has_amd is None or has_intel_arc is None:
        # Reuse the live vendor probes. Imported lazily so this module's
        # top level stays free of the (heavier) image-client import,
        # keeping it safe for the entry shim to touch the path helpers.
        from .embedded_image_client import (
            _has_amd_gpu,
            _has_intel_arc_gpu,
            _has_nvidia_gpu,
        )

        has_nvidia = has_nvidia or _has_nvidia_gpu
        has_amd = has_amd or _has_amd_gpu
        has_intel_arc = has_intel_arc or _has_intel_arc_gpu

    if has_nvidia():
        return "cuda"
    if has_amd():
        # AMD splits by OS: ROCm is Linux-only; Windows AMD goes through
        # DirectML (matches embedded_image_client's device mapping).
        return "rocm" if platform == "linux" else "directml"
    if has_intel_arc():
        return "xpu"
    return "cpu"


# ---------------------------------------------------------------------------
# Wheel resolution from a PEP 503 simple index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WheelCandidate:
    """One ``.whl`` link parsed out of a simple-index page."""

    filename: str
    url: str
    # Hash from the link fragment (``#sha256=...``), if the index
    # exposed one. ``(algo, hexdigest)`` or ``None``.
    expected_hash: tuple[str, str] | None


class _SimpleIndexParser:
    """Minimal PEP 503 simple-index HTML parser.

    The PyTorch index pages are flat lists of ``<a href=...>NAME</a>``
    anchors, one per file. We only need the href (the file URL, possibly
    relative) and any ``#sha256=`` fragment on it. Using the stdlib
    ``html.parser`` keeps this dependency-free — no bs4 / lxml in the
    frozen bundle.
    """

    def __init__(self) -> None:
        import html.parser

        self._links: list[tuple[str, str | None]] = []

        outer = self

        class _HTMLParser(html.parser.HTMLParser):
            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if tag != "a":
                    return
                href = None
                for key, value in attrs:
                    if key == "href" and value:
                        href = value
                if href is None:
                    return
                base, _, fragment = href.partition("#")
                expected = None
                # PEP 503 hash fragment form: ``sha256=<hex>``.
                if fragment.startswith("sha256="):
                    expected = fragment[len("sha256=") :]
                outer._links.append((base, expected))

        self._parser = _HTMLParser()

    def feed(self, html_text: str) -> list[tuple[str, str | None]]:
        self._parser.feed(html_text)
        return self._links


def _interpreter_tags() -> tuple[str, str, str]:
    """Return ``(py_tag, abi_tag, platform_substrings)`` describing the
    wheels THIS interpreter can load.

    Returns a coarse description used by :func:`select_wheel`:
      * ``py_tag``        e.g. ``cp314`` — CPython 3.14
      * ``abi_tag``       e.g. ``cp314`` (the stable/abi3 case is matched
                          separately by select_wheel)
      * ``platform_tag``  SPACE-SEPARATED substrings that must ALL appear in
                          the wheel's platform tag — an OS token plus an
                          ARCH token (e.g. ``"linux x86_64"``, ``"win amd64"``,
                          ``"macosx arm64"``). The arch token is mandatory:
                          PyTorch publishes ``linux_s390x`` and
                          ``manylinux_*_aarch64`` wheels whose platform tags
                          also contain ``linux``, so a bare ``"linux"`` match
                          would (and did) pick the s390x mainframe wheel on an
                          x86_64 host, yielding a torch that can't load.
    """
    impl = "cp" if sys.implementation.name == "cpython" else sys.implementation.name[:2]
    py_tag = f"{impl}{sys.version_info.major}{sys.version_info.minor}"
    import platform as _platform

    machine = _platform.machine().lower()  # x86_64 / amd64 / aarch64 / arm64 / s390x
    is_arm = machine in ("arm64", "aarch64")
    if sys.platform == "win32":
        # win_amd64 vs win_arm64 — require the arch token too so an ARM
        # wheel is never selected on an x86_64 host or vice-versa.
        plat = f"win {'arm64' if is_arm else 'amd64'}"
    elif sys.platform == "darwin":
        plat = f"macosx {'arm64' if is_arm else 'x86_64'}"
    else:
        # Linux: the arch token is what stops a bare "linux" match from
        # grabbing linux_s390x / manylinux_*_aarch64 on an x86_64 box.
        plat = f"linux {'aarch64' if is_arm else (machine or 'x86_64')}"
    return py_tag, py_tag, plat


def select_wheel(
    candidates: list[WheelCandidate],
    *,
    py_tag: str | None = None,
    abi_tag: str | None = None,
    platform_tag: str | None = None,
) -> WheelCandidate:
    """Pick the single wheel matching this interpreter from a parsed
    index file list.

    A wheel filename is ``{dist}-{version}(-{build})?-{python}-{abi}-{platform}.whl``.
    We match on:
      * python tag containing our ``py_tag`` (``cp314``) OR a compatible
        abi3 tag (``cp3X`` with X <= ours, ``abi3`` abi) — abi3 wheels
        load on any newer CPython.
      * platform tag containing ``platform_tag`` (e.g. ``win_amd64``).

    Among matches, prefer the HIGHEST version (PyTorch lists ascending,
    but we sort defensively). Raises ``ProviderValidationError`` when no
    wheel matches — that's the "no build for your Python/OS yet" case the
    UI must surface clearly (e.g. a brand-new CPython before torch has a
    wheel for it).

    Kept dependency-light: version comparison is a tuple of ints parsed
    from the leading numeric dotted segment, which is sufficient to rank
    PyTorch's release wheels (they don't mix pre-releases into these
    index pages).
    """
    from ..api.errors import ProviderValidationError

    if py_tag is None or abi_tag is None or platform_tag is None:
        d_py, d_abi, d_plat = _interpreter_tags()
        py_tag = py_tag or d_py
        abi_tag = abi_tag or d_abi
        platform_tag = platform_tag or d_plat

    our_minor = sys.version_info.minor

    def _matches(filename: str) -> bool:
        if not filename.endswith(".whl"):
            return False
        stem = filename[: -len(".whl")]
        parts = stem.split("-")
        # Need at least dist-version-py-abi-platform (5 parts); a build
        # tag inserts a 6th between version and py.
        if len(parts) < 5:
            return False
        wheel_plat = parts[-1]
        wheel_abi = parts[-2]
        wheel_py = parts[-3]
        # platform_tag is one OR MORE space-separated tokens (OS + arch)
        # that must ALL appear in the wheel's platform tag. A single token
        # (e.g. tests passing "win_amd64") still works; the multi-token
        # default ("linux x86_64") is what rejects the wrong-arch linux
        # wheels (s390x / aarch64) that also contain "linux".
        if not all(tok in wheel_plat for tok in platform_tag.split()):
            return False
        # Exact interpreter match (cp314-cp314-...).
        if py_tag in wheel_py.split("."):
            return True
        # abi3 forward-compat: cp3X-abi3 wheels load on any CPython >= 3X.
        if wheel_abi == "abi3":
            for tag in wheel_py.split("."):
                if tag.startswith("cp3") and tag[3:].isdigit():
                    if int(tag[3:]) <= our_minor:
                        return True
        return False

    def _version_key(c: WheelCandidate) -> tuple[int, ...]:
        stem = c.filename[: -len(".whl")]
        parts = stem.split("-")
        version = parts[1] if len(parts) > 1 else "0"
        nums: list[int] = []
        for chunk in version.split("."):
            digits = ""
            for ch in chunk:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            nums.append(int(digits) if digits else 0)
        return tuple(nums)

    matching = [c for c in candidates if _matches(c.filename)]
    if not matching:
        raise ProviderValidationError(
            f"no compatible wheel found for {py_tag}/{platform_tag} among "
            f"{len(candidates)} index entries — PyTorch may not publish a "
            "build for this Python version / platform yet"
        )
    matching.sort(key=_version_key, reverse=True)
    return matching[0]


def _resolve_index_url(base: str, project: str) -> str:
    """PEP 503 project page URL: ``{base}/{normalized}/``. PyTorch's
    index normalizes ``torch-directml`` to ``torch-directml`` (dash kept,
    not underscored) on its own index; PyPI normalizes to the dash form
    too. We normalize per PEP 503 (lowercase, runs of [-_.] → single
    dash)."""
    import re

    normalized = re.sub(r"[-_.]+", "-", project).lower()
    return f"{base.rstrip('/')}/{normalized}/"


def parse_index(html_text: str, base_url: str) -> list[WheelCandidate]:
    """Parse a simple-index project page into wheel candidates,
    resolving relative hrefs against ``base_url``."""
    from urllib.parse import unquote, urljoin

    parser = _SimpleIndexParser()
    out: list[WheelCandidate] = []
    for href, sha in parser.feed(html_text):
        # PyTorch percent-encodes ``+`` (local version separator) as
        # ``%2B`` in the href; decode so the filename matches the real
        # wheel name (and our tag-splitting sees ``cp314`` etc.).
        filename = unquote(href.rsplit("/", 1)[-1])
        if not filename.endswith(".whl"):
            continue
        url = urljoin(base_url, href)
        expected = ("sha256", sha) if sha else None
        out.append(WheelCandidate(filename=filename, url=url, expected_hash=expected))
    return out


# ---------------------------------------------------------------------------
# Download / unpack / atomic swap
# ---------------------------------------------------------------------------


def _default_downloader(url: str, dest: Path, on_progress: ProgressCallback | None) -> None:
    """Stream ``url`` to ``dest`` with optional progress callbacks.

    The ONLY function in this module that touches the network. Injected
    elsewhere (the ``downloader`` arg of :func:`install_flavor`) so tests
    never hit the wire. Uses stdlib ``urllib`` so the frozen bundle needs
    no ``requests``.

    BOUNDED IN TIME, ALWAYS. This runs automatically at startup (see
    ``app._auto_provision_torch_overlay``) inside the install-in-flight
    gate, so a hang here doesn't just stall the download: it pins
    :func:`is_install_in_flight` True for the process lifetime and every
    image render is refused with ``torch_installing`` forever. Two
    independent bounds prevent that:

      * ``timeout=`` on ``urlopen`` — caps connect + handshake, and (as
        the socket timeout) each individual socket read.
      * a stall watchdog around the streaming read — the read happens on
        a helper thread and the caller waits at most ``stall_timeout``
        for each chunk. This catches the cases the socket timeout can't:
        a connection that dribbles nothing while staying alive, or a
        response object whose ``read`` blocks below the socket layer.

    On either bound the response is closed and
    ``ProviderUnreachableError`` is raised, so ``install_flavor``'s
    ``finally: _exit_install()`` runs and the renderer recovers.
    """
    import urllib.request

    from ..api.errors import ProviderUnreachableError

    connect_timeout, stall_timeout = download_timeouts()
    req = urllib.request.Request(url, headers={"User-Agent": "Lucidium-torch-overlay"})
    try:
        resp = urllib.request.urlopen(req, timeout=connect_timeout)
    except OSError as exc:  # URLError / socket.timeout are both OSError
        raise ProviderUnreachableError(
            f"could not open {url} within {connect_timeout:.0f}s: {exc}"
        ) from exc
    with resp:
        _stream_response(resp, url, dest, on_progress, stall_timeout)


def _abort_response(resp: object) -> None:
    """Close a response so the (possibly blocked) reader thread unwinds.

    Best-effort: we're already on an error path, and a close that itself
    raises must not mask the timeout we're reporting."""
    try:
        close = getattr(resp, "close", None)
        if callable(close):
            close()
    except Exception:
        _log.debug("torch overlay: closing stalled response failed", exc_info=True)


class _ResponseHeaders(Protocol):
    def get(self, name: str) -> str | None: ...


class _StreamableResponse(Protocol):
    """The subset of an ``http.client.HTTPResponse`` that streaming needs.

    A Protocol (not the concrete class) because tests inject fakes; the
    ``headers`` attribute is still probed with ``hasattr`` because a fake
    may omit it."""

    @property
    def headers(self) -> _ResponseHeaders: ...

    def read(self, amt: int = ..., /) -> bytes: ...


def _stream_response(
    resp: _StreamableResponse,
    url: str,
    dest: Path,
    on_progress: ProgressCallback | None,
    stall_timeout: float,
) -> None:
    """Copy ``resp`` into ``dest``, aborting if no chunk arrives within
    ``stall_timeout`` seconds.

    Each ``resp.read`` runs on a single-worker pool so the wait is
    interruptible — a ``read`` that blocks forever would otherwise be
    unkillable from this thread. On timeout we close the response (which
    normally unblocks the reader) and abandon the worker; it's a daemon-
    equivalent pool thread and cannot hold the install gate, which is
    released by ``install_flavor``'s ``finally`` as soon as we raise.
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as _FutureTimeout

    from ..api.errors import ProviderUnreachableError

    total_header = resp.headers.get("Content-Length") if hasattr(resp, "headers") else None
    total = int(total_header) if total_header and total_header.isdigit() else None
    done = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="torch-overlay-dl")
    try:
        with open(dest, "wb") as fh:
            while True:
                future = pool.submit(resp.read, _DOWNLOAD_CHUNK)
                try:
                    chunk = future.result(timeout=stall_timeout)
                except _FutureTimeout:
                    _abort_response(resp)
                    raise ProviderUnreachableError(
                        f"download stalled: no data from {url} for "
                        f"{stall_timeout:.0f}s after {done} bytes; aborting so "
                        "image generation isn't blocked indefinitely"
                    ) from None
                except OSError as exc:
                    raise ProviderUnreachableError(
                        f"download of {url} failed after {done} bytes: {exc}"
                    ) from exc
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if on_progress is not None:
                    on_progress(done, total)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _fetch_index(url: str, downloader: Downloader) -> str:
    """Fetch a simple-index page as text via the injected downloader
    (so tests stub it). The downloader writes bytes to a temp file; we
    read them back. Kept on the same seam as wheel downloads so a single
    mock covers both."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "index.html"
        downloader(url, target, None)
        return target.read_text(encoding="utf-8", errors="replace")


def _verify_hash(path: Path, expected: tuple[str, str]) -> None:
    """Verify a downloaded file against an index-provided hash. Raises
    ``ProviderValidationError`` on mismatch — a corrupt/tampered wheel
    must never be unpacked onto sys.path."""
    import hashlib

    from ..api.errors import ProviderValidationError

    algo, expected_hex = expected
    hasher = hashlib.new(algo)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(block)
    actual = hasher.hexdigest()
    if actual.lower() != expected_hex.lower():
        raise ProviderValidationError(
            f"hash mismatch for {path.name}: index advertised {algo}:"
            f"{expected_hex} but downloaded file is {algo}:{actual}"
        )


def _unpack_wheel(wheel_path: Path, dest_dir: Path) -> None:
    """Unpack a wheel (a zip) into ``dest_dir``. The wheel's top-level
    package dirs (``torch/``, ``torchgen/``, ``torchvision/``, the
    ``*.dist-info``) land directly under ``dest_dir`` so the dir works as
    a ``sys.path`` entry. Skips the ``*.data/scripts`` payload — those
    are console-script shims we never invoke from the overlay."""
    import zipfile

    with zipfile.ZipFile(wheel_path) as zf:
        for member in zf.namelist():
            # ``{dist}-{ver}.data/scripts/...`` holds entry-point shims;
            # they belong in a bin/ dir, not on sys.path. Drop them.
            if ".data/scripts/" in member:
                continue
            zf.extract(member, dest_dir)


def _atomic_swap_dir(staging: Path, final: Path) -> None:
    """Atomically move ``staging`` into place at ``final``.

    On a fresh install ``os.replace`` of a dir works directly. When
    ``final`` already exists (reinstall / flavor switch), ``os.replace``
    can't overwrite a non-empty dir on Windows, so we move the old dir
    aside first, swap in the new one, then delete the old. The window
    where ``final`` doesn't exist is microseconds and only matters if the
    app is mid-import of torch — which never happens during an install
    (the overlay being installed is, by construction, not the active one
    being imported).
    """
    import shutil
    import uuid

    final.parent.mkdir(parents=True, exist_ok=True)
    if not final.exists():
        os.replace(staging, final)
        return
    backup = final.with_name(f"{final.name}.old-{uuid.uuid4().hex[:8]}")
    os.replace(final, backup)
    try:
        os.replace(staging, final)
    except OSError:
        # Roll back so we don't leave the flavor uninstalled.
        os.replace(backup, final)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def write_active_flavor(flavor: str) -> None:
    """Atomically point the active-overlay pointer at ``flavor``.

    Write-to-temp + ``os.replace`` so a crash mid-write never leaves a
    truncated pointer that the entry shim would misread (and then prepend
    a garbage path to sys.path)."""
    import uuid

    rt = runtime_dir()
    rt.mkdir(parents=True, exist_ok=True)
    pointer = rt / ACTIVE_OVERLAY_POINTER
    tmp = pointer.with_name(f"{ACTIVE_OVERLAY_POINTER}.tmp-{uuid.uuid4().hex[:8]}")
    tmp.write_text(flavor + "\n", encoding="utf-8")
    os.replace(tmp, pointer)


@dataclass
class InstallResult:
    flavor: str
    overlay_dir: Path
    # True when the flavor was already installed and we skipped the
    # download (idempotent re-install). The UI can surface "already
    # installed" vs "downloaded".
    skipped: bool
    activated: bool


def install_flavor(
    flavor: str,
    *,
    downloader: Downloader | None = None,
    on_progress: ProgressCallback | None = None,
    activate: bool = True,
    force: bool = False,
) -> InstallResult:
    """Download, unpack, and (by default) activate an overlay flavor.

    Steps:
      1. Validate the flavor against :data:`FLAVORS`.
      2. If already installed and not ``force`` → skip the download,
         optionally re-activate, return ``skipped=True`` (idempotent).
      3. For each wheel in the flavor spec: fetch the project's simple-
         index page, select the wheel matching THIS interpreter, download
         it to a temp dir (streamed progress), verify its hash if the
         index exposed one, and unpack it into a staging dir.
      4. Atomically swap the staging dir into ``overlays/<flavor>/``.
      5. Atomically update the active-overlay pointer (when ``activate``).

    Network I/O goes through ``downloader`` (defaults to the real stdlib
    streamer); tests inject a fake so nothing large moves. Progress is
    reported through ``on_progress`` as ``(bytes_done, total_or_None)``
    aggregated across all wheels of the flavor.

    Raises ``ProviderValidationError`` for an unknown flavor or no
    matching wheel; ``ProviderUnreachableError`` for download failures.
    """

    _enter_install()
    try:
        return _install_flavor_inner(
            flavor,
            downloader=downloader,
            on_progress=on_progress,
            activate=activate,
            force=force,
        )
    finally:
        _exit_install()


def _install_flavor_inner(
    flavor: str,
    *,
    downloader: Downloader | None,
    on_progress: ProgressCallback | None,
    activate: bool,
    force: bool,
) -> InstallResult:
    """Real install body; ``install_flavor`` wraps this with the in-flight
    counter so ``is_install_in_flight()`` reports True for the whole
    download+unpack+swap window. Kept separate so the gating is impossible
    to skip via an early-return."""
    import tempfile

    from ..api.errors import ProviderUnreachableError, ProviderValidationError

    if flavor not in FLAVORS:
        raise ProviderValidationError(
            f"unknown torch overlay flavor {flavor!r}; valid: {', '.join(sorted(FLAVORS))}"
        )
    spec = FLAVORS[flavor]
    final_dir = overlay_dir_for(flavor)

    # Idempotent fast-path: a non-empty overlay dir means a prior install
    # completed (the atomic swap only ever lands a fully-unpacked dir).
    already = final_dir.is_dir() and any(final_dir.iterdir())
    if already and not force:
        _log.info("torch overlay %s already installed; skipping download", flavor)
        if activate:
            write_active_flavor(flavor)
        return InstallResult(flavor=flavor, overlay_dir=final_dir, skipped=True, activated=activate)

    download = downloader or _default_downloader

    # Resolve every wheel first so we fail fast (no compatible wheel)
    # before downloading anything.
    selected: list[WheelCandidate] = []
    for project in spec.wheels:
        base = _PYPI_INDEX if project in spec.pypi_wheels else spec.index_url
        index_url = _resolve_index_url(base, project)
        _log.info("torch overlay %s: fetching index %s", flavor, index_url)
        html_text = _fetch_index(index_url, download)
        candidates = parse_index(html_text, index_url)
        candidate = select_wheel(candidates)
        # Fail closed: an unverifiable wheel is never unpacked onto
        # sys.path. See the HASH POLICY note in the module docstring.
        if candidate.expected_hash is None:
            raise ProviderValidationError(
                f"index {index_url} advertised no #sha256= hash for "
                f"{candidate.filename}; refusing to install an unverified "
                f"wheel for overlay flavor {flavor}"
            )
        selected.append(candidate)

    rt = runtime_dir()
    rt.mkdir(parents=True, exist_ok=True)
    # Staging dir lives under the runtime dir (same filesystem as the
    # final overlay) so the final ``os.replace`` is a cheap intra-volume
    # rename, not a cross-device copy.
    with tempfile.TemporaryDirectory(dir=str(rt), prefix=f".staging-{flavor}-") as staging_str:
        staging = Path(staging_str)
        overlay_staging = staging / "overlay"
        overlay_staging.mkdir()
        dl_dir = staging / "downloads"
        dl_dir.mkdir()

        # Aggregate progress across wheels: report cumulative bytes so a
        # single progress bar in the UI advances monotonically.
        cumulative = 0

        def _aggregate(done: int, total: int | None, _base: int = 0) -> None:
            # ``_base`` captures bytes from already-finished wheels.
            # Pass the total THROUGH (offset by ``_base``) when the server
            # sent a Content-Length: a determinate bar is what lets the UI
            # tell a stalled download from a merely slow one — with
            # ``total=None`` both render as the same indeterminate bar.
            # The total only covers wheels resolved so far, so it can step
            # up when the next wheel starts; torch dwarfs torchvision, so
            # in practice that's a sub-percent adjustment near the end.
            if on_progress is not None:
                on_progress(_base + done, None if total is None else _base + total)

        for candidate in selected:
            wheel_path = dl_dir / candidate.filename
            base_bytes = cumulative

            # Bound as a default arg (not a closure over ``cumulative``) so
            # each wheel's callback keeps ITS starting offset.
            def _wheel_progress(done: int, total: int | None, _base: int = base_bytes) -> None:
                _aggregate(done, total, _base)

            try:
                download(
                    candidate.url,
                    wheel_path,
                    _wheel_progress,
                )
            except Exception as exc:
                raise ProviderUnreachableError(
                    f"failed to download {candidate.filename} for overlay flavor {flavor}: {exc}"
                ) from exc
            assert candidate.expected_hash is not None  # enforced above
            _verify_hash(wheel_path, candidate.expected_hash)
            cumulative += wheel_path.stat().st_size if wheel_path.exists() else 0
            _unpack_wheel(wheel_path, overlay_staging)

        # Atomic publish: swap the fully-unpacked staging overlay into
        # place. Until this line, ``final_dir`` either doesn't exist or
        # holds the previous good install — never a half-written one.
        _atomic_swap_dir(overlay_staging, final_dir)

    if activate:
        write_active_flavor(flavor)

    _log.info("torch overlay %s installed at %s (activated=%s)", flavor, final_dir, activate)
    return InstallResult(flavor=flavor, overlay_dir=final_dir, skipped=False, activated=activate)


def status() -> dict[str, object]:
    """Snapshot for the backend status endpoint: recommended flavor,
    installed flavors, and the currently-active one. All cheap, no
    network."""
    return {
        "recommended": recommend_flavor(),
        "installed": installed_flavors(),
        "active": active_flavor(),
        "runtime_dir": str(runtime_dir()),
    }
