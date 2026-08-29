# PyInstaller spec for the Lucidium backend.
#
# Produces a one-folder bundle at ``backend/dist/lucidium-backend/``.
# electron-builder picks the folder up via the ``extraResources``
# block in ``frontend/package.json`` and copies it into the
# packaged app's ``resources/lucidium-backend/`` directory. The
# Electron main process then spawns
# ``<resources>/lucidium-backend/lucidium-backend.exe`` directly
# (see ``resolveBackendCommand`` in ``frontend/electron/main.ts``).
#
# Heavy ML deps (diffusers, transformers, accelerate) are pulled in
# via the engine's lazy imports — PyInstaller's static analysis
# misses them, so they're listed explicitly under ``hiddenimports``.
# Same story for diffusers' scheduler / model subpackages and the
# audio stack; the lists below are tuned against what the engine
# actually loads at runtime.
#
# TORCH IS NOT FROZEN HERE. torch / torchvision / torchgen are
# EXCLUDED from the bundle (see ``excludes``) and loaded at runtime
# from a writable, per-user "overlay" dir placed on sys.path by
# ``lucidium_pyi_entry.py``. WHY: the correct torch build is
# GPU-specific (CUDA / ROCm / DirectML / XPU / CPU) and can't be
# known at build time; bundling every flavor would blow the
# installer past 10 GiB. Instead the installer SHIPS a pre-unpacked
# CPU overlay (so it works offline out of the box — see the
# ``bundled-overlay`` datas entry below) and the app can later
# download a faster flavor. Because torch isn't in the app's static
# import graph, the frozen interpreter would normally omit the
# stdlib modules torch imports; we compensate by freezing the FULL
# stdlib as hiddenimports (see below) so ANY downloadable flavor's
# import chain resolves.
#
# This spec optimises for "it builds and runs" rather than
# minimal binary size. If the bundle gets too large, candidates
# for trimming live in ``excludes`` below — but verify each
# removal against an end-to-end render before shipping.
#
# Run via ``pwsh package.ps1`` at the repo root, which orchestrates
# the full installer build.

# noqa: E402 -- spec files run as configuration scripts; layout matters
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

block_cipher = None

# ``backend/`` — this spec lives here, so __file__'s parent is the
# backend root. PyInstaller invokes the spec with cwd at the spec's
# directory.
BACKEND_DIR = Path.cwd()
SRC_DIR = BACKEND_DIR / "src"
WORKFLOWS_DIR = BACKEND_DIR / "workflows"

# Core Lucidium package — collect everything under ``lucidium``
# explicitly so dynamically-imported submodules (e.g.
# ``lucidium.orchestration.prompts.world_refresh``) are picked up.
hidden = list(collect_submodules("lucidium"))

# Heavy ML stack — diffusers + torch + transformers all use
# late-import patterns that PyInstaller's static analyser misses.
# ``collect_submodules`` is expensive but reliable; the alternative
# is whack-a-mole on missing imports at runtime.
#
# IMPORTANT subtlety with ``collect_submodules``: when ANY submodule
# of the target package raises an unhandled ``ModuleNotFoundError``
# during walk (e.g. ``rembg.commands`` requires ``click`` which we
# don't ship; ``onnxruntime.quantization`` requires ``onnx``), the
# whole collection step warns + falls through to an empty result.
# That silently loses the rest of the package — the FIRST run
# shipped without ``rembg`` and ``pooch`` for exactly this reason
# and the packaged app's character renders kept their backgrounds
# (no RMBG cut-out). For packages where partial collection is
# unsafe we use ``collect_all`` (submodules + data + binaries +
# metadata in one pass) and explicitly extend the lists below.
for _pkg in (
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "tokenizers",
    "PIL",
    "soundfile",
    "onnxruntime",
):
    try:
        hidden.extend(collect_submodules(_pkg))
    except Exception:  # noqa: BLE001 -- pkg may not be installed
        pass

# Several deps need ``collect_all`` (submodules + data + binaries
# + metadata in one pass) because their public surface includes
# data files, .pyd C extensions, and Python sources that
# ``collect_submodules`` alone misses:
#
#   * rembg — bundles a default U2NET ONNX model under
#     ``rembg/sessions/`` plus ``checksum.json``.
#   * pooch — rembg's HF downloader; mix of code + data.
#   * skimage (scikit-image) — used by rembg.bg's morphology
#     ops. ``collect_submodules`` was missing the .py sources
#     for some morphology submodules, leaving only the .pyd
#     compiled accelerators behind so ``from skimage.morphology
#     import disk, opening`` failed at runtime, which cascaded
#     into ``rembg.bg`` failing to import, which made the
#     embedded backend silently skip background removal.
_extra_datas = []
_extra_binaries = []
# rembg's ``sessions/__init__.py`` eagerly imports EVERY session
# class — SAM, BiRefNet, U2Net, etc. — so a single transitive
# missing dep on a session we don't even use cascades into
# ``from rembg import remove`` failing entirely. The bundle below
# pulls everything any session touches: jsonschema (SAM),
# pymatting+numba+llvmlite (mask-to-alpha), pooch (model
# downloader), skimage (morphology ops in bg.py).
for _pkg in (
    "rembg",
    "pooch",
    "skimage",
    "pymatting",
    "numba",
    "llvmlite",
    "jsonschema",
    "jsonschema_specifications",
    "referencing",
    "attrs",
    "rpds",
    # rfc3987_syntax is pulled in by jsonschema's IRI format
    # checker (which rembg's eager session loader transitively
    # imports). It ships ``syntax_rfc3987.lark`` next to its
    # Python source — the .lark file is loaded at import time
    # (NOT lazily) so a missing-data error here raises
    # FileNotFoundError, NOT ImportError, which means
    # jsonschema's ``suppress(ImportError)`` around the rfc3987
    # import fails to catch it and the whole rembg chain
    # implodes. ``collect_all`` includes the data file.
    "rfc3987_syntax",
    "lark",
    # Output-side ML content filter (SAFETY.md §3). These are NOT
    # optional for the shipped artifact: SAFETY.md states the filter
    # is always on, so the frozen bundle MUST contain them or that
    # claim is false. ``collect_all`` (not ``collect_submodules``)
    # because both ship data next to their Python sources:
    #   * nudenet — bundles its detector ONNX under
    #     ``nudenet/*.onnx``; ``collect_submodules`` alone leaves the
    #     package importable but modelless, which fails at first
    #     ``NudeDetector()`` construction rather than at import.
    #   * insightface — ships ``.pyx``-derived C extensions plus data
    #     under ``insightface/data/``; its model bundle (buffalo_l)
    #     is a separate runtime download and is bundled explicitly
    #     further down (see ``_insightface_src``).
    "nudenet",
    "insightface",
):
    try:
        pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(_pkg)
        _extra_datas.extend(pkg_datas)
        _extra_binaries.extend(pkg_binaries)
        hidden.extend(pkg_hiddenimports)
    except Exception:  # noqa: BLE001 -- pkg may not be installed
        pass

# torch is loaded from a runtime overlay (excluded from the freeze; see
# ``excludes`` below). The overlay ships torch's *own* python + DLLs, but
# torch's import chain also pulls STDLIB modules that the frozen
# interpreter would otherwise omit — PyInstaller only freezes stdlib
# modules reachable from the APP's own static import graph, and torch's
# imports are invisible to that analysis because torch isn't in the
# graph at all.
#
# The spike hand-maintained a list of the specific stdlib modules torch
# touched (pickletools, cProfile, ...). That list is FRAGILE: the
# downloadable overlay can be ANY flavor (cuda / rocm / directml / xpu /
# cpu), and different flavors exercise different stdlib modules at import
# time (e.g. a DirectML build pokes at modules a CPU build never does).
# A frozen stdlib that's a subset of what *some* flavor needs fails only
# on that flavor, in the field, with a bare ModuleNotFoundError — exactly
# the kind of bug that never shows up in our own CPU smoke test.
#
# Robust fix: make the frozen stdlib a SUPERSET of anything any torch
# flavor could import. We seed ``hiddenimports`` with EVERY name in
# ``sys.stdlib_module_names``, filtered to (a) actually importable on
# this interpreter and (b) not one of the packages we deliberately drop
# in ``excludes`` (tkinter et al). This trades a modest size bump (pure-
# python stdlib is small relative to the ML data files we already ship)
# for not having to predict which stdlib corner each GPU flavor steps on.
#
# ``sys.stdlib_module_names`` includes private/underscore names and a few
# platform-specific or import-time-side-effecting modules; we skip the
# obvious troublemakers and anything that fails to import here so the
# freeze analysis never trips on a module that can't load on this host.
_STDLIB_SKIP = {
    # Covered by / conflicts with ``excludes`` below — don't re-add.
    "tkinter",
    "turtle",
    "turtledemo",
    "idlelib",
    "lib2to3",
    "antigravity",  # opens a web browser on import — never freeze it.
    "this",         # prints the Zen of Python on import.
    "__main__",     # not a real importable stdlib module in this sense.
}
import importlib as _importlib

_stdlib_hidden: list[str] = []
for _name in sorted(sys.stdlib_module_names):
    if _name in _STDLIB_SKIP:
        continue
    # Skip dunder/private top-levels we never need and that can have
    # import side-effects or be C-bootstrap-only (e.g. ``_bootlocale``).
    # Underscore-prefixed C accelerators that real stdlib modules import
    # (``_socket`` for ``socket`` etc.) are pulled in transitively when
    # their public module is added, so we don't need to add them by name.
    if _name.startswith("_"):
        continue
    try:
        _importlib.import_module(_name)
    except Exception:  # noqa: BLE001 -- not importable on this host; skip
        continue
    _stdlib_hidden.append(_name)
hidden.extend(_stdlib_hidden)

# Pydantic v2 has C-accelerated bits (``pydantic_core``) that
# PyInstaller picks up by default, but its model-rebuild path
# imports submodules dynamically; collect them.
hidden.extend(collect_submodules("pydantic"))
hidden.extend(collect_submodules("pydantic_core"))

# Package data: workflows JSONs ship next to the EXE so the
# engine's resolver finds them at the same relative path it
# uses in dev (``backend/workflows/*.json``).
#
# OpenCV's haarcascade XMLs ship under ``cv2/data/`` in the
# wheel, but ``collect_data_files("cv2")`` only picks up the
# Python sources — the XMLs sit on a non-package path that
# PyInstaller skips by default. Bundle them explicitly so the
# face-detail bbox detector can load
# ``haarcascade_frontalface_default.xml`` at runtime.
datas = [
    (str(WORKFLOWS_DIR), "workflows"),
] + _extra_datas

# Bundle rembg's U2NET ONNX model so the packaged build never
# reaches out to github.com/danielgatis/rembg/releases for it
# at runtime — that download is ~170 MB, gated behind the user's
# network, and silently falls back to "no background removal"
# when the URL fails. Source path: rembg's default download
# cache at ``%USERPROFILE%/.u2net``. ``package.ps1`` warms this
# cache before invoking PyInstaller (via a one-shot
# pooch.retrieve) so the file is reliably present.
import os as _os_module  # local alias; the spec reuses the stdlib

_u2net_src = Path(
    _os_module.environ.get("U2NET_HOME")
    or Path.home() / ".u2net",
)
if (_u2net_src / "u2net.onnx").exists():
    datas.append((str(_u2net_src / "u2net.onnx"), "u2net_models"))

# Bundle insightface's ``buffalo_l`` model pack (detection +
# genderage heads) for the same reason we bundle U2NET: without it
# the first ``FaceAnalysis.prepare()`` call reaches out to
# huggingface/insightface's release URL for ~280 MB, and a failed
# fetch silently degrades the output-side safety filter to
# ``Verdict.unavailable``. SAFETY.md §3 promises the filter is
# always on, so the model ships in the bundle.
#
# ``allowed_modules=("detection", "genderage")`` in content_filter
# means only ``det_10g.onnx`` + ``genderage.onnx`` are ever loaded,
# so we copy just those two out of the pack (~17 MB) instead of the
# whole ~330 MB bundle (which also carries recognition + landmark
# models we never touch). The packaging scripts warm
# ``~/.insightface`` before invoking PyInstaller.
#
# Runtime side: ``content_filter._bundled_insightface_root()``
# probes for ``insightface_models/models/buffalo_l`` next to the
# frozen exe and passes it as ``root=`` to ``FaceAnalysis``.
_INSIGHTFACE_NEEDED = ("det_10g.onnx", "genderage.onnx")
_insightface_src = Path(
    _os_module.environ.get("INSIGHTFACE_HOME")
    or Path.home() / ".insightface",
) / "models" / "buffalo_l"
_if_found = [f for f in _INSIGHTFACE_NEEDED if (_insightface_src / f).exists()]
if len(_if_found) == len(_INSIGHTFACE_NEEDED):
    for _f in _if_found:
        datas.append(
            (str(_insightface_src / _f), "insightface_models/models/buffalo_l"),
        )
    print(f"lucidium.spec: bundling insightface buffalo_l from {_insightface_src}")
else:
    print(
        f"lucidium.spec: WARNING — insightface buffalo_l models missing at "
        f"{_insightface_src} (found {_if_found or 'none'}); the output-side "
        f"content filter will be INACTIVE in this build (SAFETY.md §3)"
    )

# Bundle a pre-unpacked CPU torch+torchvision overlay so the app works
# OFFLINE on first launch with NO download. torch is excluded from the
# freeze (loaded from a runtime overlay), so without this the very first
# render would have no torch at all until the user downloaded a flavor.
# The packaging scripts (``package.ps1`` / ``scripts/package-linux.sh``)
# build this CPU overlay into a staging dir BEFORE invoking PyInstaller
# and point us at it via ``LUCIDIUM_BUNDLED_OVERLAY_DIR``. We copy its
# tree to ``bundled-overlay/cpu/`` inside the frozen bundle; at runtime
# ``torch_overlay.seed_bundled_overlay()`` finds it there (next to the
# frozen exe) and seeds the per-user runtime dir on first run.
#
# Adding a whole unpacked overlay as a single ``(src_dir, dest_dir)``
# tuple makes PyInstaller copy the directory recursively — same mechanism
# the workflows dir uses above. If the env var is unset (e.g. a quick
# dev ``pyinstaller lucidium.spec`` outside the packaging script) we just
# skip it: the build still produces a runnable backend, it just won't
# have an offline torch until the user downloads one.
_bundled_overlay = _os_module.environ.get("LUCIDIUM_BUNDLED_OVERLAY_DIR")
if _bundled_overlay and Path(_bundled_overlay).is_dir():
    datas.append((str(Path(_bundled_overlay)), "bundled-overlay/cpu"))
    print(f"lucidium.spec: bundling CPU torch overlay from {_bundled_overlay}")
else:
    print(
        "lucidium.spec: LUCIDIUM_BUNDLED_OVERLAY_DIR unset/missing — building "
        "WITHOUT a baked CPU overlay (first render will need a torch download)"
    )
try:
    import cv2 as _cv2  # noqa: F401 -- import for side-effects (resolves data path)
    _cv2_data_dir = Path(_cv2.data.haarcascades)
    if _cv2_data_dir.is_dir():
        # ``include_py_files=False`` to avoid duplicating .py
        # files that ``collect_submodules('cv2')`` already
        # bundles via Analysis.
        datas.append((str(_cv2_data_dir), "cv2/data"))
except Exception:  # noqa: BLE001 -- cv2 may not be installed
    pass

binaries = [] + _extra_binaries

# Diffusers / transformers ship JSON config files next to their
# Python modules; ``collect_data_files`` walks those.
for _pkg in (
    "diffusers",
    "transformers",
    "huggingface_hub",
    "tokenizers",
    "rembg",
    "onnxruntime",
    "nudenet",
    "insightface",
):
    try:
        datas.extend(collect_data_files(_pkg))
    except Exception:  # noqa: BLE001 -- pkg may not be installed
        pass

# Some libs (notably huggingface_hub, transformers) read their
# version via ``importlib.metadata.version`` at runtime — the
# metadata directories must be bundled.
for _pkg in (
    "huggingface_hub",
    "transformers",
    "diffusers",
    "accelerate",
    "safetensors",
    "tokenizers",
    # SPIKE: torch metadata intentionally NOT copied — torch is
    # excluded from the freeze and loaded from a runtime overlay
    # dir on sys.path instead. Its dist-info ships in the overlay.
    # "torch",
    # rembg's transitive deps that read their own version via
    # importlib.metadata at import time. Without metadata bundled
    # the import chain explodes with "No package metadata was
    # found for pymatting" and rembg falls back to the no-op
    # path, leaving character portraits with their backgrounds
    # intact.
    "rembg",
    "pymatting",
    "pooch",
    "scikit-image",
    "numba",
    "llvmlite",
    "onnxruntime",
    # Output-side content filter deps — insightface reads its own
    # version via importlib.metadata during model-zoo resolution.
    "nudenet",
    "insightface",
):
    try:
        datas.extend(copy_metadata(_pkg))
    except Exception:  # noqa: BLE001 -- pkg may not be installed
        pass

# Excludes — packages that get pulled in transitively but the
# engine never actually invokes. Trimming them keeps the bundle
# smaller. If a runtime ``ModuleNotFoundError`` blames any of
# these, remove from this list.
excludes = [
    # SPIKE: torch is NOT frozen into the bundle. It is loaded at
    # runtime from a writable overlay dir injected onto sys.path by
    # lucidium_pyi_entry.py (LUCIDIUM_TORCH_OVERLAY). Excluding it
    # here ensures PyInstaller's frozen importer does not resolve
    # it, so the import falls through to the sys.path overlay entry.
    # torchgen is torch's build-time codegen package — never needed
    # at runtime but collect_submodules can drag it in.
    "torch",
    "torchvision",
    "torchgen",
    "tkinter",        # Tk GUI — not used.
    "matplotlib",     # Optional dep of various ML libs; we don't plot.
    "IPython",        # Interactive shell.
    "jupyter",
    "notebook",
    "pytest",         # Test runner — bundle would never be tested in-app.
    "scipy.weave",    # Old SciPy weave compiler.
]


# Use the shim at the backend root rather than ``lucidium/app.py``
# directly: PyInstaller treats Analysis's first arg as a top-level
# script, which breaks ``from .api import ...`` style relative
# imports inside ``app.py``. The shim is a plain
# ``from lucidium.app import main`` so PyInstaller's script
# analysis sees it as an importer of the package, ``pathex``
# below makes ``lucidium`` importable, and app.py's relative
# imports resolve cleanly.
a = Analysis(
    [str(BACKEND_DIR / "lucidium_pyi_entry.py")],
    pathex=[str(SRC_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Scrub ALL frozen torch / torchvision / torchgen artifacts from the
# bundle — both Python/data files AND the giant ``torch/lib/*`` native
# DLLs.
#
# WHY this is needed even though ``excludes`` lists torch: ``excludes``
# only removes torch from PyInstaller's *module* graph (so ``import
# torch`` falls through to the sys.path overlay). It does NOT stop torch's
# files from being COLLECTED as transitive data/binaries: ``collect_all``
# of diffusers/transformers/onnxruntime walks their dependency trees and
# scoops up torch's ``*.dist-info`` (metadata) and — far worse — the
# multi-GiB ``torch/lib/*.dll`` CUDA blobs (cublas, cudnn, ...) as
# "binaries". Left in, those DLLs balloon the bundle by ~2.7 GiB, which
# DEFEATS the entire point of the overlay (the overlay ships its own
# ``torch/lib`` matched to the chosen flavor).
#
# Two harms, both fixed here by dropping the entries:
#   1. SIZE: the frozen ``torch/lib`` DLLs are dead weight — torch isn't
#      even imported from the freeze. Removing them is the size win.
#   2. METADATA SHADOWING: a frozen ``torch-<ver>.dist-info`` makes
#      ``importlib.metadata.version("torch")`` report the BUILD-TIME
#      version, not the overlay's actually-loaded one, so version-gated
#      code in diffusers/transformers decides against a phantom torch.
#
# We anchor on the top-level path segment so we only drop torch's OWN
# files: ``torch/...``, ``torchvision/...``, ``torchgen/...``, and the
# ``torch*-<ver>.dist-info`` dirs — NOT unrelated names like ``torchsde``
# and NOT our bundled overlay (its dest paths start ``bundled-overlay/``).
#
# SEPARATELY, PyInstaller's binary-dependency analysis HOISTS torch's
# native DLLs (the ones that live in ``torch/lib/*.dll`` inside the wheel)
# LOOSE into the bundle ROOT — dest paths like ``torch_cpu.dll`` (~293 MB),
# ``c10.dll``, ``shm.dll``, ``uv.dll`` with NO directory prefix. This
# happens because some collected package links against those DLLs, so
# PyInstaller treats them as shared libs and flattens them to the root.
# The package-segment filter above CANNOT catch these — their first path
# segment is the bare filename (e.g. ``torch_cpu.dll``), not ``torch``.
# They are byte-identical to the copies our CPU overlay already ships under
# ``bundled-overlay/cpu/torch/lib/`` and NOTHING in the freeze loads torch
# from the bundle root (the torch package code is scrubbed above), so these
# loose DLLs are ~296 MB of pure dead weight. Drop them by BASENAME, but
# ONLY when the entry sits at the bundle root (or under a torch package
# path) — NEVER when it lives under ``bundled-overlay/`` (the intentional
# CPU overlay, which MUST stay intact for offline first-run torch).
_TORCH_LOOSE_DLLS = {
    "torch_cpu.dll",
    "torch_python.dll",
    "torch_global_deps.dll",
    "c10.dll",
    "c10_cuda.dll",
    "shm.dll",
    "uv.dll",
    "asmjit.dll",
    "fbgemm.dll",
    "libiomp5md.dll",
    "libiompstubs5md.dll",
}


def _is_loose_torch_dll(dest_norm: str) -> bool:
    base = dest_norm.rsplit("/", 1)[-1].lower()
    if not base.endswith(".dll"):
        return False
    if base in _TORCH_LOOSE_DLLS:
        return True
    # ``torch_*.dll`` (e.g. torch_cpu, torch_python, torch_cuda) and
    # ``caffe2_*.dll`` (legacy caffe2 GPU observers shipped in torch/lib)
    # — narrow prefix patterns so we never sweep up unrelated DLLs.
    if base.startswith("torch_") or base.startswith("caffe2_"):
        return True
    return False


def _is_frozen_torch_artifact(dest: str) -> bool:
    dest_norm = dest.replace("\\", "/")
    # SAFETY GUARD: the bundled CPU overlay legitimately contains every one
    # of these torch DLLs under ``bundled-overlay/cpu/torch/lib/``. It is
    # the whole point of the overlay design — leave it completely untouched.
    if dest_norm.lower().startswith("bundled-overlay/"):
        return False
    head = dest_norm.split("/", 1)[0].lower()
    # Exact torch package dirs (torchgen ships INSIDE the torch wheel).
    if head in ("torch", "torchvision", "torchgen", "functorch"):
        return True
    # ``torch-<ver>.dist-info`` / ``torchvision-<ver>.dist-info``.
    if head.endswith(".dist-info"):
        name = head[: -len(".dist-info")].rsplit("-", 1)[0]
        return name in ("torch", "torchvision")
    # Loose torch native DLLs hoisted to the bundle root (or sitting under a
    # torch package path) — guarded above so this never touches the overlay.
    if _is_loose_torch_dll(dest_norm):
        return True
    return False


_b_before = len(a.binaries)
a.binaries = [entry for entry in a.binaries if not _is_frozen_torch_artifact(entry[0])]
_d_before = len(a.datas)
a.datas = [entry for entry in a.datas if not _is_frozen_torch_artifact(entry[0])]
_scrubbed = (_b_before - len(a.binaries)) + (_d_before - len(a.datas))
if _scrubbed:
    print(
        f"lucidium.spec: scrubbed {_scrubbed} frozen torch/torchvision artifact(s) "
        f"({_b_before - len(a.binaries)} binaries, {_d_before - len(a.datas)} datas) "
        f"— includes loose root torch native DLLs (torch_cpu.dll, c10.dll, ...)"
    )

# --- Visual C++ runtime consistency (Windows) --------------------------
#
# PyInstaller harvests the VC++ runtime DLLs from whichever dependency
# happens to carry them. Wheels ship wildly different vintages, so the
# bundle ended up with msvcp140/vcruntime140/vcruntime140_1 from VS2019
# (14.29) sitting next to MSVCP140_1/MSVCP140_ATOMIC_WAIT from VS2022
# (14.51). That mix is not supported: the runtime-overlay torch built
# against the newer runtime then fails to load with
#
#   OSError: [WinError 1114] A dynamic link library (DLL) initialization
#   routine failed. Error loading "...\\overlays\\cpu\\torch\\lib\\c10.dll"
#
# because the bundle dir is searched before System32, so c10.dll binds
# the stale 14.29 msvcp140. Normalise: for each member of the runtime
# family, keep only the highest-versioned copy available (bundled
# candidates plus the host's System32 copy).
if sys.platform == "win32":
    _VCRT_NAMES = {
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll",
        "msvcp140_codecvt_ids.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "concrt140.dll",
    }

    def _dll_version(path):
        """(major, minor, build, revision) from a DLL, or None."""
        try:
            import ctypes
            from ctypes import wintypes

            size = ctypes.windll.version.GetFileVersionInfoSizeW(path, None)
            if not size:
                return None
            buf = ctypes.create_string_buffer(size)
            if not ctypes.windll.version.GetFileVersionInfoW(path, 0, size, buf):
                return None
            ptr = ctypes.c_void_p()
            length = wintypes.UINT()
            if not ctypes.windll.version.VerQueryValueW(
                buf, "\\", ctypes.byref(ptr), ctypes.byref(length)
            ):
                return None
            ffi = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32 * 4)).contents
            ms_hi, ms_lo = ffi[2] >> 16, ffi[2] & 0xFFFF   # FileVersionMS
            ls_hi, ls_lo = ffi[3] >> 16, ffi[3] & 0xFFFF   # FileVersionLS
            return (ms_hi, ms_lo, ls_hi, ls_lo)
        except Exception:
            return None

    _vcrt_best = {}
    for _entry in a.binaries:
        _name = _os_module.path.basename(_entry[0]).lower()
        if _name not in _VCRT_NAMES:
            continue
        _cands = [_entry[1]]
        _sys32 = _os_module.path.join(
            _os_module.environ.get("SystemRoot", r"C:\Windows"), "System32", _name
        )
        if _os_module.path.isfile(_sys32):
            _cands.append(_sys32)
        _best = max(
            _cands, key=lambda p: (_dll_version(p) or (0, 0, 0, 0), p == _cands[0])
        )
        if _best != _entry[1]:
            _vcrt_best[_entry[1]] = _best
            print(
                f"lucidium.spec: VC++ runtime {_name}: using "
                f"{_dll_version(_best)} from {_best} instead of "
                f"{_dll_version(_entry[1])} from {_entry[1]}"
            )
    if _vcrt_best:
        a.binaries = [
            (dest, _vcrt_best.get(src, src), *rest) for dest, src, *rest in a.binaries
        ]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="lucidium-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX breaks several ML deps' DLLs.
    console=True,              # Backend prints port to stdout — keep console.
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="lucidium-backend",   # Output dir: backend/dist/lucidium-backend/
)
