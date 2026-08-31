"""PyInstaller entry shim.

PyInstaller's Analysis treats its first arg as a top-level script,
which means a script that uses ``from .x import y`` (relative
imports) fails to load: there's no parent package to anchor the
``.``. Pointing Analysis at ``lucidium/app.py`` directly tripped
exactly that — the bundled backend died on launch with
``ImportError: attempted relative import with no known parent
package``.

This file is the pure-import wrapper. It lives at the BACKEND root
(not inside the ``lucidium/`` package) so PyInstaller's script-
analysis pass treats ``lucidium`` as an importable package
discovered via ``pathex`` — which means ``app.py``'s relative
imports resolve normally.

Don't reuse this for anything else; it exists solely so the spec
has a well-formed entry point.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Runtime torch overlay injection.
#
# Torch (and torchvision) are NOT frozen into the PyInstaller bundle
# (see ``excludes`` in lucidium.spec). They are instead loaded from a
# writable "overlay" directory placed on sys.path here, BEFORE any
# import that could transitively pull in torch. This lets the app
# download the GPU-correct torch flavor (CUDA/ROCm/DirectML) after
# probing the user's GPU on first run.
#
# This block MUST run before ``from lucidium.app import main`` (which
# transitively imports diffusers/torch). It also MUST stay tiny and
# import nothing heavy — no lucidium app modules, no torch. So instead
# of importing ``lucidium.providers.torch_overlay`` (which, while
# stdlib-only today, could grow a heavier transitive import and would
# silently break a frozen launch), we DUPLICATE the minimal runtime-dir
# + pointer-file resolution inline. ``torch_overlay.py`` is the single
# source of truth for the layout; a unit test
# (``test_torch_overlay_entry_shim``) asserts this copy stays in
# agreement with it, so the duplication can't drift.
#
# Resolution order:
#   1. ``LUCIDIUM_TORCH_OVERLAY`` (dev/test override naming a resolved
#      overlay dir directly) — honored as-is when it's a real dir.
#   2. The active overlay computed from the per-user runtime dir + the
#      ``active_overlay`` pointer file.
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys


def _lucidium_runtime_dir() -> str:
    """Per-user runtime dir. MUST match ``torch_overlay.runtime_dir()``.

    Windows ``%LOCALAPPDATA%\\Lucidium\\runtime``; macOS
    ``~/Library/Application Support/Lucidium/runtime``; Linux
    ``${XDG_DATA_HOME:-~/.local/share}/lucidium/runtime``. Honors
    ``LUCIDIUM_RUNTIME_DIR`` for tests/dev.
    """
    override = _os.environ.get("LUCIDIUM_RUNTIME_DIR")
    if override:
        return override
    home = _os.path.expanduser("~")
    if _sys.platform == "win32":
        base = _os.environ.get("LOCALAPPDATA") or _os.path.join(home, "AppData", "Local")
        return _os.path.join(base, "Lucidium", "runtime")
    if _sys.platform == "darwin":
        return _os.path.join(home, "Library", "Application Support", "Lucidium", "runtime")
    base = _os.environ.get("XDG_DATA_HOME") or _os.path.join(home, ".local", "share")
    return _os.path.join(base, "lucidium", "runtime")


def _lucidium_active_overlay() -> str | None:
    """Resolve the active overlay dir from the pointer file, or None.

    MUST match ``torch_overlay.active_overlay_path()``.
    """
    rt = _lucidium_runtime_dir()
    pointer = _os.path.join(rt, "active_overlay")
    try:
        with open(pointer, encoding="utf-8") as _fh:
            flavor = _fh.read().strip()
    except OSError:
        return None
    if not flavor:
        return None
    overlay = _os.path.join(rt, "overlays", flavor)
    return overlay if _os.path.isdir(overlay) else None


_overlay = _os.environ.get("LUCIDIUM_TORCH_OVERLAY")
if not (_overlay and _os.path.isdir(_overlay)):
    _overlay = _lucidium_active_overlay()

# First-run seed: if NO overlay is active yet AND this is a frozen build
# that baked a CPU overlay into the bundle, copy it into the runtime dir
# and activate it so image generation works OFFLINE, out of the box, with
# NO download. This MUST happen here — BEFORE the sys.path prepend below
# and before ``from lucidium.app import main`` (which can transitively
# import torch on first render). The shim is otherwise import-light; we
# only reach into ``torch_overlay`` (whose top level is pure stdlib) on the
# cold path where there's no active overlay, and torch is NOT imported yet.
if not (_overlay and _os.path.isdir(_overlay)) and getattr(_sys, "frozen", False):
    try:
        from lucidium.providers.torch_overlay import seed_bundled_overlay

        if seed_bundled_overlay() is not None:
            _overlay = _lucidium_active_overlay()
    except Exception:
        pass

if _overlay and _os.path.isdir(_overlay):
    _sys.path.insert(0, _overlay)


# Support diagnostic: ``LUCIDIUM_CHECK_TORCH=1 lucidium-backend(.exe)``
# resolves the active overlay (after any first-run seed above), imports
# torch, prints where it loaded from + its version, and exits — WITHOUT
# starting the server. Legitimately useful for support ("is torch actually
# loading from the overlay, and which one?") and for the build's own smoke
# test. Guarded behind an env var so a normal launch never pays for it.
if _os.environ.get("LUCIDIUM_CHECK_TORCH"):
    try:
        import torch

        print(f"TORCH_FILE={getattr(torch, '__file__', '?')}")
        print(f"TORCH_VERSION={getattr(torch, '__version__', '?')}")
        print(f"TORCH_OVERLAY={_overlay or '(none)'}")
        try:
            import importlib.metadata as _md

            print(f"TORCH_METADATA_VERSION={_md.version('torch')}")
        except Exception as _e:
            print(f"TORCH_METADATA_VERSION=error:{_e}")
        _sys.exit(0)
    except Exception as _e:
        print(f"TORCH_IMPORT_ERROR={_e}")
        _sys.exit(1)

# ---------------------------------------------------------------------------
# ``-m <module>`` dispatch.
#
# The PyInstaller bootloader does NOT interpret CPython's command-line
# flags. ``lucidium-backend.exe -m lucidium.providers.gpu_selftest`` runs
# THIS shim and hands ``-m ...`` straight through as ``sys.argv``. Without
# the dispatch below, the frozen exe answers that invocation by starting
# the whole server -- which runs the GPU self-test, which spawns
# ``sys.executable -m lucidium.providers.gpu_selftest``, which starts
# another server. The recursion is unbounded: a single launch of the
# packaged build was observed fanning out to 104 lucidium-backend.exe
# processes holding ~163 GB of committed memory, at which point unrelated
# work on the machine started failing with "the paging file is too small".
#
# Dev runs never hit this -- there ``sys.executable`` is a real python
# that handles ``-m`` itself -- which is why it only ever showed up in a
# packaged build.
#
# Only ``lucidium.*`` modules are dispatchable. This is an application
# entry point, not a general-purpose interpreter, and a shipped build
# should not run an arbitrary module because someone passed a flag.
# ---------------------------------------------------------------------------
if len(_sys.argv) >= 3 and _sys.argv[1] == "-m" and _sys.argv[2].startswith("lucidium."):
    import runpy as _runpy

    _module = _sys.argv[2]
    _sys.argv = [_module, *_sys.argv[3:]]
    _runpy.run_module(_module, run_name="__main__", alter_sys=True)
    _sys.exit(0)


from lucidium.app import main  # noqa: E402

if __name__ == "__main__":
    main()
