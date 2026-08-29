#!/usr/bin/env bash
# Build a Linux AppImage for Lucidium.
#
# Mirror of ``package.ps1`` for Linux hosts (including WSL2). The
# Windows-side entry point is ``pwsh package.ps1 -Linux``, which
# dispatches to this script via ``wsl.exe bash scripts/package-linux.sh``
# so contributors on Windows can produce a Linux release without
# leaving PowerShell.
#
# Pipeline (mirrors package.ps1):
#
#   1. Verify python3, node, npm are on PATH.
#   2. Set up ``backend/.venv-linux/`` (kept separate from the
#      Windows ``backend/.venv/`` so a Windows-side dev install is
#      not clobbered when this script runs from WSL against the
#      same checkout).
#   3. ``pip install -e backend[dev]`` + pyinstaller.
#   4. Warm rembg's U2NET ONNX cache so PyInstaller can bundle it.
#   5. Build the bundled CPU torch overlay: download + unpack the CPU
#      torch/torchvision wheels (matched to the venv's Python) into a
#      build staging dir. torch is NOT frozen into the bundle anymore;
#      this pre-unpacked CPU overlay is baked into the frozen tree so
#      image generation works OFFLINE on first launch with NO download.
#      A user with a GPU can later download a faster flavor from the app.
#   6. PyInstaller against ``backend/lucidium.spec`` (torch EXCLUDED;
#      the CPU overlay from step 5 is handed to the spec via the
#      ``LUCIDIUM_BUNDLED_OVERLAY_DIR`` env var). Output:
#      ``backend/dist/lucidium-backend/lucidium-backend`` (Linux ELF;
#      same relative path as the .exe on Windows, so
#      ``frontend/package.json``'s extraResources block picks it up
#      unchanged).
#   7. Frontend build: npm install, codegen, vite build, electron
#      main+preload tsc.
#   8. ``electron-builder --linux AppImage``. Output:
#      ``frontend/release/Lucidium-*.AppImage``.
#
# CAVEAT — running from WSL against /mnt/c:
#   * The bind-mounted Windows filesystem is much slower than the
#     Linux native filesystem for npm install / pip install.
#     Expect the first run to take significantly longer than the
#     equivalent native run. Subsequent runs are faster because
#     both the venv and node_modules are cached.
#   * ``frontend/node_modules`` will be repopulated with Linux
#     binaries by this script. If you alternate between Windows and
#     WSL builds against the same checkout, you must re-run
#     ``npm install`` from each host before that host's dev
#     tooling works. The Python venvs are kept separate via the
#     ``.venv`` (Windows) / ``.venv-linux`` (Linux) split, but
#     node_modules has no analogous separation in npm. A printed
#     warning at the end calls this out.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/backend"
FRONTEND_DIR="$REPO_ROOT/frontend"
VENV_DIR="$BACKEND_DIR/.venv-linux"
VENV_PY="$VENV_DIR/bin/python"

SKIP_BACKEND=0
SKIP_FRONTEND=0
FORCE=0
CLEAN=0
# Set by build_cpu_overlay; consumed by invoke_backend_build's spec env.
BUNDLED_OVERLAY_DIR=""

usage() {
    cat <<EOF
Usage: $0 [--skip-backend] [--skip-frontend] [--force] [--clean]

  --skip-backend   Skip PyInstaller; reuse the existing
                   backend/dist/lucidium-backend/ output.
  --skip-frontend  Skip the npm + electron-builder steps.
  --force          Wipe build artefacts before each step.
  --clean          Wipe build artefacts and exit (no rebuild).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-backend) SKIP_BACKEND=1 ;;
        --skip-frontend) SKIP_FRONTEND=1 ;;
        --force) FORCE=1 ;;
        --clean) CLEAN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

write_section() {
    printf '\n==> %s\n' "$1"
}

assert_tool() {
    local tool="$1"
    local hint="$2"
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Required tool '$tool' not found on PATH. $hint" >&2
        exit 1
    fi
}

invoke_clean() {
    write_section "Cleaning build artefacts"
    local paths=(
        "$BACKEND_DIR/build"
        "$BACKEND_DIR/dist"
        "$FRONTEND_DIR/dist"
        "$FRONTEND_DIR/dist-electron"
        "$FRONTEND_DIR/release"
    )
    for p in "${paths[@]}"; do
        if [[ -e "$p" ]]; then
            echo "  removing $p"
            rm -rf "$p"
        fi
    done
}

initialise_venv() {
    if [[ -x "$VENV_PY" ]]; then
        echo "  backend venv present at $VENV_DIR"
        return
    fi
    write_section "Creating Linux venv at $VENV_DIR"
    # Use the highest python3.x available; the project requires >= 3.11.
    local pybin=""
    for candidate in python3.13 python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            pybin="$candidate"
            break
        fi
    done
    if [[ -z "$pybin" ]]; then
        echo "No python3 (>=3.11) on PATH." >&2
        exit 1
    fi
    "$pybin" -m venv "$VENV_DIR"
}

install_backend() {
    # Idempotent: only reinstall when lucidium isn't importable.
    if "$VENV_PY" -c "import lucidium, pydantic" >/dev/null 2>&1; then
        echo "  lucidium + pydantic already installed in venv"
    else
        write_section "Installing backend (with dev + safety extras)"
        "$VENV_PY" -m pip install --upgrade pip wheel
        "$VENV_PY" -m pip install -e "$BACKEND_DIR[dev,safety]"
    fi
    # The ``safety`` extra carries the output-side content filter
    # (nudenet + insightface). SAFETY.md §3 documents that filter as
    # always-on, so the frozen bundle MUST contain it — install it
    # even when the venv predates this change and lucidium already
    # imports (the branch above would otherwise skip it forever).
    if ! "$VENV_PY" -c "import nudenet, insightface" >/dev/null 2>&1; then
        write_section "Installing safety extras (output-side content filter)"
        "$VENV_PY" -m pip install -e "$BACKEND_DIR[safety]"
        "$VENV_PY" -c "import nudenet, insightface" || {
            echo "safety extras failed to import after install." >&2
            exit 1
        }
    fi
    # PyInstaller is a build-time dep, pinned exactly in the
    # ``packaging`` extra. lucidium.spec depends on PyInstaller
    # internals, so we install the pin explicitly instead of accepting
    # whatever release a pre-existing venv happens to carry — otherwise
    # the same commit can bundle differently on two machines. pip is a
    # no-op when the pinned version is already present.
    write_section "Installing pinned PyInstaller (packaging extra)"
    "$VENV_PY" -m pip install --quiet -e "$BACKEND_DIR[packaging]"
    # ``sed -n 1,2p`` rather than ``head -2``: head exits after the
    # second line, pip gets SIGPIPE on flush, and under ``set -o
    # pipefail`` that aborts the whole build with "Pipe to stdout was
    # broken". sed drains stdin, so pip always exits 0.
    "$VENV_PY" -m pip show pyinstaller | sed -n '1,2p'
}

confirm_u2net_cached() {
    write_section "Ensuring U2NET ONNX is cached for the bundle"
    local u2net_home="${U2NET_HOME:-$HOME/.u2net}"
    local target="$u2net_home/u2net.onnx"
    if [[ -f "$target" ]]; then
        local size_mb
        size_mb=$(du -m "$target" | cut -f1)
        echo "  cached: $target (${size_mb} MB)"
        return
    fi
    echo "  fetching u2net.onnx (~170 MB) into $u2net_home..."
    "$VENV_PY" - <<'PY'
from rembg.sessions.u2net import U2netSession
print("downloading to", U2netSession.u2net_home())
U2netSession.download_models()
print("done")
PY
    if [[ ! -f "$target" ]]; then
        echo "u2net.onnx still missing after download attempt at $target" >&2
        exit 1
    fi
}

confirm_content_filter_cached() {
    # Warm the output-side content filter's models before PyInstaller
    # runs, so the spec can copy insightface's buffalo_l detection +
    # genderage ONNX files into the bundle (see lucidium.spec's
    # ``_insightface_src``). nudenet ships its detector inside the
    # wheel, so constructing it here is just a smoke test that the
    # package is usable. SAFETY.md §3 documents this filter as
    # always-on; a bundle without these models would be a doc lie.
    write_section "Ensuring content-filter models are cached for the bundle"
    local if_home="${INSIGHTFACE_HOME:-$HOME/.insightface}"
    local pack="$if_home/models/buffalo_l"
    if [[ -f "$pack/det_10g.onnx" && -f "$pack/genderage.onnx" ]]; then
        echo "  cached: $pack"
    else
        echo "  fetching insightface buffalo_l (~280 MB) into $if_home..."
    fi
    "$VENV_PY" - <<'PY'
from insightface.app import FaceAnalysis
from nudenet import NudeDetector

NudeDetector()
app = FaceAnalysis(name="buffalo_l", allowed_modules=("detection", "genderage"))
app.prepare(ctx_id=-1)
print("content filter models ready")
PY
    if [[ ! -f "$pack/det_10g.onnx" || ! -f "$pack/genderage.onnx" ]]; then
        echo "insightface buffalo_l models still missing under $pack" >&2
        exit 1
    fi
}

build_cpu_overlay() {
    # Build the pre-unpacked CPU torch+torchvision overlay that ships
    # INSIDE the frozen bundle so the app works offline on first launch
    # with NO download. torch is excluded from the freeze (loaded from a
    # runtime overlay), so without a baked CPU overlay the very first
    # render would have no torch at all.
    #
    # Mirror of package.ps1's Build-CpuOverlay: point torch_overlay's
    # runtime dir at a build staging dir and call install_flavor('cpu',
    # activate=False), downloading + unpacking the CPU wheels for THIS
    # interpreter (Python/platform-matched, exactly what the frozen ELF
    # loads). The spec then bundles that dir via
    # LUCIDIUM_BUNDLED_OVERLAY_DIR. Idempotent: a non-empty staged
    # overlay is reused (no re-download) unless --force.
    write_section "Building bundled CPU torch overlay (downloads CPU wheels once)"
    # OS-SPECIFIC staging dir. The Windows build (package.ps1) stages its
    # CPU overlay under ``bundled-overlay-cpu-win``; a shared path let the
    # Windows-first build bake Windows .dll/.lib torch into the Linux
    # AppImage (the reuse check below only tests for a non-empty dir, not
    # which OS's wheels it holds).
    local staging_root="$BACKEND_DIR/build/bundled-overlay-cpu-linux"
    local overlay_dir="$staging_root/overlays/cpu"
    if [[ $FORCE -eq 1 && -d "$staging_root" ]]; then
        rm -rf "$staging_root"
    fi
    if [[ -d "$overlay_dir" ]] && [[ -n "$(ls -A "$overlay_dir" 2>/dev/null)" ]]; then
        echo "  CPU overlay already staged at $overlay_dir"
    else
        LUCIDIUM_RUNTIME_DIR="$staging_root" "$VENV_PY" -c \
            "from lucidium.providers.torch_overlay import install_flavor; r = install_flavor('cpu', activate=False); print('staged CPU overlay at', r.overlay_dir)"
    fi
    if [[ ! -d "$overlay_dir" ]] || [[ -z "$(ls -A "$overlay_dir" 2>/dev/null)" ]]; then
        echo "CPU overlay still missing after build at $overlay_dir" >&2
        exit 1
    fi
    # Exported so invoke_backend_build's PyInstaller picks it up and the
    # spec bakes it into the frozen tree.
    BUNDLED_OVERLAY_DIR="$overlay_dir"
    echo "  bundled CPU overlay: $BUNDLED_OVERLAY_DIR"
}

invoke_backend_build() {
    write_section "Bundling backend with PyInstaller (slow on first run)"
    (
        cd "$BACKEND_DIR"
        # The spec reads LUCIDIUM_BUNDLED_OVERLAY_DIR to bake the CPU
        # overlay into the frozen tree (see build_cpu_overlay + the spec).
        LUCIDIUM_BUNDLED_OVERLAY_DIR="${BUNDLED_OVERLAY_DIR:-}" \
            "$VENV_PY" -m PyInstaller --noconfirm lucidium.spec
    )
    local expected="$BACKEND_DIR/dist/lucidium-backend/lucidium-backend"
    if [[ ! -x "$expected" ]]; then
        echo "PyInstaller finished but $expected is missing or not executable -- spec output paths drifted?" >&2
        exit 1
    fi
    # PyInstaller sets the +x bit, but a previous Windows build may
    # have left files at this path without it (e.g. if the user is
    # alternating builds against /mnt/c). Belt-and-braces:
    chmod +x "$expected"
    echo "  bundled backend at $expected"
}

invoke_frontend_build() {
    write_section "Building frontend (vite + tsc-electron)"
    (
        cd "$FRONTEND_DIR"
        npm install
        npm run codegen
        npm run build
        if [[ -f "tsconfig.electron.json" ]]; then
            echo "  compiling electron main+preload..."
            npx tsc -p tsconfig.electron.json
        fi
    )
}

invoke_electron_builder() {
    write_section "Running electron-builder (--linux AppImage)"
    (
        cd "$FRONTEND_DIR"
        # CSC_IDENTITY_AUTO_DISCOVERY mirrors the Windows path —
        # we don't sign Linux builds either. Setting it explicitly
        # silences electron-builder's "discovering identity" probe.
        CSC_IDENTITY_AUTO_DISCOVERY=false npx electron-builder --linux AppImage
    )
    local release_dir="$FRONTEND_DIR/release"
    local artefact
    artefact=$(ls -t "$release_dir"/Lucidium-*.AppImage 2>/dev/null | head -n1 || true)
    if [[ -z "$artefact" ]]; then
        echo "electron-builder finished but no Lucidium-*.AppImage under $release_dir" >&2
        exit 1
    fi
    chmod +x "$artefact"
    local size_gb
    size_gb=$(du -BG "$artefact" | cut -f1)
    echo ""
    printf '\033[32mAppImage ready at:\033[0m\n'
    printf '\033[32m  %s (%s)\033[0m\n' "$artefact" "$size_gb"
}

# ---- main --------------------------------------------------------------

assert_tool python3 "Install Python (>= 3.11)."
assert_tool node    "Install Node.js (https://nodejs.org)."
assert_tool npm     "Install Node.js (https://nodejs.org)."
assert_tool npx     "Install Node.js (https://nodejs.org)."

if [[ $FORCE -eq 1 || $CLEAN -eq 1 ]]; then
    invoke_clean
fi
if [[ $CLEAN -eq 1 ]]; then
    echo "Clean complete. Run without --clean to rebuild."
    exit 0
fi

if [[ $SKIP_BACKEND -eq 0 ]]; then
    initialise_venv
    install_backend
    confirm_u2net_cached
    confirm_content_filter_cached
    build_cpu_overlay
    invoke_backend_build
else
    echo "Skipping backend bundle (--skip-backend)"
fi

if [[ $SKIP_FRONTEND -eq 0 ]]; then
    invoke_frontend_build
    invoke_electron_builder
else
    echo "Skipping frontend bundle + AppImage (--skip-frontend)"
fi

echo ""
echo "Package complete."
# Heads-up for the alternating-Windows/Linux workflow — npm
# installs into the same node_modules tree on either host, so the
# next Windows dev session needs an ``npm install`` from PowerShell
# to repopulate Windows-native binaries (esbuild's prebuilt etc).
if [[ -d /mnt/c ]]; then
    cat <<'NOTE'

NOTE: this script just rewrote frontend/node_modules/ with Linux
binaries. To resume Windows dev against the same checkout, run
``npm install`` from PowerShell in the frontend/ directory.
NOTE
fi
