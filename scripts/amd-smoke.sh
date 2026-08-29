#!/usr/bin/env bash
# Run the Lucidium GPU smoke test on Linux.
#
# Finds the project's backend venv Python and runs amd_smoke_test.py.
# Downloads the GPU torch build your machine needs (ROCm for AMD),
# loads it, and proves the GPU computes. Writes a report file under
# scripts/ that you send back to the developer.
#
# Usage:
#   scripts/amd-smoke.sh
#   scripts/amd-smoke.sh --checkpoint /path/to/sdxl.safetensors
#   scripts/amd-smoke.sh --flavor rocm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Prefer the Linux venv, then the generic venv, then python3 on PATH.
PY=""
for candidate in \
    "$REPO_ROOT/backend/.venv-linux/bin/python" \
    "$REPO_ROOT/backend/.venv/bin/python" \
    "$(command -v python3 || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then PY="$candidate"; break; fi
done
if [[ -z "$PY" ]]; then
    echo "Could not find the project's Python. Set the backend up first" >&2
    echo "(scripts/package-linux.sh or your venv) and re-run." >&2
    exit 1
fi

# AMD note: the Radeon 6700 XT and other RDNA cards that ROCm doesn't
# officially list need an architecture override. Lucidium's launcher
# sets this automatically; we set the common RDNA2 value here too so the
# smoke test matches what the app would do. Harmless on NVIDIA/Intel.
if [[ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
fi

echo "Running: $PY $SCRIPT_DIR/amd_smoke_test.py $*"
exec "$PY" "$SCRIPT_DIR/amd_smoke_test.py" "$@"
