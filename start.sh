#!/usr/bin/env bash
# Launch the Lucidium desktop app in dev mode (Linux / WSL2).
#
# Bash mirror of ``start.ps1``. Idempotent first-run setup (venv, npm
# install, schema codegen, Electron main+preload compile), then starts
# the Vite dev server in the background and launches Electron pointing
# at it. Electron's main process spawns the Python backend automatically
# (see frontend/electron/main.ts).
#
# The Linux venv lives at ``backend/.venv-linux/`` -- kept separate from
# the Windows ``backend/.venv/`` so a Windows-side dev install on the
# same checkout (e.g. when alternating between PowerShell and WSL) is not
# clobbered. This mirrors the split used by scripts/package-linux.sh.
#
# Usage: ./start.sh [--setup] [--backend] [--renderer] [--skip-codegen] [--no-setup]
#
#   --setup         Run setup steps and exit without launching the app.
#   --backend       Launch only the Python backend (no Electron, no Vite).
#   --renderer      Launch only the Vite dev server (no Electron, no backend).
#   --skip-codegen  Skip the Pydantic -> JSON Schema -> TypeScript step.
#   --no-setup      Skip the dependency-presence checks (faster restart).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

BACKEND_DIR="$REPO_ROOT/backend"
FRONTEND_DIR="$REPO_ROOT/frontend"
VENV_DIR="$BACKEND_DIR/.venv-linux"
PYTHON_EXE="$VENV_DIR/bin/python"
NODE_MODULES="$FRONTEND_DIR/node_modules"
DIST_ELECTRON="$FRONTEND_DIR/dist-electron"
ELECTRON_BIN="$NODE_MODULES/.bin/electron"
VITE_PORT=5173

DO_SETUP=1
ONLY_SETUP=0
ONLY_BACKEND=0
ONLY_RENDERER=0
SKIP_CODEGEN=0

usage() {
    sed -n '15,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --setup) ONLY_SETUP=1 ;;
        --backend) ONLY_BACKEND=1 ;;
        --renderer) ONLY_RENDERER=1 ;;
        --skip-codegen) SKIP_CODEGEN=1 ;;
        --no-setup) DO_SETUP=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

write_step() { printf '\n\033[36m==> %s\033[0m\n' "$1"; }
write_note() { printf '\033[90m    %s\033[0m\n' "$1"; }

test_required() {
    for tool in python3 node npm; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            echo "Required tool not found on PATH: $tool" >&2
            exit 1
        fi
    done
}

# Prints "major.minor" for the given python executable, or nothing.
python_version() {
    local exe="$1"
    [[ -x "$exe" ]] || return 0
    "$exe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true
}

# Returns 0 if "major.minor" satisfies the constitution's Python >= 3.11.
version_ok() {
    local ver="$1"
    [[ -n "$ver" ]] || return 1
    local major="${ver%%.*}"
    local minor="${ver#*.}"
    [[ "$major" -gt 3 || ( "$major" -eq 3 && "$minor" -ge 11 ) ]]
}

backend_installed() {
    [[ -x "$PYTHON_EXE" ]] || return 1
    "$PYTHON_EXE" -c "import lucidium, pydantic" >/dev/null 2>&1
}

initialise_backend() {
    local needs_venv=0
    if [[ ! -x "$PYTHON_EXE" ]]; then
        needs_venv=1
    elif ! version_ok "$(python_version "$PYTHON_EXE")"; then
        local stale; stale="$(python_version "$PYTHON_EXE")"
        write_step "Backend venv uses Python $stale; the project requires >= 3.11."
        write_note "Rebuilding the venv with the current python3."
        rm -rf "$VENV_DIR"
        needs_venv=1
    fi

    if [[ $needs_venv -eq 1 ]]; then
        # Pick the highest python3.x available; the project requires >= 3.11.
        local pybin=""
        for candidate in python3.13 python3.12 python3.11 python3; do
            if command -v "$candidate" >/dev/null 2>&1; then
                pybin="$candidate"
                break
            fi
        done
        [[ -n "$pybin" ]] || { echo "No python3 on PATH." >&2; exit 1; }
        local host_ver; host_ver="$(python_version "$(command -v "$pybin")")"
        if ! version_ok "$host_ver"; then
            echo "Python $host_ver on PATH is too old; install Python >= 3.11." >&2
            exit 1
        fi
        write_step "Creating backend virtualenv at $VENV_DIR (Python $host_ver)"
        "$pybin" -m venv "$VENV_DIR"
    else
        write_note "backend venv present (Python $(python_version "$PYTHON_EXE"))"
    fi

    if backend_installed; then
        write_note "lucidium + pydantic already installed in venv"
        return
    fi

    write_step "Installing backend (with dev extras)"
    "$PYTHON_EXE" -m pip install --upgrade pip wheel
    "$PYTHON_EXE" -m pip install -e "$BACKEND_DIR[dev]"

    if ! backend_installed; then
        echo "lucidium did not import after install; check pip output above" >&2
        exit 1
    fi
}

initialise_frontend() {
    if [[ -d "$NODE_MODULES" ]]; then
        write_note "frontend node_modules already present"
        return
    fi
    write_step "Installing frontend dependencies"
    npm --prefix "$FRONTEND_DIR" install
}

invoke_codegen() {
    if [[ $SKIP_CODEGEN -eq 1 ]]; then
        write_note "skipping codegen"
        return
    fi
    write_step "Exporting JSON Schemas (Pydantic -> shared-schemas/)"
    "$PYTHON_EXE" "$REPO_ROOT/scripts/codegen/export-schemas.py"

    write_step "Generating TypeScript types (json-schema-to-typescript)"
    npm --prefix "$FRONTEND_DIR" run codegen
}

build_electron_main() {
    write_step "Compiling Electron main + preload to dist-electron/"
    ( cd "$FRONTEND_DIR" && npx tsc -p tsconfig.electron.json )
}

process_command_line() {
    ps -o args= -p "$1" 2>/dev/null | head -n 1
}

process_cwd() {
    local pid="$1"
    if [[ -r "/proc/$pid/cwd" ]]; then
        readlink -f "/proc/$pid/cwd" 2>/dev/null && return
    fi
    if command -v lsof >/dev/null 2>&1; then
        lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1
    fi
}

# True when the process looks like this checkout's Vite: either it runs
# from somewhere inside the repo, or its command line names the repo.
is_our_vite() {
    local pid="$1"
    local cwd cmd
    cwd="$(process_cwd "$pid" || true)"
    cmd="$(process_command_line "$pid" || true)"
    [[ -n "$cwd" && ( "$cwd" == "$REPO_ROOT" || "$cwd" == "$REPO_ROOT"/* ) ]] && return 0
    [[ -n "$cmd" && "$cmd" == *"$REPO_ROOT"* ]] && return 0
    return 1
}

# Kill this repo's own stale Vite so we can re-bind the port. Anything
# else listening there belongs to someone else -- report it and stop.
stop_stale_vite() {
    local port="${1:-$VITE_PORT}"
    local pids=""
    if command -v lsof >/dev/null 2>&1; then
        pids="$(lsof -ti TCP:"$port" -s TCP:LISTEN 2>/dev/null || true)"
    elif command -v fuser >/dev/null 2>&1; then
        pids="$(fuser "$port"/tcp 2>/dev/null || true)"
    fi
    [[ -z "$pids" ]] && return 0

    local killed=0
    for pid in $pids; do
        [[ -z "$pid" || "$pid" == "$$" ]] && continue
        if is_our_vite "$pid"; then
            write_note "killing stale process $pid listening on $port"
            kill -TERM "$pid" 2>/dev/null || true
            killed=1
        else
            local cmd
            cmd="$(process_command_line "$pid" || true)"
            {
                echo ""
                echo "Port $port is in use by a process that does not belong to this repository:"
                echo "    PID     : $pid"
                echo "    Command : ${cmd:-<unavailable>}"
                echo ""
                echo "Refusing to kill it. Either stop that process yourself"
                echo "(kill $pid), or free port $port another way, then re-run"
                echo "$(basename "${BASH_SOURCE[0]}")."
            } >&2
            exit 1
        fi
    done
    [[ $killed -eq 1 ]] && sleep 0.5 || true
    return 0
}

wait_for_vite_ready() {
    local port="${1:-$VITE_PORT}"
    local timeout="${2:-60}"
    write_note "waiting for Vite at http://localhost:$port ..."
    local deadline=$((SECONDS + timeout))
    while [[ $SECONDS -lt $deadline ]]; do
        for host in localhost 127.0.0.1; do
            if curl -fsS -m 1 -o /dev/null "http://$host:$port" 2>/dev/null; then
                return 0
            fi
        done
        sleep 0.25
    done
    echo "Vite did not become ready within $timeout seconds" >&2
    return 1
}

start_backend_only() {
    write_step "Starting Python backend (lucidium.app)"
    write_note "The backend prints LUCIDIUM_WS_PORT=<port> on stdout when ready."
    exec "$PYTHON_EXE" -m lucidium.app
}

start_renderer_only() {
    stop_stale_vite
    write_step "Starting Vite dev server"
    exec npm --prefix "$FRONTEND_DIR" run dev
}

start_app() {
    build_electron_main
    stop_stale_vite
    write_step "Starting Vite dev server in the background"
    local vite_log="${TMPDIR:-/tmp}/lucidium-vite.log"
    # setsid so we can signal the whole Vite process group on cleanup.
    setsid npm --prefix "$FRONTEND_DIR" run dev >"$vite_log" 2>&1 &
    local vite_pid=$!

    cleanup() {
        write_step "Stopping Vite"
        # Negative PID targets the process group created by setsid.
        kill -TERM -"$vite_pid" 2>/dev/null || kill -TERM "$vite_pid" 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM

    if ! wait_for_vite_ready; then
        echo "" >&2
        echo "--- Vite output (last 40 lines of $vite_log) ---" >&2
        tail -n 40 "$vite_log" >&2 || true
        echo "----------------------------------------------" >&2
        exit 1
    fi

    export VITE_DEV_SERVER_URL="http://localhost:$VITE_PORT"
    # Electron's main process would otherwise guess backend/.venv, which
    # is the Windows-side venv path -- not the .venv-linux we set up.
    export LUCIDIUM_PYTHON="$PYTHON_EXE"
    write_step "Launching Electron"
    write_note "Close the window or press Ctrl+C to stop."
    "$ELECTRON_BIN" "$DIST_ELECTRON/main.js"
}

# --- Main ---------------------------------------------------------------

test_required

if [[ $DO_SETUP -eq 1 ]]; then
    initialise_backend
    initialise_frontend
    invoke_codegen
fi

if [[ $ONLY_SETUP -eq 1 ]]; then
    write_step "Setup complete."
    exit 0
fi

if [[ $ONLY_BACKEND -eq 1 ]]; then
    start_backend_only
fi

if [[ $ONLY_RENDERER -eq 1 ]]; then
    start_renderer_only
fi

start_app
