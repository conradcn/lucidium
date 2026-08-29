#!/usr/bin/env bash
# Lucidium dev tasks (Linux / WSL2). Run a target with `./tasks.sh <target>`.
#
# Bash mirror of ``tasks.ps1``. Uses the Linux venv at
# ``backend/.venv-linux/`` (see start.sh / scripts/package-linux.sh for
# why it is kept separate from the Windows ``backend/.venv/``).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$REPO_ROOT/backend/.venv-linux/bin/python"
FRONTEND_DIR="$REPO_ROOT/frontend"

TARGET="${1:-help}"

invoke_codegen() {
    echo "==> Exporting JSON Schemas from Pydantic..."
    "$VENV_PYTHON" "$REPO_ROOT/scripts/codegen/export-schemas.py"
    echo "==> Generating TypeScript types..."
    npm --prefix "$FRONTEND_DIR" run codegen
}

invoke_test_backend() {
    echo "==> Running backend offline tests..."
    ( cd "$REPO_ROOT/backend" && "$VENV_PYTHON" -m pytest )
}

invoke_test_frontend() {
    echo "==> Running frontend offline tests..."
    npm --prefix "$FRONTEND_DIR" test
}

invoke_test_e2e() {
    echo "==> Running Playwright e2e tests (mocked WS)..."
    npm --prefix "$FRONTEND_DIR" run "test:e2e"
}

invoke_smoke() {
    echo "==> Running full-pipeline smoke tests (start.sh --setup, --backend, --renderer)..."
    if [[ ! -x "$VENV_PYTHON" ]]; then
        echo "backend/.venv-linux missing. Run ./start.sh --setup first." >&2
        exit 1
    fi
    ( cd "$REPO_ROOT" && "$VENV_PYTHON" -m pytest tests/smoke -v )
}

case "$TARGET" in
    codegen)        invoke_codegen ;;
    test-backend)   invoke_test_backend ;;
    test-frontend)  invoke_test_frontend ;;
    test-e2e)       invoke_test_e2e ;;
    test)           invoke_test_backend; invoke_test_frontend ;;
    smoke)          invoke_smoke ;;
    all)            invoke_test_backend; invoke_test_frontend; invoke_test_e2e; invoke_smoke ;;
    *)
        cat <<'EOF'
Lucidium dev tasks
  codegen         Pydantic -> JSON Schema -> TypeScript
  test-backend    pytest with offline gate
  test-frontend   vitest with offline gate
  test-e2e        Playwright e2e against mocked WebSocket
  test            Run both unit suites (backend + frontend)
  smoke           Full-pipeline smoke tests via start.sh
  all             test + test-e2e + smoke
EOF
        ;;
esac
