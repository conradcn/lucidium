# Operating Lucidium

Where data lives, how to back it up, where logs land, and how to run the offline test suite.

## App-data layout

| Path | Contents |
|------|----------|
| Windows: `%APPDATA%\Lucidium\` | Top-level user data. |
| macOS: `~/Library/Application Support/Lucidium/` | Same shape. |
| Linux: `$XDG_DATA_HOME/Lucidium` (or `~/.local/share/Lucidium`) | Same shape. |

Inside that directory:

```
settings.json                   # global per-installation settings
session.log                     # rotating backend log (8 MB, one backup: session.log.1)
saves/
  <save-id>/
    game.json                   # full Game (world + dialog tree + characters)
    meta.json                   # name, last_played_at, settings snapshot
    images/
      <hash>.png                # content-addressed assets, deduplicated
models/
  image/                        # embedded-backend SDXL-family checkpoints
                                # (.safetensors / .ckpt); default for
                                # ImageSettings.embedded_models_dir when it is
                                # empty — see config.py::embedded_models_dir
```

Override the directory for testing or sandboxing by setting `LUCIDIUM_APP_DATA`.

The torch runtime overlay lives in a **separate** tree, deliberately not under app-data:

| Platform | Torch-overlay root |
|---|---|
| Windows | `%LOCALAPPDATA%\Lucidium\runtime\` |
| macOS | `~/Library/Application Support/Lucidium/runtime/` |
| Linux | `$XDG_DATA_HOME/lucidium/runtime` (or `~/.local/share/lucidium/runtime`) |

```
overlays/
  cpu/ cuda/ rocm/ directml/ xpu/   # unpacked torch + torchvision wheels
active_overlay                      # one-line pointer naming the active flavor
```

These are large, regenerable, machine-specific binaries, so a settings reset or a
save-folder sync never touches them, and a torch reinstall never risks a save.
Windows uses `LOCALAPPDATA` (not roaming `APPDATA`) so the bytes never roam.
Override with `LUCIDIUM_RUNTIME_DIR`; `LUCIDIUM_TORCH_OVERLAY` names a fully
resolved overlay directory directly, bypassing the pointer file. See
[torch-overlay.md](torch-overlay.md).

## Backup and restore

Saves are plain folders of JSON + PNGs. Backup is `Copy-Item`/`cp -r` of `%APPDATA%\Lucidium`. Restore is the same in reverse. Editing `game.json` by hand works as long as the schema-version contract is honored; if you want to be safe, do it while the app is closed.

## Logs

| Surface | Destination |
|---------|-------------|
| Backend (`lucidium.app`) | stderr, captured by Electron and prefixed with `[backend]`, **and** `<app-data>/session.log`. |
| Renderer | Browser DevTools console (`Ctrl+Shift+I`) or Electron's main-process console. |
| Provider HTTP errors | Visible as `[backend] ... ProviderUnreachableError: ...` and as `s2c/error` on the WebSocket; the renderer surfaces them as a top-of-screen banner. |

The backend also writes `<app-data>/session.log` (`app.py::_configure_logging`): a `RotatingFileHandler` capped at 8 MB with one backup (`session.log.1`), so the previous run stays available for a bug report without the file growing unbounded. It carries the same records as stderr, formatted `%(asctime)s %(levelname)s %(name)s: %(message)s`. If the directory cannot be created (permissions, read-only volume) the engine falls back to stderr-only and still runs — so treat the file as best-effort, not guaranteed.

## Running the offline test suite

The constitution mandates that the offline suite stays green. To run it:

```powershell
# From repo root, with backend deps installed:
pwsh tasks.ps1 test
```

Or piece by piece:

```powershell
backend\.venv\Scripts\Activate.ps1
pytest backend\tests
npm --prefix frontend test
```

Both suites refuse to make network calls. If a test reaches out, it fails by design (`respx` for Python, fetch denial in `frontend/tests/setup.ts` for the renderer).

## Running the e2e and smoke tests

The Playwright e2e suite drives Chromium through every navigation path with a mocked WebSocket — useful for catching renderer regressions without a live backend.

```powershell
npm --prefix frontend run test:e2e:install   # one-time, downloads Chromium (~140 MB)
pwsh tasks.ps1 test-e2e
```

The full-pipeline smoke tests exercise `start.ps1` end-to-end: `-Setup` produces an importable backend, the schema codegen writes the expected files, `python -m lucidium.app` accepts a real WebSocket handshake, `start.ps1 -Renderer` brings up Vite serving `index.html`, and `tsc -p tsconfig.electron.json` compiles the Electron entrypoints. These tests start real subprocesses and connect to local ports, so they live in `tests/smoke/` outside the offline-network gate.

```powershell
pwsh tasks.ps1 smoke
```

Run everything in sequence:

```powershell
pwsh tasks.ps1 all   # backend + frontend + e2e + smoke
```

## Resetting state

| Goal | Action |
|------|--------|
| Clear all saves | Delete `<app-data>/saves/`. |
| Reset settings to defaults | Delete `<app-data>/settings.json`. |
| Force a clean dev run | `Remove-Item -Recurse $env:APPDATA\Lucidium`. |

## Where to look when something breaks

| Symptom | First place |
|---------|-------------|
| Renderer freezes mid-node | Backend stderr (`[backend]` lines) for the LLM call; check `s2c/error` in DevTools. |
| Save will not load after upgrade | `Game.schema_version` in `game.json` versus `GAME_SCHEMA_VERSION` in `backend/src/lucidium/config.py`. |
| Free-text input did not invalidate speculation | `backend/src/lucidium/orchestration/obsolescence.py` and the per-turn `premise_hash` recomputation. |
| WebSocket fails to connect | Confirm the backend printed `LUCIDIUM_WS_PORT=...` on stdout. The Electron main console captures it as `[backend]` lines. |
| Cost numbers drifting | `backend/src/lucidium/orchestration/cost.py` — current estimates are character-based, not real token counts. Replace with provider-reported usage when the LLM client gains that surface. |
