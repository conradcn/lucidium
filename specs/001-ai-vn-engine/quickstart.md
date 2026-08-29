# Quickstart: AI-Driven Visual Novel Engine

**Feature**: 001-ai-vn-engine
**Audience**: a contributor sitting down to work on this feature for the first time.

## Prerequisites

- Python 3.11+ on PATH.
- Node.js 20 LTS (with `npm`).
- An OpenRouter API key (or any OpenAI-compatible endpoint) — only required for end-to-end runs.
- Disk space for the embedded image backend: image generation runs **in-process** (diffusers + an SDXL-family checkpoint), so the first end-to-end run downloads a torch overlay for your GPU and a base checkpoint (several GB each). No external image server is needed.
- Optional: ComfyUI, only if you deliberately switch `image.backend` to `comfyui` in Settings. The default is `embedded` and the app never contacts ComfyUI in that mode.
- Optional: an ACE-Step server for background music. `music.enabled` is off by default; see [docs/music.md](../../docs/music.md).

## First-time setup

`start.ps1` / `start.sh` does all of it, idempotently — venv creation, `pip install -e backend[dev]`, `npm install`, schema codegen, and the Electron main+preload compile:

```powershell
.\start.ps1 -Setup        # Windows: set everything up, don't launch
```

```bash
./start.sh --setup        # macOS / Linux
```

If you prefer to drive the steps yourself:

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
pip install -e backend[dev]
npm --prefix frontend install

# Pydantic -> JSON Schema -> TypeScript. REQUIRED after any change to
# backend/src/lucidium/api/messages.py or backend/src/lucidium/domain/.
python scripts\codegen\export-schemas.py   # writes shared-schemas/ + the specs mirror
npm --prefix frontend run codegen          # regenerates frontend/src/shared/generated/
```

Both codegen steps run automatically inside `start.ps1` / `start.sh` unless you pass `-SkipCodegen` / `--skip-codegen`.

## Run the offline test suite

This is the suite the constitution requires to stay green.

```powershell
# Backend tests with recorded fixtures (no network)
pytest backend\tests

# Frontend tests
cd frontend
npm test
cd ..
```

If any test reaches out over the network, the test fails. `respx` is configured to intercept all HTTP and treat unmocked traffic as a failure.

## Run the app end-to-end

```powershell
# Configure secrets (one-time): set llm.api_key in Settings, or edit
# %APPDATA%\Lucidium\settings.json directly.

.\start.ps1               # Windows
```

```bash
./start.sh                # macOS / Linux
```

This runs setup if needed, starts the Vite dev server in the background, and launches Electron pointing at it. Electron's main process (`frontend/electron/main.ts`) spawns the Python backend from `backend/.venv`; the backend prints `LUCIDIUM_WS_PORT=<port>` on stdout once listening, and Electron connects to that port.

Useful flags (PowerShell / bash):

| Flag | Effect |
|---|---|
| `-Setup` / `--setup` | Run setup and exit without launching. |
| `-Backend` / `--backend` | Backend only — no Electron, no Vite. For driving the WebSocket by hand. |
| `-Renderer` / `--renderer` | Vite dev server only. The renderer will fail to connect; handy for UI iteration. |
| `-SkipCodegen` / `--skip-codegen` | Skip the schema/TypeScript codegen when you know it is current. |
| `-NoSetup` / `--no-setup` | Skip the dependency-presence checks for a faster restart. |

Do **not** run `npm run dev` directly: that is bare Vite with no Electron and no backend, so nothing connects.

## Smoke checklist (manual, after a change)

These steps mirror the User Story 1 acceptance scenarios — run them on every nontrivial change.

1. **Start screen renders without a save**: `Continue` is hidden. `New Game`, `Load Game`, `Options`, `Exit` are visible.
2. **New Game interview reaches the main view**: Walk through Setting → Genre → Visual Style → Character Description → Name → Confirm using only suggested options. Confirm the white-room placeholder gets replaced with a setting-appropriate image while you are still picking Visual Style.
3. **First node is playable**: After confirmation, the Main UI loads with a background, the player character on stage, dialog text, and either an option set or "Continue" plus a free-text input.
4. **Advance through 10 nodes**: At least one option-driven branch and at least one free-text input. The screen never sits on a blank background for longer than the SC-003 fallback budget.
5. **Save survives a relaunch**: Close the app from the OS, relaunch, press Continue, verify the same node, characters, background, and world values.
6. **Story panel edit propagates**: Open Story → Characters, change a character's Description, advance two nodes, verify the new dialog reflects the edit.
7. **Settings change applies live**: Change typewriter speed, advance one node, see the difference.

## Where to look when something breaks

| Symptom | First place to look |
|---|---|
| Renderer freezes on a node | `s2c/text/streaming` is stalled. Check backend logs for an LLM provider error; check `s2c/error` in the renderer console. |
| A character's portrait identity drifts between appearances | `Character.seed` may have been reset. Confirm via `<save-id>/game.json`; log the prompt/seed pair from `orchestration/render_scheduler.py`. |
| Renders refuse to start with a `torch_installing` error | A torch overlay is downloading. Image generation is deliberately gated while one is in flight rather than silently falling back to minutes-per-image CPU SDXL. Watch `s2c/torch_overlay/progress`; state lives under `%LOCALAPPDATA%\Lucidium\runtime\`. See [docs/torch-overlay.md](../../docs/torch-overlay.md). |
| "no torch" / CPU-slow renders after a fresh clone | Dev runs use the venv's torch, not an overlay. `pip install -e backend[dev]` pulls CPU torch; install a GPU wheel into the venv, or set `LUCIDIUM_TORCH_OVERLAY` to a resolved overlay dir. |
| Image generation errors with "no models" | The embedded models dir is empty. Use the first-run one-click download, or drop a `.safetensors` into `%APPDATA%\Lucidium\models\image\`. See [docs/model-catalog.md](../../docs/model-catalog.md). |
| A model or torch download stalls or dies mid-way | Both are resumable-by-restart, not resumable-in-place: delete the partial file under the models dir / `runtime/overlays/<flavor>/` and retry. Check `%APPDATA%\Lucidium\session.log`. |
| No background music | `music.enabled` defaults to false and needs an ACE-Step server. `c2s/music/inventory` probes it; a failed probe returns `ok: false` with the error. See [docs/music.md](../../docs/music.md). |
| Save fails to load after upgrading | Check `Game.schema_version` vs `persistence/save_store.py` migrations. |
| Free-text input doesn't invalidate speculation | `orchestration/obsolescence.py` premise-hash logic. |
| WebSocket fails to connect | The sidecar didn't print `LUCIDIUM_WS_PORT=`. Check the backend's stdout in `frontend`'s dev console. |

## What is intentionally out of scope for this feature

- Cloud sync, multiplayer, or any networked multi-user state.
- Mobile/touch UI; the renderer is desktop-first.
- A plugin system for swapping orchestration policies.
- Streaming image generation; images are atomic and downloaded as files.
- A visual editor for ComfyUI workflows; users who opt into the `comfyui` backend supply workflow JSON via Settings.
