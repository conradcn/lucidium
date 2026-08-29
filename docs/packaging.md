# Packaging Lucidium

Lucidium ships as a single Electron desktop app bundled with a Python sidecar built by PyInstaller. This document covers the packaging pipeline; for day-to-day operations once installed see `docs/operations.md`.

## Outputs

| Platform | Artefact | Notes |
|----------|----------|-------|
| Windows  | `frontend/release/Lucidium-Setup.exe` | 7z **self-extracting archive**, not NSIS. The bundled backend (~1–2 GB plus the CPU torch overlay) is larger than 32-bit `makensis` can memory-map; the 7z target still yields one double-clickable `.exe`. |
| Linux    | `frontend/release/Lucidium-*.AppImage` | Cross-built from a Windows host via WSL2, or natively by running `scripts/package-linux.sh`. |

macOS is not currently produced by either script.

## Running the pipeline

Everything is driven by `package.ps1` (Windows host) or `scripts/package-linux.sh` (Linux / WSL). There is no hand-run `pyinstaller` command.

```powershell
pwsh package.ps1                  # both targets: Linux via WSL first, then Windows
pwsh package.ps1 -WindowsOnly     # skip the WSL leg (no WSL2 installed)
pwsh package.ps1 -Linux           # only the AppImage, cross-built via WSL2
pwsh package.ps1 -SkipBackend     # renderer-only iteration
pwsh package.ps1 -Force           # rebuild every step from scratch
pwsh package.ps1 -Clean           # wipe build artefacts and exit
```

Each step short-circuits when its output is already fresh unless `-Force` is passed.

## What the script actually does

1. **Verify the backend venv** and install the *pinned* PyInstaller from `backend[packaging]`. The pin matters: `backend/lucidium.spec` depends on PyInstaller internals, so an arbitrary release would produce a materially different bundle from the same commit.
2. **Warm the safety models** — `backend[safety]` plus a run that instantiates NudeNet's detector and insightface's `buffalo_l` (detection + genderage) so the spec can copy them into the bundle. A build without them ships an inert content filter, which `SAFETY.md` §3 forbids.
3. **Stage the bundled CPU torch overlay** (`Build-CpuOverlay`). torch and torchvision are *excluded* from the freeze — the correct build depends on the player's GPU, and bundling every flavour would blow past 10 GiB. The script therefore points `LUCIDIUM_RUNTIME_DIR` at `backend/build/bundled-overlay-cpu-win` (`-linux` for the Linux leg — a shared path once baked Windows `.dll`s into an AppImage) and calls `torch_overlay.install_flavor("cpu", activate=False)`, which downloads and unpacks the CPU wheels matched to *this* interpreter. Baking that into the installer is what makes image generation work offline on first launch with no download. See [torch-overlay.md](torch-overlay.md).
4. **PyInstaller** — `python -m PyInstaller --noconfirm lucidium.spec`, run with cwd = `backend/`, emitting a **onedir** tree at `backend/dist/lucidium-backend/lucidium-backend.exe`. The spec reads `LUCIDIUM_BUNDLED_OVERLAY_DIR` (set from step 3) to bake the CPU overlay in. UPX is disabled in the spec — it corrupts several bundled `.pyd`/`.dll` files. The first run takes 5–10 minutes as it walks every transitive import of the ML stack; later runs reuse `backend/build/`.
5. **Frontend** — `npm install` then `npm run build` (`tsc -b && vite build`), producing `frontend/dist/` (renderer) and `frontend/dist-electron/` (Electron main + preload).
6. **electron-builder** — run with cwd = `frontend` (electron-builder takes its project root from the `package.json` in cwd), target `7z` for Windows. `signAndEditExecutable: false` in `frontend/package.json`'s `build.win` block skips both `signtool` and the `rcedit` pass; both pull the `winCodeSign` cache archive, which contains macOS dylib symlinks that Windows can only extract with admin rights or Developer Mode. The `extraResources` block copies the PyInstaller output into the packaged app's resources.

The Electron main process (`frontend/electron/main.ts`) spawns the bundled backend on launch. In a packaged build, `resolveBackendCommand` picks `process.resourcesPath`/`lucidium-backend`; in dev, it uses `$LUCIDIUM_PYTHON` when set (both `start.ps1` and `start.sh` export it), else `backend/.venv/Scripts/python.exe` on Windows, else the first of `backend/.venv/bin/python` and `backend/.venv-linux/bin/python` that exists. If none exists it fails with an error naming both probed paths.

## Linux cross-build (WSL2)

PyInstaller bundles the *host* interpreter, so a Linux backend has to be produced from Linux. `package.ps1` shells out to `wsl.exe`: it translates the repo root with `wslpath -a`, then runs `bash scripts/package-linux.sh` inside the distro, forwarding `--skip-backend` / `--skip-frontend` / `--force` / `--clean`. `-WslDistro` overrides the distro (defaults to `wsl.exe`'s default). Requires WSL2 with Python ≥ 3.11.

The Linux leg runs **before** the Windows leg by design: the Windows leg's `npm install` afterwards repopulates `frontend/node_modules` with Windows-native binaries, leaving the checkout Windows-dev-ready. If `wsl.exe` is missing, the default flow warns and skips; `-Linux` fails loudly instead, since there is no fallback.

## Code signing

Code-signing certificates are out of scope for v1. Builds are unsigned; users will see the platform's "unverified developer" dialog on first run.

## Smoke checklist

After running the pipeline:

1. Install on a clean machine (or a clean user profile).
2. Launch the app — confirm the Start Screen renders within 3 seconds.
3. Click **New Game**, walk through the interview, reach the main view (~3 minutes per SC-001).
4. Force-quit, relaunch, click **Continue**. Confirm the same node loads.
5. Open Settings, change typewriter speed, advance — confirm the change.
6. Quit normally. Confirm no orphan Python processes (`tasklist | findstr lucidium-backend` on Windows).
