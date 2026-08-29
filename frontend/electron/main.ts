import { app, BrowserWindow, ipcMain, Menu, net, protocol, shell } from "electron";
import { spawn, type ChildProcessByStdio } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Readable } from "node:stream";
import { forbiddenResponse, resolveAssetRequest } from "./assetPath";

const PORT_PREFIX = "LUCIDIUM_WS_PORT=";
// The backend mints a fresh bearer token per launch and announces it on
// stdout immediately before the port line. Every WebSocket handshake has
// to carry it, so nothing else on the machine can drive the engine.
const TOKEN_PREFIX = "LUCIDIUM_WS_TOKEN=";
const ASSET_SCHEME = "lucidium-asset";

// Privileged registration must happen synchronously at module load,
// BEFORE app.whenReady — so the renderer can resolve assetUrl() the
// moment the page boots. Treating the scheme as standard + secure
// avoids the cross-origin block the page runs into for raw file:// URLs.
protocol.registerSchemesAsPrivileged([
  {
    scheme: ASSET_SCHEME,
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      // Required from Electron 32 onwards. The renderer page is served
      // from a ``file://`` origin, which Chromium treats as opaque, so a
      // ``fetch()`` at a non-CORS-enabled custom scheme is rejected
      // before the handler ever runs ("Cross origin requests are only
      // supported for protocol schemes: chrome, ..., http, https").
      // Electron 31 let ``supportFetchAPI`` alone carry this; newer
      // Chromium requires the scheme to opt in to CORS explicitly and
      // the response to carry ``Access-Control-Allow-Origin`` (added in
      // registerAssetProtocol / forbiddenResponse). ``<img src>`` and
      // ``<audio src>`` are no-CORS loads and worked either way — only
      // fetch() was broken.
      corsEnabled: true,
      // Deliberately NOT bypassCSP: assets served through this scheme
      // stay subject to the renderer's Content-Security-Policy (see
      // CONTENT_SECURITY_POLICY), so a file that somehow slipped past
      // the confinement check still can't execute as script.
      stream: true,
    },
  },
]);

let backend: ChildProcessByStdio<null, Readable, Readable> | null = null;
let backendPort: number | null = null;
let backendToken: string | null = null;
let mainWindow: BrowserWindow | null = null;
// Set once we're intentionally tearing down (app quit or overlay
// relaunch) so the child's exit handler doesn't try to resurrect it.
let quitting = false;
// Set true only after the real app has loaded successfully — auto-restart
// is scoped to "the backend died mid-session", not to a failed cold boot
// (which the splash-error path already handles).
let appLoaded = false;
// Crash-restart backoff bookkeeping. Reset to 0 every time a spawn
// successfully announces its port.
let restartAttempts = 0;
const MAX_RESTART_ATTEMPTS = 5;

const repoRoot = path.resolve(__dirname, "..", "..");
// Force file:// loading even when not packaged. Useful for Playwright
// tests that prebuild the renderer with `vite build` and don't want a
// Vite dev server in the loop.
const forceFile = process.env.LUCIDIUM_LOAD_DIST === "1";
const isDev = !app.isPackaged && !forceFile;

// ``--skip-warning`` suppresses the startup safety modal
// (screens/StartupWarning.tsx) for this launch.
//
// This exists for AUTOMATION, not for players. The modal is deliberately
// unskippable in normal use — it re-shows every launch, with no "don't
// show again", because a warning you can permanently silence stops
// registering. But it is also a focus-trapping ``aria-modal`` overlay
// that intercepts pointer events, so every end-to-end test has to claw
// past it before it can touch the app. Clicking it away in each spec is
// both noise and a flake source (the click races the modal's mount).
//
// Passing this flag is an explicit, per-launch act by whoever spawns the
// process; nothing in the shipped UI sets it and it does not persist.
// It is forwarded to the renderer as ``?skipWarning=1`` on the URL,
// which is the same channel main already uses for wsPort / wsToken —
// so the browser-only Playwright specs (no Electron in the loop) can
// opt in with the same query parameter.
const skipStartupWarning = process.argv.includes("--skip-warning");

function pythonExecutable(): string {
  // start.sh / start.ps1 export this so the launcher and Electron can
  // never disagree about which interpreter setup actually created.
  const override = process.env.LUCIDIUM_PYTHON;
  if (override) {
    return override;
  }
  if (process.platform === "win32") {
    return path.join(repoRoot, "backend", ".venv", "Scripts", "python.exe");
  }
  // Linux/macOS setup may land in either venv: ``.venv`` for a plain
  // manual install, ``.venv-linux`` for start.sh (kept separate so a
  // Windows-side install on the same checkout isn't clobbered).
  const candidates = [
    path.join(repoRoot, "backend", ".venv", "bin", "python"),
    path.join(repoRoot, "backend", ".venv-linux", "bin", "python"),
  ];
  const found = candidates.find((candidate) => fs.existsSync(candidate));
  if (!found) {
    throw new Error(
      `No backend Python interpreter found. Looked for:\n` +
        candidates.map((candidate) => `  ${candidate}`).join("\n") +
        `\nRun ./start.sh --setup to create one, or set LUCIDIUM_PYTHON ` +
        `to the interpreter you want used.`,
    );
  }
  return found;
}

interface BackendCommand {
  exe: string;
  args: string[];
  cwd: string;
}

/**
 * Resolve the backend invocation for the current run mode.
 *
 *   * Dev / unpackaged: use the venv's Python and run
 *     ``python -m lucidium.app`` from ``backend/``.
 *   * Packaged: electron-builder copies the PyInstaller output
 *     into ``<resourcesPath>/lucidium-backend/`` (see the
 *     ``extraResources`` block in package.json). The launcher
 *     EXE there self-contains the interpreter + lucidium + its
 *     deps, so we just spawn it directly with no module flag.
 */
function resolveBackendCommand(): BackendCommand {
  if (app.isPackaged) {
    const root = path.join(process.resourcesPath, "lucidium-backend");
    const exeName = process.platform === "win32"
      ? "lucidium-backend.exe"
      : "lucidium-backend";
    return {
      exe: path.join(root, exeName),
      args: [],
      cwd: root,
    };
  }
  return {
    exe: pythonExecutable(),
    args: ["-m", "lucidium.app"],
    cwd: path.join(repoRoot, "backend"),
  };
}

function spawnBackend(): Promise<number> {
  return new Promise((resolve, reject) => {
    const { exe, args, cwd } = resolveBackendCommand();
    const child = spawn(exe, args, {
      cwd,
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, PYTHONUNBUFFERED: "1" },
    });
    backend = child;

    child.stdout.setEncoding("utf-8");
    child.stderr.setEncoding("utf-8");

    // Guard so the promise settles exactly once: the port line resolves
    // it, an early exit / spawn error rejects it.
    let settled = false;

    let stdoutBuffer = "";
    child.stdout.on("data", (chunk: string) => {
      stdoutBuffer += chunk;
      let newlineIndex = stdoutBuffer.indexOf("\n");
      while (newlineIndex !== -1) {
        const line = stdoutBuffer.slice(0, newlineIndex).trim();
        stdoutBuffer = stdoutBuffer.slice(newlineIndex + 1);
        if (line.startsWith(TOKEN_PREFIX)) {
          // Never logged — it's the credential for the local socket.
          backendToken = line.slice(TOKEN_PREFIX.length);
        } else if (line.startsWith(PORT_PREFIX)) {
          const port = Number.parseInt(line.slice(PORT_PREFIX.length), 10);
          if (Number.isFinite(port)) {
            backendPort = port;
            if (!settled) {
              settled = true;
              resolve(port);
            }
          }
        } else if (line) {
          console.log("[backend] %s", line);
        }
        newlineIndex = stdoutBuffer.indexOf("\n");
      }
    });
    child.stderr.on("data", (chunk: string) => process.stderr.write(`[backend] ${chunk}`));
    child.on("error", (err) => {
      if (!settled) {
        settled = true;
        reject(err);
      }
    });
    child.on("exit", (code, signal) => {
      console.log("[backend] exited code=%s signal=%s", code, signal);
      // Only react if THIS child is still the live one — a restart may
      // have already replaced ``backend`` with a newer process.
      const wasCurrent = backend === child;
      if (wasCurrent) {
        backend = null;
        backendPort = null;
        backendToken = null;
      }
      if (!settled) {
        settled = true;
        reject(new Error(`backend exited (code=${code}, signal=${signal}) before announcing a port`));
      }
      // The backend abruptly died while the player was in-session —
      // resurrect it. Skipped during an intentional shutdown / overlay
      // relaunch, and during cold boot (the splash-error path owns that).
      if (wasCurrent && !quitting && appLoaded) {
        scheduleBackendRestart();
      }
    });
  });
}

// Bring the backend back after an unexpected death, with exponential
// backoff. On success, push the (possibly new) port to the renderer so
// its WebSocket client reconnects to the right place — the client reads
// the port only once at boot, so a port change would otherwise leave it
// hammering the dead socket.
function scheduleBackendRestart(): void {
  if (quitting) return;
  if (restartAttempts >= MAX_RESTART_ATTEMPTS) {
    console.error(
      "[backend] gave up restarting after %d attempts", restartAttempts,
    );
    return;
  }
  restartAttempts += 1;
  const delay = Math.min(1000 * 2 ** restartAttempts, 15_000);
  console.log(
    "[backend] scheduling restart #%d in %dms", restartAttempts, delay,
  );
  setTimeout(() => {
    if (quitting || backend) return;
    spawnBackend()
      .then((port) => {
        restartAttempts = 0;
        console.log("[backend] restarted, now on port %d", port);
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send("lucidium:backend-restarted", {
            port,
            token: backendToken,
          });
        }
      })
      .catch((err) => {
        console.error(
          "[backend] restart attempt failed: %s",
          err instanceof Error ? err.message : String(err),
        );
        scheduleBackendRestart();
      });
  }, delay);
}

/**
 * Mirror of ``backend/src/lucidium/config.py::app_data_dir`` — the
 * confinement roots have to agree with the directories the backend
 * actually writes assets into.
 */
function appDataDir(): string {
  const override = process.env.LUCIDIUM_APP_DATA;
  if (override) return path.resolve(override);
  if (process.platform === "win32") {
    const base = process.env.APPDATA || path.join(os.homedir(), "AppData", "Roaming");
    return path.join(base, "Lucidium");
  }
  if (process.platform === "darwin") {
    return path.join(os.homedir(), "Library", "Application Support", "Lucidium");
  }
  const base = process.env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share");
  return path.join(base, "Lucidium");
}

/**
 * Every directory the asset scheme is allowed to read from:
 *
 *   * ``<app-data>/saves`` — per-save ``images/`` (portraits,
 *     backgrounds, generated music) plus the ``_interview_preview``
 *     scratch dir. This is the game-assets root.
 *   * the bundled placeholder art (white room / dream guide) that the
 *     interview screen shows before anything has been rendered. In dev
 *     it's ``backend/workflows/placeholders``; packaged, the whole
 *     PyInstaller bundle sits under ``resources/lucidium-backend``.
 */
function assetRoots(): string[] {
  const roots = [path.join(appDataDir(), "saves")];
  if (app.isPackaged) {
    // PyInstaller's onedir layout puts bundled data under ``_internal``;
    // older/one-file layouts put it next to the EXE. Allow both — the
    // directory that doesn't exist simply never matches.
    const backendRoot = path.join(process.resourcesPath, "lucidium-backend");
    roots.push(path.join(backendRoot, "workflows", "placeholders"));
    roots.push(path.join(backendRoot, "_internal", "workflows", "placeholders"));
  } else {
    roots.push(path.join(repoRoot, "backend", "workflows", "placeholders"));
  }
  return roots;
}

function registerAssetProtocol(): void {
  // URL shape: ``lucidium-asset://local/C%3A/Users/.../foo.png``
  // The ``local`` host satisfies the URL parser's authority requirement
  // for a standard scheme (see assetUrl.ts); the path is the original
  // disk path with each segment percent-encoded.
  //
  // A renderer page can forge any URL it likes here, so the decoded path
  // is confined to assetRoots() and restricted to media extensions
  // before it ever reaches the filesystem — see ./assetPath.ts.
  const roots = assetRoots();
  protocol.handle(ASSET_SCHEME, async (request) => {
    const diskPath = resolveAssetRequest(request.url, roots);
    if (diskPath === null) {
      console.warn("[asset-protocol] refused %s", request.url);
      return forbiddenResponse();
    }
    const fileUrl = pathToFileURL(diskPath).toString();
    try {
      const response = await net.fetch(fileUrl);
      if (!response.ok) {
        console.error(
          "[asset-protocol] %s -> %s -> %s status=%d",
          request.url, diskPath, fileUrl, response.status,
        );
      }
      // Re-wrap so the CORS header rides along (see corsEnabled above).
      // ``response.body`` is a stream, so this stays streaming rather
      // than buffering the whole image/track into main-process memory.
      const headers = new Headers(response.headers);
      headers.set("access-control-allow-origin", "*");
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    } catch (exc) {
      console.error(
        "[asset-protocol] FAILED %s -> %s: %s",
        request.url, fileUrl, exc,
      );
      throw exc;
    }
  });
}

// Self-contained splash shown the instant the window opens, so the
// user sees the app launch immediately instead of staring at nothing
// for the several seconds the Python backend spends importing the
// (heavy) torch/diffusers/transformers stack before it announces its
// WS port. Inlined as a data: URL so it needs no bundled file and
// works identically in dev and packaged builds. Palette mirrors
// styles.css (--bg #0c0d12 / --bg-deep #06070a / --accent #d6b16d).
const SPLASH_HTML = `<!doctype html><html><head><meta charset="utf-8" /><style>
*{margin:0;padding:0;box-sizing:border-box}html,body{height:100%}
body{background:radial-gradient(circle at 50% 38%,#14151c 0%,#0c0d12 45%,#06070a 100%);
color:#d6b16d;font-family:"Iowan Old Style","Source Serif Pro",Georgia,serif;
display:flex;flex-direction:column;align-items:center;justify-content:center;
height:100vh;overflow:hidden;user-select:none;-webkit-user-select:none}
.title{font-size:2.6rem;letter-spacing:.32em;padding-left:.32em;font-weight:600;
text-shadow:0 0 28px rgba(214,177,109,.25)}
.spinner{margin-top:2.2rem;width:34px;height:34px;border-radius:50%;
border:3px solid rgba(214,177,109,.18);border-top-color:#d6b16d;animation:spin .9s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.status{margin-top:1.5rem;font-size:.95rem;color:#8a8676;letter-spacing:.04em;
font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.hint{margin-top:.5rem;font-size:.78rem;color:#4f4d45;
font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.status.error{color:#c8654f}
</style></head><body>
<div class="title">LUCIDIUM</div>
<div class="spinner" id="spinner"></div>
<div class="status" id="status">Starting the engine&hellip;</div>
<div class="hint">First launch can take a moment while the model backend warms up.</div>
</body></html>`;

function splashDataUrl(): string {
  return "data:text/html;charset=utf-8," + encodeURIComponent(SPLASH_HTML);
}

// Surface a backend-start failure ON the splash instead of leaving the
// user on a frozen spinner forever.
async function showSplashError(message: string): Promise<void> {
  if (!mainWindow) return;
  const msg = `The engine failed to start.\n${message}`;
  try {
    await mainWindow.webContents.executeJavaScript(
      `(() => { const s = document.getElementById('status');` +
        ` const sp = document.getElementById('spinner');` +
        ` if (sp) sp.style.display = 'none';` +
        ` if (s) { s.textContent = ${JSON.stringify(msg)}; s.classList.add('error'); } })();`,
    );
  } catch {
    /* window may have been closed mid-failure; nothing to surface to. */
  }
}

// Load the real renderer (dev server or packaged index.html), passing
// the resolved backend port via a query param so the WS client can read
// it synchronously from window.location.search at boot.
async function loadApp(): Promise<void> {
  if (!mainWindow) return;
  const params = new URLSearchParams();
  if (backendPort !== null) params.set("wsPort", String(backendPort));
  if (backendToken !== null) params.set("wsToken", backendToken);
  if (skipStartupWarning) params.set("skipWarning", "1");
  const portSearch = params.toString();
  if (isDev) {
    const devUrl = process.env.VITE_DEV_SERVER_URL ?? "http://127.0.0.1:5173";
    const url = portSearch ? `${devUrl}/?${portSearch}` : devUrl;
    await mainWindow.loadURL(url);
  } else {
    await mainWindow.loadFile(path.join(__dirname, "..", "dist", "index.html"), {
      search: portSearch,
    });
  }
}

async function createWindow(): Promise<void> {
  registerAssetProtocol();
  // Hide Electron's default File/Edit/View/Window/Help menu bar so
  // the game window has clean chrome — the visual novel runs full
  // bleed and there's nothing in the menu the player would use.
  Menu.setApplicationMenu(null);
  // Window/taskbar icon. On Windows-packaged builds the icon is
  // already embedded in the .exe by electron-builder, but dev mode
  // and the Linux AppImage window need it set explicitly. In dev the
  // PNG lives at the repo root; packaged it's copied into resources
  // via the extraResources block in package.json.
  const iconPath = app.isPackaged
    ? path.join(process.resourcesPath, "icon.png")
    : path.join(repoRoot, "icon.png");
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    show: false,
    icon: iconPath,
    // Dark canvas so there's no white flash before the splash paints.
    backgroundColor: "#0c0d12",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.once("ready-to-show", () => mainWindow?.show());

  // Route every ``target="_blank"`` (or window.open) request to
  // the user's default browser instead of letting Electron spawn
  // a new BrowserWindow inside the app. The first-time-setup
  // OpenRouter link in particular needs the real browser so the
  // player can sign in, click around, and copy the API key
  // without the app forcing them to do all that in a stripped-
  // down embedded window. ``deny`` tells Electron NOT to create
  // an in-app window; the ``setImmediate`` shell call is the
  // canonical pattern (it also prevents the deferred error if
  // ``shell.openExternal`` rejects a URL the app shouldn't be
  // launching).
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/i.test(url)) {
      void shell.openExternal(url);
    }
    return { action: "deny" };
  });
  // Same idea, but for plain ``<a href>`` clicks that don't carry
  // ``target="_blank"`` — without this, clicking such a link would
  // navigate the renderer AWAY from index.html and the app would lose
  // its UI entirely. preventDefault comes FIRST and unconditionally:
  // an exotic scheme (file:, data:, javascript:) is exactly the case we
  // least want to fall through and allow, so anything that isn't an
  // http(s) link the OS browser should own is simply blocked.
  const blockNavigation = (event: { preventDefault: () => void }, url: string): void => {
    event.preventDefault();
    if (/^https?:/i.test(url)) void shell.openExternal(url);
  };
  mainWindow.webContents.on("will-navigate", blockNavigation);
  // Sub-frames and server-side redirects reach neither of the above,
  // so they get the identical policy.
  mainWindow.webContents.on("will-frame-navigate", (event) => {
    blockNavigation(event, event.url);
  });
  mainWindow.webContents.on("will-redirect", blockNavigation);

  // Renderer-isolation tests (LUCIDIUM_SKIP_BACKEND=1) push state
  // through a WS mock and don't spawn a Python backend — load the app
  // straight away (no splash) so e2e timing/behaviour is unchanged.
  if (process.env.LUCIDIUM_SKIP_BACKEND === "1") {
    await loadApp();
    appLoaded = true;
    return;
  }

  // 1. Paint the splash immediately; 2. boot the backend with the
  //    splash visible; 3. swap in the real app once the WS port is
  //    announced. This turns the multi-second backend import (heavy ML
  //    stack) from a blank-window dead zone into a responsive loading
  //    screen.
  await mainWindow.loadURL(splashDataUrl());
  try {
    await spawnBackend();
  } catch (err) {
    await showSplashError(err instanceof Error ? err.message : String(err));
    return;
  }
  await loadApp();
  // The real app is up; from here an abrupt backend death should be
  // auto-recovered rather than left dead.
  appLoaded = true;
}

ipcMain.handle("lucidium:get-ws-port", () => backendPort);
ipcMain.handle("lucidium:get-ws-token", () => backendToken);

// Renderer-initiated relaunch. Fired when the torch overlay auto-
// provisioner activates a new wheel — the running Python sidecar has
// the OLD torch on sys.path and can't hot-swap, so the next launch is
// the only way to actually use the freshly-installed flavor. We tear
// the backend down explicitly (rather than relying on app.exit() to
// kill children) so the SDXL pipeline releases VRAM before the new
// process tries to load it on the same GPU.
ipcMain.on("lucidium:relaunch-for-overlay", () => {
  void (async () => {
    try {
      await shutdownBackend();
    } catch {
      /* shutdown is best-effort — proceed with the relaunch regardless */
    }
    app.relaunch();
    app.exit(0);
  })();
});

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  shutdownBackend().finally(() => {
    if (process.platform !== "darwin") app.quit();
  });
});

app.on("before-quit", (event) => {
  if (!backend) return;
  event.preventDefault();
  shutdownBackend().finally(() => app.exit(0));
});

async function shutdownBackend(): Promise<void> {
  // Every caller of shutdownBackend is tearing the process down for good
  // (app quit) or relaunching it (overlay swap). Mark ``quitting`` so the
  // child's exit handler doesn't fire the auto-restart.
  quitting = true;
  const child = backend;
  if (!child) return;
  child.kill("SIGTERM");
  await new Promise<void>((resolve) => child.once("exit", () => resolve()));
}
