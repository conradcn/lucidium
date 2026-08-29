/**
 * Real-Electron smoke test.
 *
 * Unlike the other e2e specs, this one launches the actual Electron
 * binary (the same one ``start.ps1`` runs) against a prebuilt renderer
 * and a real Python backend. It catches regressions like "the renderer
 * is connecting to the wrong WebSocket port" that a mocked WebSocket
 * cannot reproduce.
 *
 * Prerequisites:
 *   * ``backend/.venv`` has lucidium + dev deps installed
 *     (``.\start.ps1 -Setup``).
 *   * ``frontend/dist/`` contains a Vite production build
 *     (``npm --prefix frontend run build``).
 *   * ``frontend/dist-electron/main.js`` exists
 *     (``npx tsc -p frontend/tsconfig.electron.json``).
 *
 * The test sets ``LUCIDIUM_LOAD_DIST=1`` so the Electron main loads
 * ``dist/index.html`` directly rather than trying to talk to a Vite dev
 * server.
 */

import path from "node:path";
import { existsSync } from "node:fs";

import { _electron as electron, test, expect } from "@playwright/test";

const repoRoot = path.resolve(__dirname, "..", "..");
const electronMain = path.join(repoRoot, "frontend", "dist-electron", "main.js");
const distIndex = path.join(repoRoot, "frontend", "dist", "index.html");
const venvPython =
  process.platform === "win32"
    ? path.join(repoRoot, "backend", ".venv", "Scripts", "python.exe")
    : path.join(repoRoot, "backend", ".venv", "bin", "python");

test.describe("Electron app (real backend, real WebSocket)", () => {
  test.skip(
    !existsSync(electronMain),
    `Electron main not built. Run: npx tsc -p ${path.join("frontend", "tsconfig.electron.json")}`,
  );
  test.skip(
    !existsSync(distIndex),
    "Renderer dist not built. Run: npm --prefix frontend run build",
  );
  test.skip(
    !existsSync(venvPython),
    "backend venv missing. Run: .\\start.ps1 -Setup",
  );

  // The webServer started by playwright.config.ts is irrelevant for
  // these tests; Electron will load the file:// URL directly. We don't
  // try to disable it because the spec set runs in parallel with the
  // others.

  test("clicking each home-screen button lands on a non-empty phase", async () => {
    const electronApp = await electron.launch({
      args: [electronMain, "--skip-warning"],
      cwd: path.join(repoRoot, "frontend"),
      env: {
        ...process.env,
        LUCIDIUM_LOAD_DIST: "1",
      },
      timeout: 60_000,
    });

    try {
      // Surface console + page errors from the renderer to make
      // debugging tractable when this fails on a developer's machine.
      electronApp.on("console", (msg) => console.log(`[electron-main] ${msg.type()} ${msg.text()}`));
      const window = await electronApp.firstWindow({ timeout: 30_000 });
      // The packaged renderer runs under a strict CSP (injected at build
      // time — see vite.config.mts). Any resource it actually needs but
      // the policy forbids shows up here as a console error.
      const cspViolations: string[] = [];
      window.on("console", (msg) => console.log(`[renderer] ${msg.type()} ${msg.text()}`));
      window.on("console", (msg) => {
        if (/content security policy/i.test(msg.text())) cspViolations.push(msg.text());
      });
      window.on("pageerror", (err) => console.log(`[renderer pageerror] ${err.message}`));
      await window.waitForLoadState("domcontentloaded");

      // Snapshot what we have so far so a failing assertion is informative.
      const url = window.url();
      const html = await window.content();
      console.log(`[electron-smoke] window URL: ${url}`);
      console.log(`[electron-smoke] head: ${html.slice(0, 200)}`);

      // The renderer must reach the start screen.
      await expect(window.getByTestId("start-screen")).toBeVisible({ timeout: 15_000 });

      // The connection banner must clear within 15 s — that proves the
      // renderer actually reached the spawned backend at the right
      // port. If the wsPort wiring is broken, the banner never clears
      // and this assertion catches it.
      await expect(window.getByTestId("connection-banner")).toHaveCount(0, { timeout: 15_000 });

      // The content-warning modal (StartupWarning) sits over the start
      // screen. It is ``aria-modal`` and intercepts pointer events, so
      // any button click below silently retries until the test times
      // out unless it is dismissed first. The acknowledgement is NOT
      // persisted, so it comes back after every ``reload()`` too.
      const dismissStartupWarning = async (): Promise<void> => {
        const warning = window.getByTestId("startup-warning");
        if (await warning.isVisible()) {
          await window.getByRole("button", { name: "I understand" }).click();
          await expect(warning).toHaveCount(0);
        }
      };
      await dismissStartupWarning();

      // Every home-screen button reaches a phase with a known testid.
      await window.getByRole("button", { name: "New Game" }).click();
      await expect(window.getByTestId("phase-interview")).toBeVisible({ timeout: 5_000 });
      // The interview either has options (unlikely without an API key)
      // or shows the loading screen — never blank.
      await expect(
        window.locator('[data-testid="interview-setting"], [data-testid="interview-loading"]'),
      ).toBeVisible();

      // Reload to reset to start screen for the next click.
      await window.reload();
      await expect(window.getByTestId("start-screen")).toBeVisible();
      await dismissStartupWarning();

      await window.getByRole("button", { name: "Load Game" }).click();
      await expect(window.getByTestId("phase-load")).toBeVisible();
      await expect(window.getByRole("heading", { name: "Load game" })).toBeVisible();
      await window.getByRole("button", { name: "Back" }).click();
      await expect(window.getByTestId("phase-start")).toBeVisible();

      await window.getByRole("button", { name: "Options" }).click();
      await expect(window.getByTestId("phase-settings")).toBeVisible();
      await expect(window.getByRole("heading", { name: "Settings" })).toBeVisible();

      expect(cspViolations).toEqual([]);
    } finally {
      await electronApp.close();
    }
  });
});
