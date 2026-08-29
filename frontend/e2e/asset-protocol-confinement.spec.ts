/**
 * Real-Electron security smoke: the ``lucidium-asset://`` scheme must be
 * confined to the app's asset roots, and the packaged renderer must run
 * clean under its Content-Security-Policy.
 *
 * Prerequisites are the same as ``electron-smoke.spec.ts`` (a built
 * ``dist/`` and ``dist-electron/``) — but no Python backend: this spec
 * runs with ``LUCIDIUM_SKIP_BACKEND=1`` and points ``LUCIDIUM_APP_DATA``
 * at a throwaway directory holding one fake portrait.
 */

import { mkdtempSync, mkdirSync, writeFileSync, rmSync, existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import { _electron as electron, test, expect } from "@playwright/test";

const repoRoot = path.resolve(__dirname, "..", "..");
const electronMain = path.join(repoRoot, "frontend", "dist-electron", "main.js");
const distIndex = path.join(repoRoot, "frontend", "dist", "index.html");

// Smallest valid PNG (1x1, transparent) so the <img> probe really decodes.
const PNG_1X1 = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "base64",
);

function assetUrl(diskPath: string): string {
  const segments = diskPath
    .replace(/\\/g, "/")
    .split("/")
    .filter((s) => s.length > 0)
    .map((s) => encodeURIComponent(s));
  return `lucidium-asset://local/${segments.join("/")}`;
}

test.describe("lucidium-asset:// confinement (real Electron)", () => {
  test.skip(!existsSync(electronMain), "Electron main not built (tsc -p tsconfig.electron.json)");
  test.skip(!existsSync(distIndex), "Renderer dist not built (npm run build)");

  test("serves in-root media, 403s everything else, and raises no CSP violations", async () => {
    const appData = mkdtempSync(path.join(os.tmpdir(), "lucidium-e2e-"));
    const imagesDir = path.join(appData, "saves", "game-1", "images");
    mkdirSync(imagesDir, { recursive: true });
    const portrait = path.join(imagesDir, "portrait.png");
    writeFileSync(portrait, PNG_1X1);
    writeFileSync(path.join(appData, "settings.json"), '{"llm":{"api_key":"secret"}}');

    const electronApp = await electron.launch({
      args: [electronMain, "--skip-warning"],
      cwd: path.join(repoRoot, "frontend"),
      env: {
        ...process.env,
        LUCIDIUM_LOAD_DIST: "1",
        LUCIDIUM_SKIP_BACKEND: "1",
        LUCIDIUM_APP_DATA: appData,
      },
      timeout: 60_000,
    });

    const cspViolations: string[] = [];
    try {
      const window = await electronApp.firstWindow({ timeout: 30_000 });
      window.on("console", (msg) => {
        if (/content security policy/i.test(msg.text())) cspViolations.push(msg.text());
      });
      await window.waitForLoadState("domcontentloaded");
      await expect(window.getByTestId("start-screen")).toBeVisible({ timeout: 15_000 });

      // The CSP really is in force on the loaded document.
      const csp = await window.evaluate(
        () =>
          document
            .querySelector('meta[http-equiv="Content-Security-Policy"]')
            ?.getAttribute("content") ?? "",
      );
      expect(csp).toContain("default-src 'none'");
      expect(csp).toContain("lucidium-asset:");

      const status = (url: string): Promise<number> =>
        window.evaluate(
          (u) => fetch(u).then((r) => r.status).catch(() => -1),
          url,
        );

      // 1. A legitimate portrait under <app-data>/saves loads.
      expect(await status(assetUrl(portrait))).toBe(200);
      // ...and decodes as a real image in the page.
      const decoded = await window.evaluate(
        (u) =>
          new Promise<boolean>((resolve) => {
            const img = new Image();
            img.onload = () => resolve(img.naturalWidth > 0);
            img.onerror = () => resolve(false);
            img.src = u;
          }),
        assetUrl(portrait),
      );
      expect(decoded).toBe(true);

      // 2. An arbitrary absolute path outside the root is refused.
      expect(await status("lucidium-asset://local/C%3A/Windows/win.ini")).toBe(403);
      expect(await status(assetUrl("/etc/passwd"))).toBe(403);

      // 3. %2e%2e survives Chromium's normalisation and must still be
      //    caught — this walks out of saves/ to settings.json.
      const traversal = `${assetUrl(imagesDir)}/%2e%2e/%2e%2e/%2e%2e/settings.json`;
      expect(traversal).toContain("%2e%2e");
      expect(await status(traversal)).toBe(403);

      // 4. Non-media inside the root is refused too.
      writeFileSync(path.join(appData, "saves", "game-1", "game.json"), "{}");
      expect(await status(assetUrl(path.join(appData, "saves", "game-1", "game.json")))).toBe(403);

      // Screens past the start menu need a live backend, so the deeper
      // CSP walk lives in ``electron-smoke.spec.ts`` (which spawns one);
      // here we only assert the boot path is violation-free.
      expect(cspViolations).toEqual([]);
    } finally {
      await electronApp.close();
      rmSync(appData, { recursive: true, force: true });
    }
  });
});
