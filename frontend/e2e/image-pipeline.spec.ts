/**
 * End-to-end image-pipeline test.
 *
 * Boots the real Electron binary (no Vite dev server, no Python
 * backend), then asks the renderer to load an image from disk via the
 * lucidium-asset:// custom protocol registered in main.ts. The image
 * is one of the bundled placeholders that ship with the app, so this
 * is the same shape of path the backend would hand down at runtime.
 *
 * Two assertions, in increasing strictness:
 *   1. ``fetch(lucidium-asset:///...)`` resolves to 200 with image bytes.
 *      Proves the protocol handler in main.ts is wired up correctly
 *      and can read the file off disk through net.fetch + pathToFileURL.
 *   2. An <img> element with the same URL completes (``naturalWidth > 0``).
 *      Proves Chromium accepts the bytes as a renderable image — i.e.
 *      the response carried the right MIME type / status / headers.
 *
 * The test deliberately does NOT spawn the Python backend; we want
 * this to fail loudly if the protocol regression returns, separately
 * from any LLM/ComfyUI flake.
 */

import path from "node:path";
import { existsSync, readFileSync } from "node:fs";

import { _electron as electron, test, expect } from "@playwright/test";

const mockWsSource = readFileSync(path.join(__dirname, "mock-ws.js"), "utf-8");

const repoRoot = path.resolve(__dirname, "..", "..");
const electronMain = path.join(repoRoot, "frontend", "dist-electron", "main.js");
const distIndex = path.join(repoRoot, "frontend", "dist", "index.html");
const dreamGuidePng = path.join(
  repoRoot,
  "backend",
  "workflows",
  "placeholders",
  "dream_guide.png",
);
const whiteRoomPng = path.join(
  repoRoot,
  "backend",
  "workflows",
  "placeholders",
  "white_room.png",
);

test.describe("Image pipeline (real Electron, lucidium-asset protocol)", () => {
  test.skip(
    !existsSync(electronMain),
    `Electron main not built. Run: npx tsc -p ${path.join("frontend", "tsconfig.electron.json")}`,
  );
  test.skip(
    !existsSync(distIndex),
    "Renderer dist not built. Run: npm --prefix frontend run build",
  );
  test.skip(
    !existsSync(dreamGuidePng) || !existsSync(whiteRoomPng),
    "Bundled placeholder PNGs missing. Run scripts/bundle-placeholders.* to regenerate.",
  );

  test("custom protocol serves on-disk PNG bytes that decode as an image", async () => {
    const electronApp = await electron.launch({
      args: [electronMain, "--skip-warning"],
      cwd: path.join(repoRoot, "frontend"),
      env: {
        ...process.env,
        LUCIDIUM_LOAD_DIST: "1",
        // The backend isn't needed for this test — but the renderer
        // still tries to connect. The connection-banner test in
        // electron-smoke.spec.ts covers that path; here we just want
        // the protocol to function regardless.
        LUCIDIUM_SKIP_BACKEND: "1",
      },
      timeout: 60_000,
    });

    try {
      const window = await electronApp.firstWindow({ timeout: 30_000 });
      window.on("console", (msg) =>
        console.log(`[renderer] ${msg.type()} ${msg.text()}`),
      );
      await window.waitForLoadState("domcontentloaded");

      const assetUrl = await window.evaluate((diskPath: string) => {
        // Reproduce the same encoding the renderer's assetUrl helper
        // produces. Importing it here would require the page already
        // booted into the bundled JS; this small re-implementation
        // keeps the test self-contained.
        const forward = diskPath.replace(/\\/g, "/");
        const segments = forward
          .split("/")
          .filter((seg) => seg.length > 0)
          .map((seg) => encodeURIComponent(seg));
        return `lucidium-asset://local/${segments.join("/")}`;
      }, dreamGuidePng);

      // (1) The protocol handler answers fetch().
      const fetchResult = await window.evaluate(async (url) => {
        const res = await fetch(url);
        const buf = await res.arrayBuffer();
        return {
          ok: res.ok,
          status: res.status,
          byteLength: buf.byteLength,
          // PNG magic bytes — proves we got the actual file, not an
          // HTML error page that happens to be 200.
          first8: Array.from(new Uint8Array(buf).slice(0, 8)),
        };
      }, assetUrl);

      expect(fetchResult.ok).toBe(true);
      expect(fetchResult.status).toBe(200);
      expect(fetchResult.byteLength).toBeGreaterThan(1024);
      // PNG signature: 89 50 4E 47 0D 0A 1A 0A
      expect(fetchResult.first8).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

      // (2) An <img> with that URL renders.
      const imgState = await window.evaluate(async (url) => {
        const img = new Image();
        img.src = url;
        await new Promise((resolve, reject) => {
          img.onload = () => resolve(null);
          img.onerror = (e) => reject(new Error(`onerror: ${String(e)}`));
        });
        return { complete: img.complete, naturalWidth: img.naturalWidth };
      }, assetUrl);

      expect(imgState.complete).toBe(true);
      expect(imgState.naturalWidth).toBeGreaterThan(0);
    } finally {
      await electronApp.close();
    }
  });

  test("interview shows dream guide and white room when paths are set", async () => {
    const electronApp = await electron.launch({
      args: [electronMain, "--skip-warning"],
      cwd: path.join(repoRoot, "frontend"),
      env: {
        ...process.env,
        LUCIDIUM_LOAD_DIST: "1",
        LUCIDIUM_SKIP_BACKEND: "1",
      },
      timeout: 60_000,
    });

    try {
      const window = await electronApp.firstWindow({ timeout: 30_000 });
      window.on("console", (msg) =>
        console.log(`[renderer] ${msg.type()} ${msg.text()}`),
      );
      window.on("pageerror", (err) =>
        console.log(`[renderer pageerror] ${err.message}`),
      );

      // Install the WebSocket mock and minimal handlers, then reload
      // so the renderer's WS client picks them up on the next boot.
      await window.addInitScript({ content: mockWsSource });
      await window.addInitScript({
        content: `
          (() => {
            window.__lucidium.handlers = {
              "c2s/hello": function () {
                return [{ type: "s2c/hello", payload: { protocol_version: 1, has_save: false } }];
              },
              "c2s/saves/list": function () {
                return [{ type: "s2c/saves/list", payload: { saves: [] } }];
              },
              "c2s/new_game/start": function () {
                return [];  // patches are pushed manually below
              },
            };
          })();
        `,
      });
      await window.reload();
      await window.waitForLoadState("domcontentloaded");
      await expect(window.getByTestId("start-screen")).toBeVisible({ timeout: 15_000 });

      // Click into the interview, then push an interview-snapshot
      // patch directly through the mock WS as if the backend sent the
      // bundled paths.
      await window.getByRole("button", { name: "New Game" }).click();
      await window.evaluate(
        ({ dreamGuide, whiteRoom }: { dreamGuide: string; whiteRoom: string }) => {
          const w = window as unknown as {
            __lucidium?: { pushFromServer: (e: unknown) => void };
          };
          w.__lucidium?.pushFromServer({
            type: "s2c/state/patch",
            payload: {
              ops: [
                { op: "replace", path: "/interview/dream_guide_image_path", value: dreamGuide },
                { op: "replace", path: "/interview/white_room_image_path", value: whiteRoom },
                {
                  op: "replace",
                  path: "/interview/setting_options",
                  value: ["stone harbor", "neon city"],
                },
              ],
            },
          });
        },
        { dreamGuide: dreamGuidePng, whiteRoom: whiteRoomPng },
      );

      const guide = window.getByTestId("interview-dream-guide");
      const room = window.getByTestId("interview-white-room");
      await expect(guide).toBeVisible({ timeout: 5_000 });
      await expect(room).toBeAttached({ timeout: 5_000 });

      // The dream-guide <img> must actually decode — the same kind
      // of regression as #9 would surface here as naturalWidth=0.
      const guideState = await guide.evaluate((node) => {
        const img = node as HTMLImageElement;
        return {
          src: img.src,
          complete: img.complete,
          naturalWidth: img.naturalWidth,
        };
      });
      expect(guideState.src).toContain("lucidium-asset://");
      // Wait for decode if still loading.
      await window.waitForFunction(
        (testId) => {
          const img = document.querySelector(
            `[data-testid="${testId}"]`,
          ) as HTMLImageElement | null;
          return Boolean(img && img.complete && img.naturalWidth > 0);
        },
        "interview-dream-guide",
        { timeout: 10_000 },
      );
      const finalGuide = await guide.evaluate((node) => {
        const img = node as HTMLImageElement;
        return { complete: img.complete, naturalWidth: img.naturalWidth };
      });
      expect(finalGuide.complete).toBe(true);
      expect(finalGuide.naturalWidth).toBeGreaterThan(0);
    } finally {
      await electronApp.close();
    }
  });
});
