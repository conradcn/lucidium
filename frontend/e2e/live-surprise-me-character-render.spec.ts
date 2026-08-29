/**
 * Reproduces the user-reported "characters not always rendering"
 * bug. Surprise Me path → wait for main view → assert that at
 * least one on-stage character has a non-placeholder portrait
 * image with non-zero naturalWidth (i.e., the file actually
 * loaded in the renderer). Dumps a screenshot either way for
 * visual confirmation.
 *
 * Background: backend logs show
 *   image render START
 *   image-call workflow=character.json
 *   (silence — no completion, no error)
 * The character stays in placeholder state forever. With the
 * embedded_image_client step traces just added, the LAST step=
 * line emitted before the silence will identify which gate
 * (restore_to_gpu / lock acquire / _run_pipeline) is stuck.
 *
 * Run:
 *   LIVE_E2E_TURNS=0 npx playwright test live-surprise-me-character-render
 */
import path from "node:path";
import { existsSync, mkdirSync } from "node:fs";

import { _electron as electron, expect, test } from "@playwright/test";

const repoRoot = path.resolve(__dirname, "..", "..");
const electronMain = path.join(repoRoot, "frontend", "dist-electron", "main.js");
const distIndex = path.join(repoRoot, "frontend", "dist", "index.html");
const venvPython =
  process.platform === "win32"
    ? path.join(repoRoot, "backend", ".venv", "Scripts", "python.exe")
    : path.join(repoRoot, "backend", ".venv", "bin", "python");

const SCREENSHOT_DIR = path.join(
  repoRoot, "frontend", "test-results", "surprise-me-character-render",
);

test.describe("Live Surprise Me — character render check", () => {
  test.skip(!existsSync(electronMain), "Electron main not built");
  test.skip(!existsSync(distIndex), "Renderer dist not built");
  test.skip(!existsSync(venvPython), "Backend venv missing");
  test.skip(
    process.env.LUCIDIUM_SKIP_LIVE === "1",
    "Live e2e disabled by LUCIDIUM_SKIP_LIVE=1",
  );

  test.setTimeout(10 * 60_000);

  test("Surprise Me brings up a character portrait within 4 minutes", async () => {
    mkdirSync(SCREENSHOT_DIR, { recursive: true });
    const electronApp = await electron.launch({
      args: [electronMain, "--skip-warning"],
      cwd: path.join(repoRoot, "frontend"),
      env: { ...process.env, LUCIDIUM_LOAD_DIST: "1" },
      timeout: 60_000,
    });
    // Last few hundred lines of the backend tail — printed on
    // failure so the diagnostic step= traces are visible without
    // having to re-tail session.log by hand.
    const backendTail: string[] = [];
    const TAIL_KEEP = 400;
    try {
      const window = await electronApp.firstWindow({ timeout: 30_000 });
      window.on("console", (msg) => {
        const text = msg.text();
        // Only echo errors / warnings — keep stdout grep-friendly.
        if (msg.type() === "error" || msg.type() === "warning") {
          console.log(`[renderer] ${msg.type()} ${text}`);
        }
      });
      window.on("pageerror", (err) => {
        console.log(`[renderer pageerror] ${err.message}`);
      });
      electronApp.process().stderr?.on("data", (chunk) => {
        const text = chunk.toString();
        process.stderr.write(`[main-stderr] ${text}`);
        for (const line of text.split(/\r?\n/)) {
          if (!line.trim()) continue;
          backendTail.push(line);
          if (backendTail.length > TAIL_KEEP) backendTail.shift();
        }
      });
      electronApp.process().stdout?.on("data", (chunk) =>
        process.stdout.write(`[main-stdout] ${chunk}`),
      );

      await expect(window.getByTestId("start-screen")).toBeVisible({
        timeout: 30_000,
      });
      await expect(window.getByTestId("connection-banner")).toHaveCount(0, {
        timeout: 30_000,
      });

      // Dismiss the per-session safety modal if present.
      const startupWarning = window.getByTestId("startup-warning");
      if (await startupWarning.count()) {
        await window
          .getByRole("button", { name: "I understand" })
          .click();
        await expect(startupWarning).toHaveCount(0, { timeout: 5_000 });
      }

      // Click Surprise Me. If the player profile is empty, a nudge
      // modal pops — fail-fast with a clear message instead of
      // silently waiting for main-view that will never arrive.
      const surprise = window.getByRole("button", { name: "Surprise Me" });
      await expect(surprise).toBeVisible({ timeout: 10_000 });
      await surprise.click();
      const nudge = window.locator(".surprise-me-nudge");
      const showedNudge = await nudge.count();
      if (showedNudge > 0) {
        await window.screenshot({
          path: path.join(SCREENSHOT_DIR, "01-empty-profile-nudge.png"),
          fullPage: true,
        });
        throw new Error(
          "Surprise Me triggered the empty-profile nudge — populate "
          + "user_profile in settings.json (likes/dislikes/notes) so the "
          + "scenario LLM call has something to work against, then re-run.",
        );
      }

      // Wait for main view. World_init is the bottleneck on slow
      // openrouter days (~30 s; saw 110 s once); 6 min budget.
      await window.screenshot({
        path: path.join(SCREENSHOT_DIR, "02-after-surprise-click.png"),
        fullPage: true,
      });
      await expect(window.getByTestId("main-view")).toBeVisible({
        timeout: 6 * 60_000,
      });
      await window.screenshot({
        path: path.join(SCREENSHOT_DIR, "03-main-view-landed.png"),
        fullPage: true,
      });

      // Now the actual assertion: at least one on-stage character's
      // <img> must have loaded (naturalWidth > 0). Image gen takes
      // ~15 s of fresh GPU time on a 4090; 4 min budget gives
      // headroom for the eviction-restore cycle that comes after
      // the music-server fail-fast.
      const characterRendered = async (): Promise<{
        loaded: boolean;
        characters: Array<{ name: string; src: string; loaded: boolean }>;
      }> => {
        return await window.evaluate(() => {
          const stage = document.querySelector(".main-view .stage");
          if (!stage) return { loaded: false, characters: [] };
          const characters = Array.from(stage.querySelectorAll(".character"));
          const summaries = characters.map((el) => {
            const nameEl = el.querySelector(".name");
            const img = el.querySelector("img") as HTMLImageElement | null;
            return {
              name: nameEl?.textContent?.trim() ?? "(no-name)",
              src: img?.src ?? "",
              loaded: img?.complete === true && (img?.naturalWidth ?? 0) > 0,
            };
          });
          return {
            loaded: summaries.some((c) => c.loaded),
            characters: summaries,
          };
        });
      };

      let lastSnapshot: Awaited<ReturnType<typeof characterRendered>> | null = null;
      try {
        await expect
          .poll(
            async () => {
              lastSnapshot = await characterRendered();
              return lastSnapshot.loaded;
            },
            { timeout: 4 * 60_000, intervals: [1_000, 2_000, 5_000] },
          )
          .toBe(true);
      } catch (err) {
        await window.screenshot({
          path: path.join(SCREENSHOT_DIR, "04-FAIL-no-character.png"),
          fullPage: true,
        });
        const summary = lastSnapshot
          ? lastSnapshot.characters
              .map((c) => `${c.name} src=${c.src.slice(-40)} loaded=${c.loaded}`)
              .join(" | ")
          : "(no snapshot)";
        const tail = backendTail.slice(-150).join("\n");
        throw new Error(
          `No on-stage character portrait loaded within 4 min.\n`
          + `Stage state: ${summary || "(no .character nodes)"}\n`
          + `--- backend tail (last 150 lines) ---\n${tail}\n`
          + `--- end backend tail ---\n`
          + `Underlying: ${err instanceof Error ? err.message : String(err)}`,
        );
      }

      // Success — grab the final screenshot for visual confirmation.
      await window.screenshot({
        path: path.join(SCREENSHOT_DIR, "05-character-loaded.png"),
        fullPage: true,
      });
      // Print the snapshot so the test report shows which character
      // landed and what the path was.
      console.log(`[OK] character rendered: ${JSON.stringify(lastSnapshot)}`);
    } finally {
      await electronApp.close();
    }
  });
});
