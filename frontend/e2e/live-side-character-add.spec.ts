/**
 * Reproduces the user-reported "Add side character grays out the
 * field but doesn't add anyone to the game" bug.
 *
 * Walks the interview to Review, types a side-character
 * description, clicks Add, and asserts:
 *   1. The text entry stays disabled while the LLM is in flight
 *      (this is by design — useOptimisticAction guards against
 *      double-submit until the snapshot's side_characters list
 *      grows).
 *   2. Within 60 s, a list item with a non-empty name appears
 *      under the "Side characters" heading AND the text entry
 *      re-enables.
 *
 * If the assertion fails, the test dumps the relevant backend
 * lines (the new instrumentation in
 * ``new_game_add_side_character_handler`` logs the request, the
 * raw LLM payload on validation failure, and the success append)
 * so the fix path is one log read away.
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
  repoRoot, "frontend", "test-results", "side-character-add",
);

test.describe("Live New Game — Add side character", () => {
  test.skip(!existsSync(electronMain), "Electron main not built");
  test.skip(!existsSync(distIndex), "Renderer dist not built");
  test.skip(!existsSync(venvPython), "Backend venv missing");
  test.skip(
    process.env.LUCIDIUM_SKIP_LIVE === "1",
    "Live e2e disabled by LUCIDIUM_SKIP_LIVE=1",
  );

  test.setTimeout(15 * 60_000);

  test("adding a side character at Review surfaces the new name in the list", async () => {
    mkdirSync(SCREENSHOT_DIR, { recursive: true });
    const electronApp = await electron.launch({
      args: [electronMain, "--skip-warning"],
      cwd: path.join(repoRoot, "frontend"),
      env: { ...process.env, LUCIDIUM_LOAD_DIST: "1" },
      timeout: 60_000,
    });
    const backendTail: string[] = [];
    const TAIL_KEEP = 600;
    try {
      const window = await electronApp.firstWindow({ timeout: 30_000 });
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

      const startupWarning = window.getByTestId("startup-warning");
      if (await startupWarning.count()) {
        await window
          .getByRole("button", { name: "I understand" })
          .click();
        await expect(startupWarning).toHaveCount(0, { timeout: 5_000 });
      }

      // New Game → walk every step by clicking the FIRST option-grid
      // button (covers setting / visual_style / genre /
      // character_description / name). Each transition has a
      // ~120 s budget for the LLM-driven options to populate.
      await window.getByRole("button", { name: "New Game" }).click();
      const interviewSteps = [
        "setting", "visual_style", "genre",
        "character_description", "name",
      ];
      for (const stepName of interviewSteps) {
        await window
          .locator(".option-grid button")
          .first()
          .waitFor({ state: "visible", timeout: 120_000 });
        await expect(window.locator(".option-grid button").first())
          .toBeEnabled({ timeout: 5_000 });
        await window.locator(".option-grid button").first().click();
        console.log(`[test] clicked first option for ${stepName}`);
      }

      // Review screen.
      await expect(window.getByTestId("interview-confirm"))
        .toBeVisible({ timeout: 60_000 });
      await window.screenshot({
        path: path.join(SCREENSHOT_DIR, "01-review-empty.png"),
        fullPage: true,
      });

      // Type a side character description, click Add.
      const sideText = "the gruff retired bounty hunter who runs the dock bar";
      const sideInput = window.getByTestId("side-character-description");
      await sideInput.fill(sideText);
      const addButton = window.getByRole("button", { name: "Add" });
      await expect(addButton).toBeEnabled({ timeout: 5_000 });
      await addButton.click();

      // The optimistic guard disables both immediately; verify it
      // happens (the user's reported "grays out the field" symptom
      // is the EXPECTED first half — the bug is whether it RE-
      // enables and shows the new name).
      await expect(sideInput).toBeDisabled({ timeout: 2_000 });
      console.log(`[test] field grayed — waiting for backend echo`);

      // The new character must appear in ``.side-character-list``
      // AND the text entry must re-enable. Both conditions track
      // the snapshot's side_characters list growing.
      try {
        await expect(window.locator(".side-character-list li").first())
          .toBeVisible({ timeout: 90_000 });
        await expect(sideInput).toBeEnabled({ timeout: 30_000 });
      } catch (err) {
        await window.screenshot({
          path: path.join(SCREENSHOT_DIR, "02-FAIL-stuck-state.png"),
          fullPage: true,
        });
        // Filter the captured backend tail for the relevant lines.
        const interesting = backendTail.filter((line) =>
          /add_side_character|LlmCharacterPayload|state_patch|s2c\/state\/patch|s2c\/error|new_game/.test(line),
        );
        const tail = interesting.slice(-80).join("\n");
        throw new Error(
          `Side character did not appear within 90 s after click.\n`
          + `--- relevant backend lines ---\n${tail || "(none captured)"}\n`
          + `--- end ---\n`
          + `Underlying: ${err instanceof Error ? err.message : String(err)}`,
        );
      }

      const visibleNames = await window
        .locator(".side-character-list li")
        .allTextContents();
      console.log(`[OK] side characters visible: ${JSON.stringify(visibleNames)}`);
      expect(visibleNames.length).toBeGreaterThan(0);
      expect(visibleNames[0]?.trim().length ?? 0).toBeGreaterThan(0);
      await window.screenshot({
        path: path.join(SCREENSHOT_DIR, "03-side-character-listed.png"),
        fullPage: true,
      });
    } finally {
      await electronApp.close();
    }
  });
});
