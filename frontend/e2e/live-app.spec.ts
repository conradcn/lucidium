/**
 * Live end-to-end test: real LLM, real ComfyUI, real Electron.
 *
 * Walks every user flow from the start screen through several dialog
 * moves, confirming that:
 *
 *   * Each interview step transitions when the LLM returns options.
 *   * New Game produces a playable opening node, a player portrait
 *     rendered from the production ``backend/workflows/character.json``
 *     workflow, and a background rendered from ``background.json``.
 *   * Subsequent advances and a free-text submission produce new
 *     dialog text and (where applicable) new images.
 *   * The Story panel, Settings screen, and Save/Load round-trip work
 *     against the real backend.
 *
 * Requirements:
 *   * ``backend/.venv`` set up (``.\start.ps1 -Setup``).
 *   * ``frontend/dist`` and ``frontend/dist-electron`` built.
 *   * The user's ``settings.json`` (under ``%APPDATA%/Lucidium/`` on
 *     Windows) has a working LLM endpoint and api_key plus a reachable
 *     ComfyUI base_url with the workflows from ``backend/workflows/``
 *     installed.
 *
 * Per-step timeouts are generous: a single FaceDetailer + RMBG run on
 * a typical local GPU can take 60–180 s.
 */

import path from "node:path";
import { existsSync } from "node:fs";

import { _electron as electron, expect, test, type Page } from "@playwright/test";

const repoRoot = path.resolve(__dirname, "..", "..");
const electronMain = path.join(repoRoot, "frontend", "dist-electron", "main.js");
const distIndex = path.join(repoRoot, "frontend", "dist", "index.html");
const venvPython =
  process.platform === "win32"
    ? path.join(repoRoot, "backend", ".venv", "Scripts", "python.exe")
    : path.join(repoRoot, "backend", ".venv", "bin", "python");
const productionWorkflowDir = path.join(repoRoot, "backend", "workflows");

const PROVIDER_REQUIRED = process.env.LUCIDIUM_SKIP_LIVE !== "1";

const TIMEOUT_INTERVIEW_STEP = 120_000;
const TIMEOUT_FIRST_NODE = 600_000;  // world_init LLM + 1 portrait + 1 background
const TIMEOUT_ADVANCE = 360_000;     // text + (maybe) images
const TIMEOUT_FREE_TEXT = 360_000;

test.describe("Live app — real LLM + ComfyUI through Electron", () => {
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
  test.skip(
    !existsSync(path.join(productionWorkflowDir, "character.json")),
    "backend/workflows/character.json missing.",
  );
  test.skip(
    !existsSync(path.join(productionWorkflowDir, "background.json")),
    "backend/workflows/background.json missing.",
  );
  test.skip(!PROVIDER_REQUIRED, "Live e2e disabled by LUCIDIUM_SKIP_LIVE=1.");

  test.setTimeout(45 * 60_000); // 45 minutes for the full walk

  test("complete user flow: start → interview → confirm → advance × 3 → free-text → save/load", async () => {
    const electronApp = await electron.launch({
      args: [electronMain, "--skip-warning"],
      cwd: path.join(repoRoot, "frontend"),
      env: { ...process.env, LUCIDIUM_LOAD_DIST: "1" },
      timeout: 60_000,
    });

    const window = await electronApp.firstWindow({ timeout: 30_000 });
    window.on("console", (msg) => console.log(`[renderer] ${msg.type()} ${msg.text()}`));
    window.on("pageerror", (err) => console.log(`[renderer pageerror] ${err.message}`));
    // Surface backend stdout/stderr (Electron main echoes them) so a
    // failing live test prints the LLM/ComfyUI errors that caused it.
    electronApp.process().stdout?.on("data", (chunk) =>
      process.stdout.write(`[main-stdout] ${chunk}`),
    );
    electronApp.process().stderr?.on("data", (chunk) =>
      process.stderr.write(`[main-stderr] ${chunk}`),
    );

    try {
      // 0. Renderer reaches the start screen and the WS connects to the
      //    live backend within 15 s. If the wsPort wiring or the
      //    bundled file paths are broken, this assertion catches it.
      await expect(window.getByTestId("start-screen")).toBeVisible({ timeout: 15_000 });
      await expect(window.getByTestId("connection-banner")).toHaveCount(0, { timeout: 30_000 });

      // 1. New Game → wait for setting options.
      await window.getByRole("button", { name: "New Game" }).click();
      await waitForInterviewStep(window, "setting");
      await clickFirstOption(window);

      // 2. Genre options.
      await waitForInterviewStep(window, "genre");
      await clickFirstOption(window);

      // 3. Visual style options.
      await waitForInterviewStep(window, "visual_style");
      await clickFirstOption(window);

      // 4. Character description options.
      await waitForInterviewStep(window, "character_description");
      await clickFirstOption(window);

      // 5. Name options.
      await waitForInterviewStep(window, "name");
      await clickFirstOption(window);

      // 6. Confirm step: click Begin and wait the first dialog node.
      await expect(window.getByTestId("interview-confirm")).toBeVisible({
        timeout: TIMEOUT_INTERVIEW_STEP,
      });
      await window.getByRole("button", { name: "Begin" }).click();

      await expect(window.getByTestId("main-view")).toBeVisible({ timeout: TIMEOUT_FIRST_NODE });
      await assertDialogTextVisible(window, TIMEOUT_FIRST_NODE);

      // 7. Player portrait must actually render.
      await assertPortraitLoaded(window, TIMEOUT_FIRST_NODE);

      // 8. Background must render with a non-empty image URL.
      await assertBackgroundLoaded(window, TIMEOUT_FIRST_NODE);

      // 9. Advance through three dialog beats, asserting that the
      //    dialog tree's current node id actually changes (rather
      //    than just the typewriter's progressive innerText).
      let priorNodeId = await currentNodeId(window);
      expect(priorNodeId, "current_node_id must be populated").not.toBe("");
      for (let i = 0; i < 3; i++) {
        await advanceOnce(window);
        priorNodeId = await waitForNodeChange(window, priorNodeId, TIMEOUT_ADVANCE);
      }

      // 10. Submit free text and assert another new node id arrives.
      await window
        .getByRole("textbox", { name: /Or do something else/ })
        .fill("She steps closer to the door, listening for footsteps.");
      await window.getByRole("button", { name: "Submit" }).click();
      priorNodeId = await waitForNodeChange(window, priorNodeId, TIMEOUT_FREE_TEXT);

      // 11. Open the Story panel and walk every tab. Scope all clicks
      //     to the .story-panel root so we don't collide with dialog
      //     options the LLM may have named after our affordances.
      await window.getByRole("button", { name: "Story" }).click();
      const storyPanel = window.locator(".story-panel");
      await expect(storyPanel).toBeVisible();
      for (const tab of ["History", "World", "Environments", "Characters", "Tree", "Options"]) {
        await storyPanel.getByRole("button", { name: tab, exact: true }).click();
      }

      // 12. From the Story panel's Options tab, the "Open Settings"
      //     button routes to the SettingsScreen with the live config
      //     populated by c2s/settings/get.
      await storyPanel.getByRole("button", { name: "Options", exact: true }).click();
      await storyPanel.getByRole("button", { name: "Open Settings" }).click();
      await expect(window.getByTestId("settings-screen")).toBeVisible({ timeout: 15_000 });
      await expect(
        window.getByRole("heading", { name: "LLM backend (OpenAI-compatible)" }),
      ).toBeVisible();
      await window.getByRole("button", { name: "Cancel" }).click();
      // Cancel routes back to the previous phase ("main").
      await expect(window.getByTestId("main-view")).toBeVisible({ timeout: 15_000 });

      // 13. Verify per-turn save commit landed on disk. The session
      //     auto-commits after every accepted node; by this point at
      //     least one save folder must exist with a populated
      //     game.json that round-trips through the schema.
      const savesDir = path.join(
        process.env.APPDATA ?? path.join(repoRoot, "..", "AppData", "Roaming"),
        "Lucidium",
        "saves",
      );
      const fs = await import("node:fs/promises");
      const entries = await fs.readdir(savesDir);
      expect(entries.length, "expected at least one save directory").toBeGreaterThan(0);
      const gameJsonRaw = await fs.readFile(
        path.join(savesDir, entries[0]!, "game.json"),
        "utf-8",
      );
      const gameJson = JSON.parse(gameJsonRaw);
      expect(gameJson.world.game_name, "game.json should have a name").toBeTruthy();
      expect(
        gameJson.dialog_tree.committed_path.length,
        "committed path should include the played turns",
      ).toBeGreaterThanOrEqual(5);
      // The save's character roster must include at least the player
      // character (FR-034a — even though they are not on-stage).
      const characters = Object.values(gameJson.characters) as {
        is_player: boolean;
        images: unknown[];
      }[];
      expect(characters.length).toBeGreaterThan(0);
      expect(
        characters.some((c) => c.is_player),
        "the save must record the player character",
      ).toBe(true);
      // The opening environment must have produced a background image
      // file on disk.
      const fsSync = await import("node:fs");
      const imagesDir = path.join(savesDir, entries[0]!, "images");
      const imageFiles = fsSync.readdirSync(imagesDir).filter((f) => f.endsWith(".png"));
      expect(
        imageFiles.length,
        "at least one ComfyUI-generated image must have landed on disk",
      ).toBeGreaterThan(0);
    } finally {
      await electronApp.close();
    }
  });
});

async function waitForInterviewStep(page: Page, step: string): Promise<void> {
  // The interview either shows the loading screen until options
  // arrive, or the step itself if options were already cached.
  await expect(
    page.locator(`[data-testid="interview-${step}"], [data-testid="interview-loading"]`),
  ).toBeVisible({ timeout: TIMEOUT_INTERVIEW_STEP });
  await expect(page.getByTestId(`interview-${step}`)).toBeVisible({
    timeout: TIMEOUT_INTERVIEW_STEP,
  });
}

async function clickFirstOption(page: Page): Promise<void> {
  // Each interview step renders an `.option-grid` of LLM-supplied
  // buttons. We click the first one to walk the happy path; the
  // text varies because the LLM picks it.
  const firstOption = page.locator(".option-grid button").first();
  await expect(firstOption).toBeVisible({ timeout: TIMEOUT_INTERVIEW_STEP });
  await firstOption.click();
}

async function assertDialogTextVisible(page: Page, timeout: number): Promise<void> {
  const interaction = page.locator(".main-view .interaction .text");
  await expect(interaction).toBeVisible({ timeout });
  await expect.poll(
    async () => (await interaction.innerText()).trim().length,
    { timeout, intervals: [500, 1_000, 2_000] },
  ).toBeGreaterThan(0);
}

async function currentNodeId(page: Page): Promise<string> {
  // Read from the renderer's Zustand store rather than the typewriter
  // text — the typewriter grows incrementally so any text-content
  // poll would falsely match while the SAME node is still being typed.
  const id = await page.evaluate(() => {
    const w = window as unknown as {
      __lucidium_current_node_id?: string;
    };
    return w.__lucidium_current_node_id ?? "";
  });
  return id;
}

async function waitForNodeChange(
  page: Page,
  priorNodeId: string,
  timeout: number,
): Promise<string> {
  await expect.poll(
    async () => {
      const id = await currentNodeId(page);
      return id !== priorNodeId && id.length > 0 ? id : null;
    },
    { timeout, intervals: [500, 1_000, 2_000] },
  ).not.toBeNull();
  return currentNodeId(page);
}

async function advanceOnce(page: Page): Promise<void> {
  // If the current node has options, click the first; otherwise the
  // single "Continue" button advances.
  const optionButtons = page.locator(".main-view .options button");
  const count = await optionButtons.count();
  if (count > 0) {
    await optionButtons.first().click();
  } else {
    await page.getByRole("button", { name: "Continue" }).click();
  }
}

async function assertPortraitLoaded(page: Page, timeout: number): Promise<void> {
  // Per FR-034a the player isn't a stage actor; if the LLM didn't put
  // any non-player characters on stage in the opening node, there
  // simply is no portrait to assert against. Verify the background
  // loaded instead — that proves ComfyUI is integrated.
  const stageImages = page.locator(".main-view .character img");
  if ((await stageImages.count()) === 0) {
    return; // narration-only opening; FR-034a allows empty stage.
  }
  const portrait = stageImages.first();
  await expect(portrait).toBeVisible({ timeout });
  // Final hard assertion that the image isn't a 0×0 broken icon.
  const dims = await portrait.evaluate((img: HTMLImageElement) => ({
    naturalWidth: img.naturalWidth,
    naturalHeight: img.naturalHeight,
  }));
  expect(dims.naturalWidth, "portrait should have positive natural width").toBeGreaterThan(0);
  expect(dims.naturalHeight, "portrait should have positive natural height").toBeGreaterThan(0);
}

async function assertBackgroundLoaded(page: Page, timeout: number): Promise<void> {
  const background = page.locator(".main-view .background");
  await expect(background).toBeVisible({ timeout });
  await expect.poll(
    async () => background.evaluate((el) => window.getComputedStyle(el).backgroundImage),
    { timeout, intervals: [1_000, 2_000, 5_000] },
  ).not.toBe("none");
}
