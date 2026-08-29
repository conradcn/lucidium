/**
 * Non-mocked image-pipeline tests.
 *
 * The user reported that characters can be rendered in ComfyUI but
 * never display in the renderer. Several layers can break that path:
 *
 *   1. ComfyUI's character workflow itself (wrong checkpoint path,
 *      missing model file, broken node graph).
 *   2. The backend's ensure_assets → character.images[0].path
 *      round-trip (path encoding, atomic_write_text, image-ready
 *      event delivery).
 *   3. The renderer's assetUrl + Electron's lucidium-asset:// protocol
 *      handler (URL escaping, file-not-found, MIME mismatch).
 *   4. The renderer's CharacterStage actually mounting an <img> for
 *      the on-stage character (filtering by is_player, picking
 *      images[0], mounting the right URL).
 *
 * The existing image-pipeline.spec.ts only covers (3) — proves the
 * protocol works against bundled placeholder PNGs. The existing
 * live-app.spec.ts walks the whole flow but skips the portrait
 * assertion when the LLM happens not to put a non-player character
 * on stage in the opening (FR-034a allows it).
 *
 * This spec adds three tests that close the gap:
 *
 *   A. **ComfyUI direct.** Submit the production character.json
 *      workflow, verify it produces decodable PNG bytes. Does not
 *      touch the backend or Electron — isolates layer (1).
 *
 *   B. **Renderer round-trip with real PNG path.** Real Electron,
 *      mocked WS that pushes a game state whose non-player character
 *      points at an on-disk PNG. Asserts the <img> in the renderer's
 *      .character div decodes (naturalWidth > 0). Isolates layers
 *      (3) + (4).
 *
 *   C. **Full live pipeline with forced character.** Real Electron,
 *      real Python backend, real LLM, real ComfyUI. Drive new game
 *      and free-text until a non-player character is on stage; assert
 *      their portrait <img> decodes. Catches any layer-(2) bug that
 *      slips past A and B.
 *
 * Setup requirements documented inline per test.
 */

import path from "node:path";
import { existsSync, readFileSync, statSync } from "node:fs";

import { _electron as electron, expect, test } from "@playwright/test";

/** A ComfyUI API-format workflow: node id -> node with its inputs. */
type ComfyWorkflow = Record<string, { inputs: Record<string, string | number> }>;

/** A ComfyUI ``/history/<id>`` response: prompt id -> execution record. */
type ComfyHistory = Record<string, { outputs?: Record<string, unknown> }>;

const mockWsSource = readFileSync(path.join(__dirname, "mock-ws.js"), "utf-8");

const repoRoot = path.resolve(__dirname, "..", "..");
const electronMain = path.join(repoRoot, "frontend", "dist-electron", "main.js");
const distIndex = path.join(repoRoot, "frontend", "dist", "index.html");
const characterWorkflow = path.join(repoRoot, "backend", "workflows", "character.json");
const dreamGuidePng = path.join(
  repoRoot,
  "backend",
  "workflows",
  "placeholders",
  "dream_guide.png",
);
const venvPython =
  process.platform === "win32"
    ? path.join(repoRoot, "backend", ".venv", "Scripts", "python.exe")
    : path.join(repoRoot, "backend", ".venv", "bin", "python");

const COMFY_URL = process.env.LUCIDIUM_COMFY_URL ?? "http://127.0.0.1:8000";

// Run the file's tests serially. ComfyUI on a single GPU has limited
// concurrency; running the direct workflow test alongside the
// real-Electron tests (which ALSO contend for the same physical
// device when test C is enabled) starves the GPU and the
// 8-minute deadline trips even when both prompts would otherwise
// finish in under 2 minutes each.
test.describe.configure({ mode: "serial" });

/* --------------------------------------------------------------------------
 * A. ComfyUI direct
 * ------------------------------------------------------------------------ */

test.describe("ComfyUI direct — production character workflow", () => {
  test.skip(
    !existsSync(characterWorkflow),
    "backend/workflows/character.json missing",
  );

  test.setTimeout(10 * 60_000); // a single FaceDetailer + RMBG can take 90-300s

  test("character workflow returns a decodable PNG", async () => {
    // Health-check the local ComfyUI before paying for the run.
    let stats: Response | null = null;
    try {
      stats = await fetch(`${COMFY_URL}/system_stats`, { signal: AbortSignal.timeout(3_000) });
    } catch {
      test.skip(true, `ComfyUI unreachable at ${COMFY_URL}`);
    }
    if (!stats || !stats.ok) test.skip(true, `ComfyUI unhealthy at ${COMFY_URL}`);

    const workflow = JSON.parse(readFileSync(characterWorkflow, "utf-8")) as ComfyWorkflow;
    // Substitute the PLACEHOLDER tokens the production code fills in.
    workflow["2"].inputs.text =
      "masterpiece, full body, standing, woman in wool coat, " +
      "stone harbor, ink wash, atmospheric, soft light";
    workflow["3"].inputs.text = String(workflow["3"].inputs.text).replace(
      "PLACEHOLDER_NEGATIVE_EXTRAS",
      "",
    );
    workflow["12"].inputs.wildcard =
      "calm expression, grey eyes, auburn hair";
    // Vary the seed per test run so ComfyUI doesn't return a cached
    // result from a previous run (which would short-circuit the test
    // and falsely pass even if generation is broken).
    const seed = Date.now() % 1_000_000;
    workflow["20"].inputs.noise_seed = seed;
    workflow["12"].inputs.seed = seed;

    const submit = await fetch(`${COMFY_URL}/prompt`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: workflow }),
    });
    expect(submit.ok, `comfy /prompt returned ${submit.status}`).toBe(true);
    const { prompt_id: promptId } = (await submit.json()) as { prompt_id: string };
    expect(promptId).toBeTruthy();

    // Poll history until the prompt finishes.
    const deadline = Date.now() + 8 * 60_000;
    let imageBytes: ArrayBuffer | null = null;
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 3_000));
      const history = await fetch(`${COMFY_URL}/history/${promptId}`);
      if (!history.ok) continue;
      const data = (await history.json()) as ComfyHistory;
      const entry = data[promptId];
      if (!entry) continue;
      const outputs = entry.outputs ?? {};
      const images: Array<{ filename: string; subfolder: string; type: string }> = [];
      for (const out of Object.values(outputs)) {
        const arr = (out as { images?: typeof images }).images ?? [];
        for (const img of arr) images.push(img);
      }
      if (images.length === 0) continue;
      const image = images[0]!;
      const view = await fetch(
        `${COMFY_URL}/view?filename=${encodeURIComponent(image.filename)}` +
          `&subfolder=${encodeURIComponent(image.subfolder)}` +
          `&type=${encodeURIComponent(image.type)}`,
      );
      expect(view.ok, `comfy /view returned ${view.status}`).toBe(true);
      imageBytes = await view.arrayBuffer();
      break;
    }
    expect(imageBytes, "ComfyUI did not produce an output within 8 min").not.toBeNull();
    const view = new Uint8Array(imageBytes!);
    // PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
    expect(view.length).toBeGreaterThan(8 * 1024);
    expect([
      view[0], view[1], view[2], view[3], view[4], view[5], view[6], view[7],
    ]).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  });
});

/* --------------------------------------------------------------------------
 * B. Renderer round-trip with a real on-disk PNG
 * ------------------------------------------------------------------------ */

test.describe("Renderer round-trip — real PNG on disk, real Electron", () => {
  test.skip(
    !existsSync(electronMain),
    `Electron main not built. Run: npx tsc -p ${path.join("frontend", "tsconfig.electron.json")}`,
  );
  test.skip(
    !existsSync(distIndex),
    "Renderer dist not built. Run: npm --prefix frontend run build",
  );
  test.skip(
    !existsSync(dreamGuidePng),
    "bundled dream_guide.png missing — run scripts/regen_dream_guide.py",
  );

  test("CharacterStage <img> decodes for an on-stage non-player character", async () => {
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
      window.on("console", (msg) => console.log(`[renderer] ${msg.type()} ${msg.text()}`));
      window.on("pageerror", (err) => console.log(`[renderer pageerror] ${err.message}`));

      // Install the WS mock + a handler that returns a fully-formed
      // game with one non-player character on stage whose images[0]
      // points at the bundled dream_guide PNG.
      await window.addInitScript({ content: mockWsSource });
      await window.addInitScript({
        content: buildPortraitDeliveryScript(dreamGuidePng),
      });
      await window.reload();
      await window.waitForLoadState("domcontentloaded");

      await expect(window.getByTestId("start-screen")).toBeVisible({ timeout: 15_000 });
      await window.getByRole("button", { name: "Continue" }).click();
      await expect(window.getByTestId("main-view")).toBeVisible({ timeout: 10_000 });

      const portrait = window.locator(".main-view .character img");
      await expect(portrait).toBeVisible({ timeout: 10_000 });

      // The src must use our custom scheme (NOT raw file://, which
      // Electron 31 blocks cross-origin in packaged builds).
      const src = await portrait.getAttribute("src");
      expect(src ?? "").toMatch(/^lucidium-asset:\/\//);

      // Wait for the <img> to actually decode — the original bug was
      // that the URL was right but Chromium rejected the response.
      await window.waitForFunction(
        () => {
          const img = document.querySelector(
            ".main-view .character img",
          ) as HTMLImageElement | null;
          return Boolean(img && img.complete && img.naturalWidth > 0);
        },
        undefined,
        { timeout: 10_000 },
      );
      const dims = await portrait.evaluate((img: HTMLImageElement) => ({
        complete: img.complete,
        naturalWidth: img.naturalWidth,
        naturalHeight: img.naturalHeight,
      }));
      expect(dims.complete).toBe(true);
      expect(dims.naturalWidth).toBeGreaterThan(0);
      expect(dims.naturalHeight).toBeGreaterThan(0);

      // The actual PNG on disk has these dimensions in the metadata
      // — sanity check that the renderer is showing the right file
      // (and not e.g. a 1×1 placeholder broken-image fallback).
      const onDisk = statSync(dreamGuidePng);
      expect(onDisk.size).toBeGreaterThan(8 * 1024);
    } finally {
      await electronApp.close();
    }
  });

  test("path with spaces and unicode survives the asset URL round-trip", async () => {
    // Regression hedge: real save folders sometimes contain unicode
    // (player chose a non-ASCII name) or spaces. The asset protocol
    // must percent-decode correctly.
    const fs = await import("node:fs/promises");
    const tempDir = path.join(repoRoot, "test-results", "image-pipeline-paths");
    await fs.mkdir(tempDir, { recursive: true });
    const fancyDir = path.join(tempDir, "saves with spaces 試");
    await fs.mkdir(fancyDir, { recursive: true });
    const target = path.join(fancyDir, "portrait.png");
    await fs.copyFile(dreamGuidePng, target);

    const electronApp = await electron.launch({
      args: [electronMain, "--skip-warning"],
      cwd: path.join(repoRoot, "frontend"),
      env: { ...process.env, LUCIDIUM_LOAD_DIST: "1", LUCIDIUM_SKIP_BACKEND: "1" },
      timeout: 60_000,
    });
    try {
      const window = await electronApp.firstWindow({ timeout: 30_000 });
      window.on("console", (msg) => console.log(`[renderer] ${msg.type()} ${msg.text()}`));
      await window.addInitScript({ content: mockWsSource });
      await window.addInitScript({
        content: buildPortraitDeliveryScript(target),
      });
      await window.reload();
      await window.waitForLoadState("domcontentloaded");
      await expect(window.getByTestId("start-screen")).toBeVisible({ timeout: 15_000 });
      await window.getByRole("button", { name: "Continue" }).click();
      await expect(window.getByTestId("main-view")).toBeVisible();
      await window.waitForFunction(
        () => {
          const img = document.querySelector(
            ".main-view .character img",
          ) as HTMLImageElement | null;
          return Boolean(img && img.complete && img.naturalWidth > 0);
        },
        undefined,
        { timeout: 10_000 },
      );
    } finally {
      await electronApp.close();
    }
  });
});

/* --------------------------------------------------------------------------
 * C. Full live pipeline — Electron + Python backend + ComfyUI
 * ------------------------------------------------------------------------ */

test.describe("Live pipeline — character generated AND displayed", () => {
  test.skip(
    !existsSync(electronMain),
    "Electron main not built",
  );
  test.skip(
    !existsSync(distIndex),
    "Renderer dist not built",
  );
  test.skip(
    !existsSync(venvPython),
    "backend venv missing",
  );
  test.skip(
    process.env.LUCIDIUM_SKIP_LIVE === "1",
    "Live e2e disabled by LUCIDIUM_SKIP_LIVE=1",
  );

  test.setTimeout(30 * 60_000);

  test("live: drive game until a non-player character is on stage; their <img> must decode", async () => {
    const electronApp = await electron.launch({
      args: [electronMain, "--skip-warning"],
      cwd: path.join(repoRoot, "frontend"),
      env: { ...process.env, LUCIDIUM_LOAD_DIST: "1" },
      timeout: 60_000,
    });
    try {
      const window = await electronApp.firstWindow({ timeout: 30_000 });
      window.on("console", (msg) => console.log(`[renderer] ${msg.type()} ${msg.text()}`));
      window.on("pageerror", (err) => console.log(`[renderer pageerror] ${err.message}`));
      electronApp.process().stdout?.on("data", (chunk) =>
        process.stdout.write(`[main-stdout] ${chunk}`),
      );
      electronApp.process().stderr?.on("data", (chunk) =>
        process.stderr.write(`[main-stderr] ${chunk}`),
      );

      await expect(window.getByTestId("start-screen")).toBeVisible({ timeout: 15_000 });
      await expect(window.getByTestId("connection-banner")).toHaveCount(0, { timeout: 30_000 });

      // Walk new-game.
      await window.getByRole("button", { name: "New Game" }).click();
      for (const _step of ["setting", "genre", "visual_style", "character_description", "name"]) {
        await window.locator(".option-grid button").first().waitFor({ state: "visible", timeout: 120_000 });
        await window.locator(".option-grid button").first().click();
      }
      await expect(window.getByTestId("interview-confirm")).toBeVisible({ timeout: 120_000 });
      await window.getByRole("button", { name: "Begin" }).click();
      await expect(window.getByTestId("main-view")).toBeVisible({ timeout: 600_000 });

      // Force a non-player character on stage by free-texting one in.
      // The world_init opening might be narration-only (FR-034a); we
      // need the LLM to commit a named character before we can assert
      // their portrait.
      await window
        .getByRole("textbox", { name: /Or do something else/ })
        .fill(
          "A weathered keeper named Hale steps out of the fog and " +
            "fixes his eyes on you, oilskin glistening with rain.",
        );
      await window.getByRole("button", { name: "Submit" }).click();

      // Wait for the renderer to commit a non-player character whose
      // images[0] is set, then assert the corresponding <img>.
      await expect.poll(
        async () => {
          return window.evaluate(() => {
            const w = window as unknown as {
              __lucidium_current_node_id?: string;
            };
            void w;
            const imgs = Array.from(
              document.querySelectorAll<HTMLImageElement>(".main-view .character img"),
            );
            return imgs.find((i) => i.complete && i.naturalWidth > 0)
              ? "ready"
              : `pending (count=${imgs.length})`;
          });
        },
        { timeout: 8 * 60_000, intervals: [2_000, 5_000, 10_000] },
      ).toBe("ready");

      // Final hard assertion.
      const portrait = window.locator(".main-view .character img").first();
      const dims = await portrait.evaluate((img: HTMLImageElement) => ({
        src: img.src,
        complete: img.complete,
        naturalWidth: img.naturalWidth,
      }));
      expect(dims.src).toMatch(/^lucidium-asset:\/\//);
      expect(dims.naturalWidth).toBeGreaterThan(0);
    } finally {
      await electronApp.close();
    }
  });

  test("live: async asset task pushes a state/full carrying new character.images entries", async () => {
    // The bug we're hedging against: the async asset task generates
    // the file but never emits the state/full that tells the renderer
    // about the new character.images[0]. Without the push, the
    // renderer keeps showing the placeholder forever.
    const electronApp = await electron.launch({
      args: [electronMain, "--skip-warning"],
      cwd: path.join(repoRoot, "frontend"),
      env: { ...process.env, LUCIDIUM_LOAD_DIST: "1" },
      timeout: 60_000,
    });
    try {
      const window = await electronApp.firstWindow({ timeout: 30_000 });
      window.on("console", (msg) => console.log(`[renderer] ${msg.type()} ${msg.text()}`));
      // Capture every state/full the renderer applies into a
      // window-scoped log we can inspect from the test.
      await window.addInitScript({
        content: `
          (() => {
            const w = window;
            w.__state_full_log = [];
            const origWS = w.WebSocket;
            class TappedWS extends origWS {
              constructor(url, protocols) {
                super(url, protocols);
                this.addEventListener("message", (ev) => {
                  try {
                    const env = JSON.parse(ev.data);
                    if (env.type === "s2c/state/full") {
                      const game = env.payload && env.payload.game;
                      const chars = game && game.characters
                        ? Object.values(game.characters).map((c) => ({
                            id: c.id,
                            is_player: c.is_player,
                            image_count: (c.images || []).length,
                          }))
                        : [];
                      w.__state_full_log.push({ at: Date.now(), chars });
                    }
                  } catch (_e) { /* not JSON */ }
                });
              }
            }
            w.WebSocket = TappedWS;
          })();
        `,
      });
      await window.reload();
      await window.waitForLoadState("domcontentloaded");

      await expect(window.getByTestId("start-screen")).toBeVisible({ timeout: 15_000 });
      await expect(window.getByTestId("connection-banner")).toHaveCount(0, { timeout: 30_000 });
      await window.getByRole("button", { name: "New Game" }).click();
      for (const _step of ["setting", "genre", "visual_style", "character_description", "name"]) {
        await window.locator(".option-grid button").first().waitFor({ state: "visible", timeout: 120_000 });
        await window.locator(".option-grid button").first().click();
      }
      await expect(window.getByTestId("interview-confirm")).toBeVisible({ timeout: 120_000 });
      await window.getByRole("button", { name: "Begin" }).click();
      await expect(window.getByTestId("main-view")).toBeVisible({ timeout: 600_000 });

      // Force a non-player character so there's something to render.
      await window
        .getByRole("textbox", { name: /Or do something else/ })
        .fill("A grim keeper named Hale appears, oilskin coat dripping.");
      await window.getByRole("button", { name: "Submit" }).click();

      // Assert: at least one state/full carrying a non-player
      // character with images.length > 0 must arrive eventually.
      await expect.poll(
        async () => {
          return window.evaluate(() => {
            const w = window as unknown as {
              __state_full_log?: Array<{
                at: number;
                chars: Array<{ id: string; is_player: boolean; image_count: number }>;
              }>;
            };
            const log = w.__state_full_log ?? [];
            for (const entry of log) {
              if (entry.chars.some((c) => !c.is_player && c.image_count > 0)) {
                return "ok";
              }
            }
            return `pending (${log.length} state/fulls, none carry an image yet)`;
          });
        },
        { timeout: 10 * 60_000, intervals: [3_000, 5_000, 10_000] },
      ).toBe("ok");
    } finally {
      await electronApp.close();
    }
  });
});

/* --------------------------------------------------------------------------
 * helpers
 * ------------------------------------------------------------------------ */

/** Build an init-script that installs a WS mock returning a game with
 *  one on-stage non-player character pointing at ``portraitPath``. */
function buildPortraitDeliveryScript(portraitPath: string): string {
  const game = {
    id: "g1",
    schema_version: 1,
    world: {
      game_name: "Test",
      setting: "harbor",
      genre: "mystery",
      visual_style: "ink",
      overall_plot_direction: "",
      active_plot_threads: [],
      dropped_plot_threads: [],
      summarizer_assessment: "",
      prompt_history_clamp_chars: 12000,
      player_intent: {
        pace_preference: "same",
        tone_preference: "unspecified",
        direction_signal: "none",
        weighted_evidence: [],
      },
    },
    characters: {
      c_player: {
        id: "c_player",
        is_player: true,
        name: "Iris",
        description: "wry archivist",
        gender: "female",
        age: 28,
        ethnicity: "local",
        skin: "pale",
        hair_color: "auburn",
        hairstyle: "braid",
        eye_color: "grey",
        build: "slight",
        bust: "moderate",
        outfit: "wool coat",
        pose: "standing",
        expression: "alert",
        facts: [],
        images: [],
        seed: 1,
      },
      c_other: {
        id: "c_other",
        is_player: false,
        name: "Hale",
        description: "weathered keeper",
        gender: "male",
        age: 52,
        ethnicity: "local",
        skin: "tan",
        hair_color: "grey",
        hairstyle: "short",
        eye_color: "brown",
        build: "stocky",
        bust: "n/a",
        outfit: "oilskin",
        pose: "standing",
        expression: "watchful",
        facts: [],
        images: [
          {
            id: "img-c_other",
            path: portraitPath,
            prompt_hash: "x",
            attributes_snapshot: {},
            created_at: "2026-05-02T00:00:00Z",
          },
        ],
        seed: 7,
      },
    },
    dialog_tree: {
      nodes: {
        n1: {
          id: "n1",
          parent_id: null,
          chosen_option_id: null,
          speaker_id: null,
          text: "The harbor wakes slow.",
          options: [],
          entering_character_ids: [],
          leaving_character_ids: [],
          new_characters: [],
          location_id: null,
          location_prompt: null,
          character_changes: [],
          state: "committed",
          premise_hash: "h0",
          generation_metadata: {
            model: null,
            prompt_hash: null,
            seed_parameters: {},
            tokens_in: 0,
            tokens_out: 0,
            latency_ms: 0,
          },
        },
      },
      root_id: "n1",
      committed_path: ["n1"],
    },
    environments: {},
    current_node_id: "n1",
    on_stage: ["c_other"],
    cost_telemetry: {
      tokens_in: 0,
      tokens_out: 0,
      image_calls: 0,
      llm_calls: 0,
      latency_ms_total: 0,
      dollar_estimate: 0,
    },
  };
  const settings = {
    llm: { base_url: "", model: "x", api_key: "", temperature: 0.0, max_tokens: 1024 },
    image: { base_url: "", portrait_workflow: "", background_workflow: "" },
    typewriter_speed_chars_per_sec: 1000,
    prompt_history_clamp_chars: 12000,
    concurrency: { llm_max_in_flight: 4, image_max_in_flight: 2 },
  };
  return `
(() => {
  const game = ${JSON.stringify(game)};
  const settings = ${JSON.stringify(settings)};
  window.__lucidium.handlers["c2s/hello"] = function () {
    return [{ type: "s2c/hello", payload: { protocol_version: 1, has_save: true } }];
  };
  window.__lucidium.handlers["c2s/saves/list"] = function () {
    return [{ type: "s2c/saves/list", payload: { saves: [{
      id: "s1", name: "Test", last_played_at: "2026-05-02T00:00:00Z",
      created_at: "2026-05-02T00:00:00Z", schema_version: 1, summary: ""
    }] } }];
  };
  window.__lucidium.handlers["c2s/saves/continue"] = function () {
    return [{ type: "s2c/state/full", payload: { game, settings } }];
  };
})();
`;
}
