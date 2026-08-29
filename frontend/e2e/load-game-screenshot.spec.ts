import { test } from "@playwright/test";

import { APP_URL, installHappyPathMock } from "./helpers";

/**
 * One-off screenshot capture for the Load Game screen with varied
 * save shapes (long names, long summaries, no-space stress test).
 * Not a regression test — the assertion is only that the screen
 * rendered. Used to eyeball layout fixes; output lands in
 * ``test-results/load-game-screen.png``.
 */
test("Load Game screenshot — varied save shapes", async ({ page }) => {
  await installHappyPathMock(page, { hasSave: true });
  await page.addInitScript({
    content: `
      (() => {
        const settings = {
          llm: { base_url: "https://openrouter.ai/api/v1", model: "qwen/qwen-2.5-72b-instruct", api_key: "x", temperature: 0.8, max_tokens: 1024 },
          image: { backend: "comfyui", base_url: "http://127.0.0.1:8000", portrait_workflow: "p", background_workflow: "b", embedded_models_dir: "", embedded_model_name: "" },
          typewriter_speed_chars_per_sec: 60,
          prompt_history_clamp_chars: 12000,
          concurrency: { llm_max_in_flight: 4, image_max_in_flight: 2 },
          mature_content: false,
          user_profile: {
            likes: [], dislikes: [], notes: [],
            summarizer_likes: [], summarizer_dislikes: [], summarizer_notes: [],
          },
          first_time_setup_complete: true,
        };
        window.__lucidium.handlers["c2s/settings/get"] = function () {
          return [{
            type: "s2c/state/patch",
            payload: { ops: [{ op: "replace", path: "/settings", value: settings }] },
          }];
        };
        window.__lucidium.handlers["c2s/saves/list"] = function () {
          return [{
            type: "s2c/saves/list",
            payload: {
              saves: [
                {
                  id: "s1",
                  name: "Embers of the Veil",
                  last_played_at: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
                  created_at: "2026-04-01T10:00:00Z",
                  schema_version: 1,
                  summary: "Iris hesitates at the threshold of the archive, the brass key cold in her palm.",
                },
                {
                  id: "s2",
                  name: "A Long Saved Game Title That Goes On And On",
                  last_played_at: "2026-04-30T10:00:00Z",
                  created_at: "2026-04-29T10:00:00Z",
                  schema_version: 1,
                  summary: "A particularly long save summary intended to test whether the layout truncates, wraps, or otherwise handles overflow gracefully without pushing the action buttons off the visible row. The third sentence in this summary verifies the three-line clamp is taking effect.",
                },
                {
                  id: "s3",
                  name: "Storm chaser",
                  last_played_at: "2026-04-15T10:00:00Z",
                  created_at: "2026-04-14T10:00:00Z",
                  schema_version: 1,
                  summary: "Hale watches the harbor light wink out.",
                },
                {
                  id: "s4",
                  name: "antidisestablishmentarianismantidisestablishmentarianism",
                  last_played_at: "2026-03-15T10:00:00Z",
                  created_at: "2026-03-14T10:00:00Z",
                  schema_version: 1,
                  summary: "Stress-test for unbreakable long words in the save name column — overflow-wrap should split mid-token instead of pushing the row past the meta column edge.",
                },
                {
                  id: "s5",
                  name: "Bittersweet hotel",
                  last_played_at: "2026-03-01T10:00:00Z",
                  created_at: "2026-02-28T10:00:00Z",
                  schema_version: 1,
                  summary: "The flapper drifts to the window without lifting her gaze from the smoke.",
                },
                {
                  id: "s6",
                  name: "Cathedral of falling stars",
                  last_played_at: "2026-02-15T10:00:00Z",
                  created_at: "2026-02-14T10:00:00Z",
                  schema_version: 1,
                  summary: "The witch-hunter checks the powder in his second pistol.",
                },
                {
                  id: "s7",
                  name: "Convenience-store winter",
                  last_played_at: "2026-02-01T10:00:00Z",
                  created_at: "2026-01-31T10:00:00Z",
                  schema_version: 1,
                  summary: "The clerk rotates the rice balls. Outside, the rain finds its rhythm.",
                },
              ],
            },
          }];
        };
      })();
    `,
  });
  // ``APP_URL`` carries ``skipWarning=1``, so the per-launch warning
  // modal is suppressed and the start screen is clickable straight away.
  await page.goto(APP_URL);
  await page.getByRole("button", { name: "Load Game" }).click();
  await page.waitForSelector("[data-testid='load-game-screen']");
  // Wait briefly for save rows to land via the mocked s2c/saves/list reply.
  await page.waitForFunction(
    () => document.querySelectorAll(".save-list li").length >= 7,
    null,
    { timeout: 4_000 },
  );
  await page.screenshot({
    path: "test-results/load-game-screen.png",
    fullPage: true,
  });
});
