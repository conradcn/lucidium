import { test } from "@playwright/test";

import { APP_URL, installHappyPathMock } from "./helpers";

/**
 * One-off screenshot capture for the first-time setup wizard at
 * the image-backend step. Output lands in
 * ``test-results/first-time-setup-screen.png``. Not a regression
 * test — the assertion is only that the screen rendered.
 */
test.use({ viewport: { width: 1600, height: 900 } });

test("First-time setup screenshot — image step", async ({ page }) => {
  await installHappyPathMock(page);
  // Force the wizard to fire by sending a settings reply with
  // ``first_time_setup_complete: false`` (the default mock fills
  // it in as truthy via missing field, which the renderer reads
  // as ``!truthy → wizard``).
  await page.addInitScript({
    content: `
      (() => {
        const settings = {
          llm: { base_url: "https://openrouter.ai/api/v1", model: "x", api_key: "", temperature: 0.8, max_tokens: 1024 },
          image: { backend: "embedded", base_url: "http://127.0.0.1:8000", portrait_workflow: "p", background_workflow: "b", embedded_models_dir: "", embedded_model_name: "" },
          typewriter_speed_chars_per_sec: 60,
          prompt_history_clamp_chars: 12000,
          concurrency: { llm_max_in_flight: 4, image_max_in_flight: 2 },
          mature_content: false,
          user_profile: {
            likes: [], dislikes: [], notes: [],
            summarizer_likes: [], summarizer_dislikes: [], summarizer_notes: [],
          },
          first_time_setup_complete: false,
        };
        window.__lucidium.handlers["c2s/settings/get"] = function () {
          return [{
            type: "s2c/state/patch",
            payload: { ops: [{ op: "replace", path: "/settings", value: settings }] },
          }];
        };
        window.__lucidium.handlers["c2s/embedded/list_models"] = function () {
          return [{
            type: "s2c/embedded/models",
            payload: { models_dir: "C:\\\\Users\\\\me\\\\AppData\\\\Roaming\\\\Lucidium\\\\models\\\\image", models: [] },
          }];
        };
      })();
    `,
  });
  await page.goto(APP_URL);
  await page.getByRole("button", { name: "I understand" }).click();
  // Settings arrive via state/patch on a microtask cycle; wait
  // for the wizard's phase wrapper directly so the routing
  // effect has time to land before we start clicking.
  await page.waitForSelector("[data-testid='phase-wizard']", { timeout: 10_000 });
  // Walk through to the image step — that's the one most affected
  // by the width change.
  await page.getByRole("button", { name: "Begin setup" }).click();
  await page.getByRole("button", { name: /Skip for now|^Next$/ }).click();
  await page.waitForSelector(".setup-choice-grid");
  await page.screenshot({
    path: "test-results/first-time-setup-screen.png",
    fullPage: true,
  });
});
