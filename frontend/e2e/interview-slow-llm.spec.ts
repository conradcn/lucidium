/**
 * Non-blocking interview contract under a slow LLM.
 *
 * The new game flow promises that every step transition appears
 * IMMEDIATELY, even when the LLM that produces the next step's
 * options is slow. The renderer should:
 *   1. Receive a state/patch that flips ``/interview/step`` and
 *      includes empty options for the next step.
 *   2. Display the next step's heading immediately, with the
 *      renderer's built-in DEFAULT_OPTIONS already usable and a
 *      spinner marking them as provisional, while the LLM is
 *      still in flight.
 *   3. Patch the options into the grid via a follow-up state/patch
 *      whenever the backend's prefetch finally lands.
 *
 * Step layout:
 *   * Setting (Step 1) — hard-coded options, inline.
 *   * Visual Style (Step 2) — hard-coded options, inline. NO LLM
 *     loading state ever appears here; if it does, the backend has
 *     regressed to LLM-fetching visual styles.
 *   * Genre (Step 3) — hard-coded options, inline.
 *   * Character Description (Step 4) — LLM-driven; SLOW path tested.
 *   * Name (Step 5) — LLM-driven; SLOW path tested.
 *
 * This spec injects a multi-second LLM delay into the LLM-driven
 * steps. The step transitions MUST appear in a fraction of that
 * delay (see IMMEDIATE_MS); the options for the LLM-driven steps
 * MUST appear later, before the test ends.
 */

import { test, expect } from "@playwright/test";

import { APP_URL, installMockWs } from "./helpers";

// The simulated LLM latency. Deliberately long: the wider the gap
// between this and IMMEDIATE_MS, the more decisively a passing run
// proves the step transition never sat on the LLM's critical path.
const SLOW_DELIVERY_MS = 10_000;

// The budget for a step transition. This is NOT a frame-rate target —
// it is the ceiling that separates "advanced optimistically" from
// "waited for the LLM". At a third of SLOW_DELIVERY_MS, any transition
// that actually blocked on the delivery would overshoot it by 3x.
//
// The generous absolute value is deliberate. The measured transition
// under `fullyParallel` load is ~300-500 ms, but the tail runs long
// when workers contend for CPU (2.1 s observed locally over 80
// samples), and a hosted CI runner is slower still. A tighter bound
// fails on scheduler jitter rather than on a real regression.
const IMMEDIATE_MS = 3_000;

test.describe("Interview: non-blocking transitions under a slow LLM", () => {
  // Two steps wait out the full SLOW_DELIVERY_MS, which puts this
  // spec past Playwright's 30 s default on its own.
  test.describe.configure({ timeout: 90_000 });


  test("each step appears immediately; options arrive via follow-up patch", async ({
    page,
  }) => {
    await installMockWs(page);
    await page.addInitScript({
      content: `
(() => {
  const SLOW_MS = ${SLOW_DELIVERY_MS};
  // The mock simulates the real backend's two-message protocol for
  // LLM-driven steps: a SYNCHRONOUS state/patch flipping the step
  // pointer (with empty options), followed by a deferred state/patch
  // carrying the options once the simulated LLM finishes.
  function transition(stepFrom, fieldFrom, valueFrom, stepTo, optionsField, slowOptions) {
    return [
      // Inline transition — fires immediately.
      { type: "s2c/state/patch", payload: { ops: [
        { op: "replace", path: "/interview/" + fieldFrom, value: valueFrom },
        { op: "replace", path: "/interview/step", value: stepTo },
        ...(optionsField ? [{ op: "replace", path: optionsField, value: [] }] : []),
      ] } },
      // Deferred options — arrives after SLOW_MS.
      ...(optionsField && slowOptions ? [{
        type: "s2c/state/patch",
        payload: { ops: [{ op: "replace", path: optionsField, value: slowOptions }] },
        __delayMs: SLOW_MS,
      }] : []),
    ];
  }

  window.__lucidium.handlers["c2s/hello"] = function () {
    return [{ type: "s2c/hello", payload: { protocol_version: 1, has_save: false } }];
  };
  window.__lucidium.handlers["c2s/saves/list"] = function () {
    return [{ type: "s2c/saves/list", payload: { saves: [] } }];
  };
  window.__lucidium.handlers["c2s/new_game/start"] = function () {
    return [{
      type: "s2c/state/patch",
      payload: { ops: [{
        op: "replace", path: "/interview",
        value: { step: "setting", setting_options: ["stone harbor"] },
      }] },
    }];
  };
  window.__lucidium.handlers["c2s/new_game/answer"] = function (payload) {
    if (payload.step === "setting") {
      // Setting → Visual Style. Visual styles are HARD-CODED — they
      // ride along inline. There is no slow path here. (If the
      // backend ever regresses and starts LLM-fetching visual
      // styles, this mock will keep working but the real app will
      // show a loading spinner — caught by the backend assertion in
      // test_interview_parallelism.)
      return [{
        type: "s2c/state/patch",
        payload: { ops: [
          { op: "replace", path: "/interview/setting", value: payload.answer },
          { op: "replace", path: "/interview/step", value: "visual_style" },
          { op: "replace", path: "/interview/visual_style_options", value: ["ink wash", "anime", "noir"] },
        ] },
      }];
    }
    if (payload.step === "visual_style") {
      // Visual Style → Genre. Genre is hard-coded; inline.
      return [{
        type: "s2c/state/patch",
        payload: { ops: [
          { op: "replace", path: "/interview/visual_style", value: payload.answer },
          { op: "replace", path: "/interview/step", value: "genre" },
          { op: "replace", path: "/interview/genre_options", value: ["Mystery", "Romance", "Horror"] },
        ] },
      }];
    }
    if (payload.step === "genre") {
      // Genre → Character. SLOW (LLM-driven).
      return transition(
        "genre", "genre", payload.answer,
        "character_description", "/interview/character_description_options",
        ["tidewatch cartographer", "disgraced harbormaster"]
      );
    }
    if (payload.step === "character_description") {
      // Character → Name. SLOW (LLM-driven).
      return transition(
        "character_description", "character_description", payload.answer,
        "name", "/interview/name_options",
        ["Sable Wren", "Corin Ashgrove"]
      );
    }
    if (payload.step === "name") {
      return [{
        type: "s2c/state/patch",
        payload: { ops: [
          { op: "replace", path: "/interview/character_name", value: payload.answer },
          { op: "replace", path: "/interview/step", value: "confirm" },
        ] },
      }];
    }
    return [];
  };
})();
`,
    });

    await page.goto(APP_URL);
    await page.getByRole("button", { name: "New Game" }).click();
    await expect(page.getByTestId("interview-setting")).toBeVisible();

    // 1. Setting → Visual Style. Visual style options are HARD-CODED
    //    so they MUST arrive inline — no loading state. If a loading
    //    wrap shows up here, the backend has regressed to LLM-fetching
    //    visual styles.
    let before = Date.now();
    await page.getByRole("button", { name: "stone harbor" }).click();
    await expect(page.getByTestId("interview-visual_style")).toBeVisible({
      timeout: IMMEDIATE_MS,
    });
    let elapsed = Date.now() - before;
    expect(elapsed, `setting→visual_style transition was ${elapsed} ms`).toBeLessThanOrEqual(
      IMMEDIATE_MS,
    );
    // No loading affordance — the options grid is up immediately.
    await expect(page.getByTestId("interview-loading")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "ink wash" })).toBeVisible();

    // 2. Visual Style → Genre. Hard-coded — instant on every count.
    before = Date.now();
    await page.getByRole("button", { name: "ink wash" }).click();
    await expect(page.getByTestId("interview-genre")).toBeVisible({
      timeout: IMMEDIATE_MS,
    });
    elapsed = Date.now() - before;
    expect(elapsed, `visual_style→genre transition was ${elapsed} ms`).toBeLessThanOrEqual(
      IMMEDIATE_MS,
    );
    await expect(page.getByTestId("interview-loading")).toHaveCount(0);

    // 3. Genre → Character Description. SLOW (LLM-driven).
    before = Date.now();
    await page.getByRole("button", { name: "Mystery", exact: true }).click();
    // The step itself must be up immediately. It is NOT a loading splash
    // any more: the renderer shows its built-in DEFAULT_OPTIONS right
    // away, flagged with a spinner, so the player can pick and move on
    // instead of waiting on the LLM.
    await expect(page.getByTestId("interview-character_description")).toBeVisible({
      timeout: IMMEDIATE_MS,
    });
    elapsed = Date.now() - before;
    expect(elapsed, `genre→character transition was ${elapsed} ms`).toBeLessThanOrEqual(
      IMMEDIATE_MS,
    );
    // Defaults are on screen and usable while the LLM is still in flight.
    await expect(page.getByTestId("interview-options-loading")).toBeVisible();
    await expect(
      page.getByRole("button", { name: "cynical detective" }),
    ).toBeVisible();
    // ...and the real options replace them when they land, clearing the
    // spinner. These values are deliberately absent from DEFAULT_OPTIONS
    // so this can only pass if the follow-up patch actually applied.
    await expect(
      page.getByRole("button", { name: "tidewatch cartographer" }),
    ).toBeVisible({ timeout: SLOW_DELIVERY_MS + 1_000 });
    await expect(page.getByTestId("interview-options-loading")).toHaveCount(0);

    // 4. Character → Name. SLOW.
    before = Date.now();
    await page.getByRole("button", { name: "tidewatch cartographer" }).click();
    await expect(page.getByTestId("interview-name")).toBeVisible({
      timeout: IMMEDIATE_MS,
    });
    elapsed = Date.now() - before;
    expect(elapsed, `character→name transition was ${elapsed} ms`).toBeLessThanOrEqual(
      IMMEDIATE_MS,
    );
    await expect(page.getByTestId("interview-options-loading")).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Mira Quill" }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Sable Wren" }),
    ).toBeVisible({ timeout: SLOW_DELIVERY_MS + 1_000 });
    await expect(page.getByTestId("interview-options-loading")).toHaveCount(0);

    // 5. Name → Review. Inline.
    before = Date.now();
    await page.getByRole("button", { name: "Sable Wren" }).click();
    await expect(page.getByRole("heading", { name: "Review" })).toBeVisible({
      timeout: IMMEDIATE_MS,
    });
    elapsed = Date.now() - before;
    expect(elapsed, `name→review transition was ${elapsed} ms`).toBeLessThanOrEqual(
      IMMEDIATE_MS,
    );
  });
});
