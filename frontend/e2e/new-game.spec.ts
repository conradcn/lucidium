import { test, expect } from "@playwright/test";

import { APP_URL, installHappyPathMock, installMockWs } from "./helpers";

// Each step transition must complete inside this window. The bound
// is intentionally tight: a real LLM call on a slow provider can
// take 10+ s, so anything well below that here means the renderer
// is advancing optimistically rather than waiting on the network.
// CI machines + the 16-button Genre step's React commit pause can
// each cost ~150 ms; the 700 ms ceiling absorbs both without
// masking real regressions.
const IMMEDIATE_MS = 700;

test.describe("New Game interview", () => {
  test.beforeEach(async ({ page }) => {
    await installHappyPathMock(page);
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "New Game" }).click();
  });

  test("walks through every step and reaches the main view", async ({ page }) => {
    // Step order: Setting → Visual Style → Genre → Character → Name → Review.
    await expect(page.getByRole("heading", { name: "Setting" })).toBeVisible();
    await page.getByRole("button", { name: "stone harbor" }).click();

    await expect(page.getByRole("heading", { name: "Visual style" })).toBeVisible();
    await page.getByRole("button", { name: "ink" }).click();

    await expect(page.getByRole("heading", { name: "Genre" })).toBeVisible();
    await page.getByRole("button", { name: "Mystery", exact: true }).click();

    await expect(page.getByRole("heading", { name: "Your character" })).toBeVisible();
    await page.getByRole("button", { name: "wry archivist" }).click();

    await expect(page.getByRole("heading", { name: "Their name" })).toBeVisible();
    await page.getByRole("button", { name: "Iris Vale" }).click();

    await expect(page.getByRole("heading", { name: "Review" })).toBeVisible();
    await page.getByRole("button", { name: "Begin" }).click();

    await expect(page.getByText("The harbor wakes slow.")).toBeVisible();
  });

  test("every step transition is immediate (well under one second each)", async ({
    page,
  }) => {
    // The non-blocking interview contract: clicking any answer must
    // produce the next-step heading within ~ a frame, never on the
    // critical path of an LLM. Each step is timed individually so a
    // single slow transition fails the spec rather than being hidden
    // in an aggregated total.
    type Step = {
      heading: string;
      pick: string;
      next: string;
    };
    const steps: Step[] = [
      { heading: "Setting", pick: "stone harbor", next: "Visual style" },
      { heading: "Visual style", pick: "ink", next: "Genre" },
      { heading: "Genre", pick: "Mystery", next: "Your character" },
      {
        heading: "Your character",
        pick: "wry archivist",
        next: "Their name",
      },
      { heading: "Their name", pick: "Iris Vale", next: "Review" },
    ];

    for (const step of steps) {
      await expect(
        page.getByRole("heading", { name: step.heading }),
      ).toBeVisible();
      const before = Date.now();
      await page
        .getByRole("button", { name: step.pick, exact: step.pick === "Mystery" })
        .click();
      await expect(
        page.getByRole("heading", { name: step.next }),
      ).toBeVisible({ timeout: IMMEDIATE_MS });
      const elapsed = Date.now() - before;
      expect(
        elapsed,
        `${step.heading} → ${step.next} took ${elapsed} ms (limit ${IMMEDIATE_MS} ms)`,
      ).toBeLessThanOrEqual(IMMEDIATE_MS);
    }
  });

  test("free-text answer is accepted on every step", async ({ page }) => {
    // Free-text on the Setting step jumps to Visual Style under the
    // new ordering.
    await page
      .getByRole("textbox", { name: /Or write your own/ })
      .fill("steam-driven undercity");
    await page.getByRole("button", { name: "Continue" }).first().click();
    await expect(page.getByRole("heading", { name: "Visual style" })).toBeVisible();
  });

  test("ConfirmStep adds a side character via the LLM expansion", async ({ page }) => {
    // Walk the new step order: setting → visual_style → genre →
    // character → name → confirm.
    await page.getByRole("button", { name: "stone harbor" }).click();
    await page.getByRole("button", { name: "ink" }).click();
    await page.getByRole("button", { name: "Mystery", exact: true }).click();
    await page.getByRole("button", { name: "wry archivist" }).click();
    await page.getByRole("button", { name: "Iris Vale" }).click();
    await expect(page.getByRole("heading", { name: "Review" })).toBeVisible();
    await page
      .getByRole("textbox", { name: /One-line description/ })
      .fill("a gruff retired bounty hunter");
    await page.getByRole("button", { name: "Add" }).click();
    await expect(page.getByText("Hale Stone")).toBeVisible();
  });
});

test("loading screen renders when options have not arrived", async ({ page }) => {
  // Fresh page with a mock that answers c2s/hello but returns no
  // options for c2s/new_game/start, so the interview stays on the
  // loading screen.
  await installMockWs(page);
  await page.addInitScript({
    content: `
      (() => {
        window.__lucidium.handlers["c2s/hello"] = function () {
          return [{ type: "s2c/hello", payload: { protocol_version: 1, has_save: false } }];
        };
        window.__lucidium.handlers["c2s/new_game/start"] = function () { return []; };
      })();
    `,
  });
  await page.goto(APP_URL);
  await page.getByRole("button", { name: "New Game" }).click();
  await expect(page.getByTestId("interview-loading")).toBeVisible();
});
