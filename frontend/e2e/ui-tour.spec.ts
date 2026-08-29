/**
 * UI tour: drive Chromium through every screen, take a screenshot of
 * each, and write them to ``test-results/ui-tour/`` so a reviewer can
 * audit the visual state of the renderer all in one place.
 *
 * Not part of the regular CI suite — run on demand with
 *
 *   npx playwright test e2e/ui-tour.spec.ts
 *
 * The screenshots that land under ``test-results/ui-tour/`` are
 * deliberately full-page captures; screen-relative layout issues are
 * easier to spot at the actual rendered viewport than in the
 * browser-DevTools tree.
 */

import path from "node:path";
import { mkdirSync } from "node:fs";

import { test, expect, type Page } from "@playwright/test";

import { APP_URL, installHappyPathMock, installMockWs } from "./helpers";

const SHOTS_DIR = path.resolve(__dirname, "..", "test-results", "ui-tour");
mkdirSync(SHOTS_DIR, { recursive: true });

async function shot(page: Page, name: string): Promise<void> {
  // Wait for animations and any in-flight typewriter to settle so the
  // capture isn't of a partial state.
  await page.waitForTimeout(300);
  await page.screenshot({
    path: path.join(SHOTS_DIR, `${name}.png`),
    fullPage: true,
  });
}

test.describe.configure({ mode: "serial" });

test.describe("UI tour — every screen", () => {
  test("01 start screen, no save", async ({ page }) => {
    await installHappyPathMock(page, { hasSave: false });
    await page.goto(APP_URL);
    await expect(page.getByTestId("start-screen")).toBeVisible();
    await shot(page, "01-start-no-save");
  });

  test("02 start screen, with save", async ({ page }) => {
    await installHappyPathMock(page, { hasSave: true });
    await page.goto(APP_URL);
    await expect(page.getByRole("button", { name: "Continue" })).toBeVisible();
    await shot(page, "02-start-with-save");
  });

  test("03 interview — setting step", async ({ page }) => {
    await installHappyPathMock(page);
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "New Game" }).click();
    await expect(page.getByTestId("interview-setting")).toBeVisible();
    await shot(page, "03-interview-setting");
  });

  test("03b interview — setting step with dream guide on stage", async ({ page }) => {
    await installHappyPathMock(page);
    // Use a tiny SVG data-URL as the placeholder image so the test
    // doesn't depend on the bundled PNGs being on disk; the goal here
    // is to confirm the dream-guide layout (centered, top half).
    const tinySvgDataUrl =
      "data:image/svg+xml;utf8," +
      encodeURIComponent(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 600">' +
          '<rect width="220" height="600" fill="#3a2a4a"/>' +
          '<circle cx="110" cy="180" r="60" fill="#d6b16d"/>' +
          '<rect x="60" y="240" width="100" height="320" rx="20" fill="#7a5e8a"/>' +
          "</svg>",
      );
    await page.addInitScript({
      content: `
        (() => {
          const guide = ${JSON.stringify(tinySvgDataUrl)};
          const room = ${JSON.stringify(tinySvgDataUrl)};
          const start = window.__lucidium.handlers["c2s/new_game/start"];
          window.__lucidium.handlers["c2s/new_game/start"] = function () {
            const replies = start ? start.apply(null, arguments) : [];
            replies.push({
              type: "s2c/state/patch",
              payload: {
                ops: [
                  { op: "replace", path: "/interview/dream_guide_image_path", value: guide },
                  { op: "replace", path: "/interview/white_room_image_path", value: room },
                ],
              },
            });
            return replies;
          };
        })();
      `,
    });
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "New Game" }).click();
    await expect(page.getByTestId("interview-setting")).toBeVisible();
    // The dream guide is now rendered through the carousel layer
    // (testid ``start-guide-stack``) shared with the start screen,
    // not the legacy InterviewStage.
    await expect(page.getByTestId("start-guide-stack")).toBeVisible();
    await shot(page, "03b-interview-setting-with-guide");
  });

  test("04 interview — genre step", async ({ page }) => {
    await installHappyPathMock(page);
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "New Game" }).click();
    await expect(page.getByTestId("interview-setting")).toBeVisible();
    // Setting → Visual Style → Genre under the new ordering.
    await page.getByRole("button", { name: "stone harbor" }).click();
    await expect(page.getByTestId("interview-visual_style")).toBeVisible();
    await page.getByRole("button", { name: "ink" }).click();
    await expect(page.getByTestId("interview-genre")).toBeVisible();
    await shot(page, "04-interview-genre");
  });

  test("05 interview — confirm step", async ({ page }) => {
    await installHappyPathMock(page);
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "New Game" }).click();
    // Step order: setting → visual_style → genre → character → name.
    await page.getByRole("button", { name: "stone harbor" }).click();
    await page.getByRole("button", { name: "ink" }).click();
    await page.getByRole("button", { name: "Mystery", exact: true }).click();
    await page.getByRole("button", { name: "wry archivist" }).click();
    await page.getByRole("button", { name: "Iris Vale" }).click();
    await expect(page.getByTestId("interview-confirm")).toBeVisible();
    await shot(page, "05-interview-confirm");
  });

  test("06 interview — loading state", async ({ page }) => {
    await installMockWs(page);
    await page.addInitScript({
      content: `
        (() => {
          window.__lucidium.handlers["c2s/hello"] = function () {
            return [{ type: "s2c/hello", payload: { protocol_version: 1, has_save: false } }];
          };
          window.__lucidium.handlers["c2s/saves/list"] = function () {
            return [{ type: "s2c/saves/list", payload: { saves: [] } }];
          };
          window.__lucidium.handlers["c2s/new_game/start"] = function () { return []; };
        })();
      `,
    });
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "New Game" }).click();
    await expect(page.getByTestId("interview-loading")).toBeVisible();
    await shot(page, "06-interview-loading");
  });

  test("07 main view, with save", async ({ page }) => {
    await installHappyPathMock(page, { hasSave: true });
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "Continue" }).click();
    await expect(page.getByTestId("main-view")).toBeVisible();
    await shot(page, "07-main-view");
  });

  test("07b main view — Hale speaking (speaker tag visible)", async ({ page }) => {
    await installHappyPathMock(page, { hasSave: true });
    // Override the saves/continue handler to deliver a node whose
    // speaker_id is c2 (Hale) so the speaker tag renders.
    await page.addInitScript({
      content: `
        (() => {
          const orig = window.__lucidium.handlers["c2s/saves/continue"];
          window.__lucidium.handlers["c2s/saves/continue"] = function () {
            const replies = orig ? orig.apply(null, arguments) : [];
            for (const r of replies) {
              if (r.type === "s2c/state/full" && r.payload && r.payload.game) {
                const tree = r.payload.game.dialog_tree;
                tree.nodes.n1.speaker_id = "c2";
                tree.nodes.n1.text = "The lighthouse went dark before dawn.";
              }
            }
            return replies;
          };
        })();
      `,
    });
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "Continue" }).click();
    await expect(page.getByTestId("main-view")).toBeVisible();
    await expect(page.getByTestId("speaker-tag")).toBeVisible();
    await shot(page, "07b-main-view-with-speaker");
  });

  test("08 main view — story panel open", async ({ page }) => {
    await installHappyPathMock(page, { hasSave: true });
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "Continue" }).click();
    await expect(page.getByTestId("main-view")).toBeVisible();
    await page.getByRole("button", { name: "Story" }).click();
    await expect(page.locator(".story-panel")).toBeVisible();
    await shot(page, "08-main-story-panel");
  });

  test("08b story panel — each tab", async ({ page }) => {
    await installHappyPathMock(page, { hasSave: true });
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "Continue" }).click();
    await expect(page.getByTestId("main-view")).toBeVisible();
    await page.getByRole("button", { name: "Story" }).click();
    await expect(page.locator(".story-panel")).toBeVisible();

    for (const tab of ["History", "World", "Environments", "Characters", "Tree", "Options"]) {
      await page
        .locator(".story-panel .tabs button", { hasText: tab })
        .click();
      const slug = tab.toLowerCase();
      await shot(page, `08b-story-tab-${slug}`);
    }
  });

  test("08c story panel — scroll preserved across Settings round-trip", async ({ page }) => {
    // Inject a tall body in the History tab via a CSS-only approach:
    // we add a stretch element to the story panel body so scrolling
    // is meaningful in the tour-mock state.
    await installHappyPathMock(page, { hasSave: true });
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "Continue" }).click();
    await expect(page.getByTestId("main-view")).toBeVisible();
    await page.getByRole("button", { name: "Story" }).click();
    await expect(page.locator(".story-panel")).toBeVisible();

    // Inject a tall stretch into the body so we have something to scroll.
    await page.evaluate(() => {
      const body = document.querySelector(
        '[data-testid="story-panel-body"]',
      );
      if (!body) return;
      const stretch = document.createElement("div");
      stretch.id = "_stretch";
      stretch.style.height = "2000px";
      stretch.style.background =
        "linear-gradient(transparent, rgba(214,177,109,0.06), transparent)";
      body.appendChild(stretch);
    });
    await page.evaluate(() => {
      const body = document.querySelector(
        '[data-testid="story-panel-body"]',
      ) as HTMLElement | null;
      if (body) body.scrollTop = 800;
    });
    await page.waitForTimeout(200);
    await shot(page, "08c-story-scrolled");

    // Round-trip through the in-game Settings screen — clicking the
    // Options tab inside the story panel opens settings.
    await page
      .locator(".story-panel .tabs button", { hasText: "Options" })
      .click();
    await page.getByRole("button", { name: "Open Settings" }).click();
    await expect(page.getByTestId("settings-screen")).toBeVisible();
    await shot(page, "08c-settings-from-ingame");

    // Cancel back to MainView, reopen Story. The persisted active
    // tab is "options" (we just clicked it), so we explicitly switch
    // back to History — that's the tab whose scroll we set to 800.
    await page.getByRole("button", { name: "Cancel" }).click();
    await expect(page.getByTestId("main-view")).toBeVisible();
    await page.getByRole("button", { name: "Story" }).click();
    await expect(page.locator(".story-panel")).toBeVisible();
    await page
      .locator(".story-panel .tabs button", { hasText: "History" })
      .click();
    // Re-stretch the History body so the saved scrollTop=800 is
    // representable on the new mount (the previous stretch was
    // destroyed by the unmount).
    await page.evaluate(() => {
      const body = document.querySelector(
        '[data-testid="story-panel-body"]',
      );
      if (!body) return;
      const stretch = document.createElement("div");
      stretch.id = "_stretch";
      stretch.style.height = "2000px";
      body.appendChild(stretch);
    });
    // Trigger StoryPanel's restore effect by toggling the tab once.
    await page
      .locator(".story-panel .tabs button", { hasText: "World" })
      .click();
    await page
      .locator(".story-panel .tabs button", { hasText: "History" })
      .click();
    await page.evaluate(() => {
      const body = document.querySelector(
        '[data-testid="story-panel-body"]',
      );
      if (!body) return;
      const stretch = document.createElement("div");
      stretch.id = "_stretch";
      stretch.style.height = "2000px";
      body.appendChild(stretch);
    });
    await page.waitForTimeout(300);
    const restoredScroll = await page.evaluate(() => {
      const body = document.querySelector(
        '[data-testid="story-panel-body"]',
      ) as HTMLElement | null;
      return body?.scrollTop ?? -1;
    });
    expect(restoredScroll).toBeGreaterThan(600);
    await shot(page, "08c-story-after-roundtrip");
  });

  test("09 settings screen", async ({ page }) => {
    await installHappyPathMock(page);
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "Options" }).click();
    await expect(page.getByTestId("settings-screen")).toBeVisible();
    await shot(page, "09-settings");
  });

  test("10 load game — empty", async ({ page }) => {
    await installHappyPathMock(page);
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "Load Game" }).click();
    await expect(page.getByText("No saves yet.")).toBeVisible();
    await shot(page, "10-load-empty");
  });

  test("11 load game — with saves", async ({ page }) => {
    await installHappyPathMock(page);
    await page.addInitScript({
      content: `
        (() => {
          window.__lucidium.handlers["c2s/saves/list"] = function () {
            return [{
              type: "s2c/saves/list",
              payload: {
                saves: [
                  { id: "save-A", name: "Long lost morning", last_played_at: "2026-04-30T10:00:00Z", created_at: "2026-04-29T10:00:00Z", schema_version: 1, summary: "Iris hesitates at the threshold." },
                  { id: "save-B", name: "Storm chaser", last_played_at: "2026-04-28T10:00:00Z", created_at: "2026-04-27T10:00:00Z", schema_version: 1, summary: "Hale watches the harbor light wink out." },
                ],
              },
            }];
          };
        })();
      `,
    });
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "Load Game" }).click();
    await expect(page.getByText("Long lost morning")).toBeVisible();
    await shot(page, "11-load-with-saves");
  });

  test("12 connection banner — disconnected", async ({ page }) => {
    await installMockWs(page);
    // Leave the default c2s/hello handler — the connection opens but
    // no save info → the banner won't show. Force a disconnected
    // state by overriding WebSocket to never open.
    await page.addInitScript({
      content: `
        (() => {
          function Stalled(url) { this.url = url; this.readyState = 0; this.onopen = null; this.onmessage = null; this.onclose = null; this.onerror = null; }
          Stalled.prototype.send = function(){};
          Stalled.prototype.close = function(){ this.readyState = 3; };
          Stalled.prototype.addEventListener = function(){};
          Stalled.prototype.removeEventListener = function(){};
          window.WebSocket = Stalled;
        })();
      `,
    });
    await page.goto(APP_URL);
    await page.waitForTimeout(5500);  // let the "stuck" message kick in
    await expect(page.getByTestId("connection-banner")).toBeVisible();
    await shot(page, "12-connection-banner");
  });
});
