/**
 * Architectural test: every routable phase MUST render something
 * recognizable, both in the happy-path connected state and when the
 * WebSocket is dead. This file is the regression test for "all clicks
 * lead to a blank screen". A new screen that doesn't add itself to
 * SCREENS will not be exercised here, but every existing screen is.
 */

import { test, expect, type Page, type Locator } from "@playwright/test";

import { APP_URL, installHappyPathMock, installMockWs } from "./helpers";

interface ScreenSpec {
  /** Human-readable label used in the test name. */
  name: string;
  /** Click path from the home screen, in order. */
  navigate: (page: Page) => Promise<void>;
  /** ``data-testid`` on the screen's root element. */
  testid: string;
  /** Locator that must be visible when the screen is healthy. */
  assertHealthy: (page: Page) => Locator;
  /** Optional locator that must be visible when the WS is dead.
   *  Defaults to ``data-testid={testid}``. */
  assertOffline?: (page: Page) => Locator;
}

const SCREENS: ScreenSpec[] = [
  {
    name: "start",
    navigate: async () => undefined,
    testid: "start-screen",
    assertHealthy: (page) => page.getByRole("heading", { name: "LUCIDIUM" }),
  },
  {
    name: "interview",
    navigate: async (page) => {
      await page.getByRole("button", { name: "New Game" }).click();
    },
    testid: "interview-setting",
    assertHealthy: (page) => page.getByRole("heading", { name: "Setting" }),
    assertOffline: (page) => page.getByTestId("interview-loading"),
  },
  {
    name: "load-game",
    navigate: async (page) => {
      await page.getByRole("button", { name: "Load Game" }).click();
    },
    testid: "load-game-screen",
    assertHealthy: (page) => page.getByRole("heading", { name: "Load game" }),
  },
  {
    name: "settings",
    navigate: async (page) => {
      await page.getByRole("button", { name: "Options" }).click();
    },
    testid: "settings-screen",
    assertHealthy: (page) => page.getByRole("heading", { name: "Settings" }),
  },
  {
    name: "main",
    navigate: async (page) => {
      await page.getByRole("button", { name: "Continue" }).click();
    },
    testid: "main-view",
    assertHealthy: (page) => page.getByText("The harbor wakes slow."),
  },
];

test.describe("Every screen renders non-empty content (happy path)", () => {
  for (const screen of SCREENS) {
    test(`${screen.name} is non-empty when WS is connected`, async ({ page }) => {
      await installHappyPathMock(page, { hasSave: true });
      await page.goto(APP_URL);
      await expect(page.getByRole("heading", { name: "LUCIDIUM" })).toBeVisible();
      await screen.navigate(page);

      const root = page.getByTestId(screen.testid);
      await expect(root).toBeVisible();
      const text = (await root.innerText()).trim();
      expect(text.length, `${screen.name} root has no text`).toBeGreaterThan(0);
      await expect(screen.assertHealthy(page)).toBeVisible();
      await expect(page.getByTestId("connection-banner")).toHaveCount(0);
    });
  }
});

test.describe("Every screen still renders something with a dead WS", () => {
  for (const screen of SCREENS.filter((s) => s.name !== "main")) {
    test(`${screen.name} shows banner + recognizable content when WS never opens`, async ({
      page,
    }) => {
      // Install ONLY the bare WebSocket mock — no handlers configured,
      // so c2s/hello never gets a reply, the connection stays in
      // "open" with no s2c/hello, and screens that depend on backend
      // state must still render something.
      await installMockWs(page);
      await page.addInitScript({
        content: `
          (() => {
            // Make the mock socket never call onopen — simulates a
            // backend that isn't listening at all.
            var Original = window.WebSocket;
            function Stalled(url) {
              this.url = url;
              this.readyState = 0;
              this.onopen = null;
              this.onmessage = null;
              this.onerror = null;
              this.onclose = null;
            }
            Stalled.prototype.send = function () {};
            Stalled.prototype.close = function () { this.readyState = 3; };
            Stalled.prototype.addEventListener = function () {};
            Stalled.prototype.removeEventListener = function () {};
            window.WebSocket = Stalled;
          })();
        `,
      });
      await page.goto(APP_URL);
      await screen.navigate(page);

      // The connection banner MUST be visible — that's the contract for
      // "connection isn't open".
      await expect(page.getByTestId("connection-banner")).toBeVisible();

      // The screen's root element MUST still render with some
      // recognizable affordance — never a fully blank page.
      const fallback = screen.assertOffline ? screen.assertOffline(page) : page.getByTestId(screen.testid);
      await expect(fallback).toBeVisible();
    });
  }
});

test("all home-screen buttons land on a recognized phase testid", async ({ page }) => {
  // Broad-spectrum guard: it would have caught the "every button
  // leads to a blank screen" regression because each phase's root
  // wrapper carries a data-testid that this test asserts.
  //
  // The SPA does not use real browser history, so we reload to return
  // to the start screen between path probes.
  await installHappyPathMock(page, { hasSave: true });

  for (const [label, expectedTestId] of [
    ["New Game", "phase-interview"],
    ["Load Game", "phase-load"],
    ["Options", "phase-settings"],
    ["Continue", "phase-main"],
  ] as const) {
    await page.goto(APP_URL);
    await expect(page.getByTestId("phase-start")).toBeVisible();
    await page.getByRole("button", { name: label }).click();
    await expect(page.getByTestId(expectedTestId)).toBeVisible();
  }
});
