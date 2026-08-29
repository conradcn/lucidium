import { test, expect } from "@playwright/test";

import { APP_URL, installHappyPathMock } from "./helpers";

/**
 * Smoke test: every Start-Screen button leads to a screen with visible
 * content. This is the test that would have caught "all buttons lead
 * to a blank screen".
 */
test.describe("All start-screen buttons reach a non-empty screen", () => {
  test.beforeEach(async ({ page }) => {
    await installHappyPathMock(page, { hasSave: true });
    await page.goto(APP_URL);
  });

  test("Continue", async ({ page }) => {
    await page.getByRole("button", { name: "Continue" }).click();
    await expect(page.getByText("The harbor wakes slow.")).toBeVisible();
  });

  test("New Game", async ({ page }) => {
    await page.getByRole("button", { name: "New Game" }).click();
    // Either the loading screen or the SettingStep heading is visible —
    // never an empty page.
    await expect(
      page.locator("[data-testid=interview-loading], [data-testid=interview-setting]"),
    ).toBeVisible();
  });

  test("Load Game", async ({ page }) => {
    await page.getByRole("button", { name: "Load Game" }).click();
    await expect(page.getByRole("heading", { name: "Load game" })).toBeVisible();
  });

  test("Options", async ({ page }) => {
    await page.getByRole("button", { name: "Options" }).click();
    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
  });
});
