import { test, expect } from "@playwright/test";

import { APP_URL, installHappyPathMock } from "./helpers";

test.describe("Start screen", () => {
  test("renders New Game / Load Game / Options / Exit when no save exists", async ({ page }) => {
    await installHappyPathMock(page, { hasSave: false });
    await page.goto(APP_URL);

    await expect(page.getByRole("heading", { name: "LUCIDIUM" })).toBeVisible();
    await expect(page.getByRole("button", { name: "New Game" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Load Game" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Options" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Exit" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Continue" })).toHaveCount(0);
  });

  test("shows Continue when a save exists", async ({ page }) => {
    await installHappyPathMock(page, { hasSave: true });
    await page.goto(APP_URL);

    await expect(page.getByRole("button", { name: "Continue" })).toBeVisible();
  });

  test("Continue loads a save and routes into the main view", async ({ page }) => {
    await installHappyPathMock(page, { hasSave: true });
    await page.goto(APP_URL);

    await page.getByRole("button", { name: "Continue" }).click();
    await expect(page.getByText("The harbor wakes slow.")).toBeVisible();
    await expect(page.getByRole("button", { name: "Walk to the archive." })).toBeVisible();
  });
});
