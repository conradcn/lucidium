import { test, expect } from "@playwright/test";

import { APP_URL, installHappyPathMock } from "./helpers";

test.describe("Load Game screen", () => {
  test("shows the empty state and a Back button when no saves exist", async ({ page }) => {
    await installHappyPathMock(page);
    await page.goto(APP_URL);

    await page.getByRole("button", { name: "Load Game" }).click();
    await expect(page.getByRole("heading", { name: "Load game" })).toBeVisible();
    await expect(page.getByText("No saves yet.")).toBeVisible();

    await page.getByRole("button", { name: "Back" }).click();
    await expect(page.getByRole("heading", { name: "LUCIDIUM" })).toBeVisible();
  });

  test("renders saves with rename, delete, and load actions", async ({ page }) => {
    await installHappyPathMock(page);
    // Override the saves-list handler via an init script so it survives
    // the page.goto() that init scripts re-run on. (page.evaluate before
    // goto is too early — the handlers object only exists after a real
    // page loads.)
    await page.addInitScript({
      content: `
        (() => {
          window.__lucidium.handlers["c2s/saves/list"] = function () {
            return [{
              type: "s2c/saves/list",
              payload: {
                saves: [
                  { id: "save-A", name: "Long lost morning",
                    last_played_at: "2026-04-30T10:00:00Z",
                    created_at: "2026-04-29T10:00:00Z",
                    schema_version: 1,
                    summary: "Iris hesitates at the threshold." },
                  { id: "save-B", name: "Storm chaser",
                    last_played_at: "2026-04-28T10:00:00Z",
                    created_at: "2026-04-27T10:00:00Z",
                    schema_version: 1,
                    summary: "Hale watches the harbor light wink out." },
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
    await expect(page.getByText("Storm chaser")).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Load" }).first(),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Rename" }).first(),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Delete" }).first(),
    ).toBeVisible();
  });
});
