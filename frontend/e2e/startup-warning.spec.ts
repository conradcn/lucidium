/**
 * Coverage for the startup safety modal and its automation escape
 * hatch.
 *
 * Every OTHER browser spec navigates to ``APP_URL``, which carries
 * ``skipWarning=1`` so the modal doesn't sit on top of the app eating
 * their clicks. That means this file is the only place the modal's real
 * behaviour is still exercised — it deliberately navigates to "/" so it
 * sees what a player sees.
 *
 * Both halves matter:
 *   * without the flag the warning MUST appear and MUST block, because
 *     it is a safety notice (see screens/StartupWarning.tsx);
 *   * with the flag it must be gone, because ~50 other specs depend on
 *     that being true.
 */
import { test, expect } from "@playwright/test";

import { APP_URL, installHappyPathMock } from "./helpers";

test.describe("Startup warning", () => {
  test("appears on a normal launch and blocks the app until acknowledged", async ({
    page,
  }) => {
    await installHappyPathMock(page);
    await page.goto("/");

    const warning = page.getByTestId("startup-warning");
    await expect(warning).toBeVisible();

    // It is a real modal, not just a banner. The start-screen buttons
    // are still PAINTED behind the overlay — so assert the property
    // that actually matters, that the overlay swallows the click,
    // rather than that the button is invisible.
    await expect(
      page.getByRole("button", { name: "New Game" }).click({ timeout: 1_000 }),
    ).rejects.toThrow(/intercepts pointer events/);

    await page.getByRole("button", { name: "I understand" }).click();
    await expect(warning).toHaveCount(0);

    // ...and the app underneath is reachable once acknowledged.
    await expect(page.getByRole("button", { name: "New Game" })).toBeVisible();
  });

  test("stays dismissed for the rest of the session", async ({ page }) => {
    await installHappyPathMock(page);
    await page.goto("/");

    await page.getByRole("button", { name: "I understand" }).click();
    await expect(page.getByTestId("startup-warning")).toHaveCount(0);

    // Moving between screens must not resurrect it.
    await page.getByRole("button", { name: "Options" }).click();
    await expect(page.getByTestId("startup-warning")).toHaveCount(0);
  });

  test("is suppressed by the skipWarning flag", async ({ page }) => {
    await installHappyPathMock(page);
    await page.goto(APP_URL);

    await expect(page.getByTestId("startup-warning")).toHaveCount(0);
    // The app is immediately interactive — this is the property every
    // other spec relies on.
    await expect(page.getByRole("button", { name: "New Game" })).toBeVisible();
  });

  test("re-appears on a fresh launch without the flag", async ({ page }) => {
    await installHappyPathMock(page);

    await page.goto(APP_URL);
    await expect(page.getByTestId("startup-warning")).toHaveCount(0);

    // Dismissal is per-session and the flag does not persist: a plain
    // reload shows the warning again. This is the "no opt-out" property
    // the modal's docstring insists on.
    await page.goto("/");
    await expect(page.getByTestId("startup-warning")).toBeVisible();
  });
});
