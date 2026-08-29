import { test, expect } from "@playwright/test";

import { APP_URL, installHappyPathMock } from "./helpers";

test.describe("Settings screen", () => {
  test("renders LLM, Image, and Presentation sections populated from backend", async ({
    page,
  }) => {
    await installHappyPathMock(page);
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "Options" }).click();

    await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "LLM backend (OpenAI-compatible)" }),
    ).toBeVisible();
    // "Image backend", not "Image backend (ComfyUI)" — the embedded
    // diffusers backend means ComfyUI is no longer the only option.
    await expect(
      page.getByRole("heading", { name: "Image backend" }),
    ).toBeVisible();
    await expect(page.getByRole("heading", { name: "Presentation" })).toBeVisible();

    // The Pydantic-default base URLs come back from the mocked
    // c2s/settings/get round-trip via the /settings patch.
    await expect(
      page.locator('input[type="text"]').filter({ hasText: "" }).first(),
    ).toHaveValue("https://openrouter.ai/api/v1");
    await expect(
      page.locator('input[value="http://127.0.0.1:8000"]'),
    ).toHaveCount(1);
  });

  test("an edit autosaves as c2s/settings/update, and Done returns to start", async ({
    page,
  }) => {
    await installHappyPathMock(page);
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "Options" }).click();

    // There is no longer a Save button: the screen autosaves each edit
    // on a debounce and exits via "Done". So the update has to be
    // provoked by an actual edit rather than by the exit click.
    await page
      .locator('input[value="http://127.0.0.1:8000"]')
      .fill("http://127.0.0.1:9999");

    await expect
      .poll(
        () =>
          page.evaluate(() =>
            window.__lucidium.sent.filter(
              (m) => m.type === "c2s/settings/update",
            ).length,
          ),
        { timeout: 5_000 },
      )
      .toBeGreaterThan(0);

    await page.getByRole("button", { name: "Done" }).click();
    await expect(page.getByRole("heading", { name: "LUCIDIUM" })).toBeVisible();
  });
});
