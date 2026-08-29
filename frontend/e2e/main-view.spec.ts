import { test, expect } from "@playwright/test";

import { APP_URL, installHappyPathMock } from "./helpers";

test.describe("Main view", () => {
  test.beforeEach(async ({ page }) => {
    await installHappyPathMock(page, { hasSave: true });
    await page.goto(APP_URL);
    await page.getByRole("button", { name: "Continue" }).click();
    await expect(page.getByText("The harbor wakes slow.")).toBeVisible();
  });

  test("renders the on-stage non-player character name tag", async ({ page }) => {
    // Per FR-034a: the player character (Iris) must NOT be rendered
    // as a stage actor; the non-player on_stage character (Hale) does.
    await expect(page.getByText("Hale", { exact: true })).toBeVisible();
    await expect(page.locator(".main-view .stage").getByText("Iris", { exact: true })).toHaveCount(0);
  });

  test("clicking an option dispatches c2s/play/advance and renders new text", async ({
    page,
  }) => {
    await page.getByRole("button", { name: "Walk to the archive." }).click();
    await expect(
      page.getByText("She turns the brass key in the archive door."),
    ).toBeVisible();
    const sent = await page.evaluate(() => window.__lucidium.sent.map((m) => m.type));
    expect(sent).toContain("c2s/play/advance");
  });

  test("free-text submission dispatches c2s/play/free_text", async ({ page }) => {
    await page
      .getByRole("textbox", { name: /Or do something else/ })
      .fill("Iris kneels by the door.");
    await page.getByRole("button", { name: "Submit" }).click();
    const sent = await page.evaluate(() =>
      window.__lucidium.sent.find((m) => m.type === "c2s/play/free_text"),
    );
    expect(sent).toBeDefined();
    expect((sent as { payload: { text: string } }).payload.text).toBe(
      "Iris kneels by the door.",
    );
  });

  test("Story button toggles the side panel and shows every tab", async ({ page }) => {
    await page.getByRole("button", { name: "Story" }).click();
    await expect(page.getByRole("heading", { name: "Story" })).toBeVisible();
    // Tab labels are the UI's wording, not the internal tab keys:
    // ``environments`` renders as "Scenes" and ``characters`` as "Cast".
    // "Music" is intentionally absent — StoryPanel hides that tab unless
    // ``settings.music.enabled``, which the happy-path mock leaves unset.
    for (const tab of ["History", "World", "Scenes", "Cast", "Tree", "Options"]) {
      await expect(page.getByRole("button", { name: tab })).toBeVisible();
    }
  });

  test.describe("Story panel tabs", () => {
    test.beforeEach(async ({ page }) => {
      await page.getByRole("button", { name: "Story" }).click();
    });

    test("History tab renders committed nodes as editable text", async ({ page }) => {
      await page.getByRole("button", { name: "History" }).click();
      await expect(page.getByRole("textbox").first()).toBeVisible();
    });

    test("World tab autosaves an edit as c2s/edit/world", async ({ page }) => {
      await page.getByRole("button", { name: "World" }).click();
      const gameNameField = page.locator("textarea").first();
      await gameNameField.fill("Embers Reborn");

      // There is no Save button: WorldInfoTab commits each field on a
      // 400 ms debounce after the last keystroke. Poll rather than
      // sleeping a fixed interval so the assertion isn't timing-fragile.
      await expect
        .poll(
          () =>
            page.evaluate(
              () =>
                window.__lucidium.sent.filter(
                  (m) => m.type === "c2s/edit/world",
                ).length,
            ),
          { timeout: 5_000 },
        )
        .toBeGreaterThan(0);
    });

    test("Characters tab lists the player character", async ({ page }) => {
      await page.getByRole("button", { name: "Cast" }).click();
      await expect(page.getByRole("heading", { name: /Iris/ })).toBeVisible();
    });

    test("Tree tab shows the committed root node", async ({ page }) => {
      await page.getByRole("button", { name: "Tree" }).click();
      // ``.first()``: the tree renders the node label in both the row
      // and its detail pane, so the bare text matches twice and strict
      // mode rejects the ambiguous locator.
      await expect(page.getByText(/n1 · committed/).first()).toBeVisible();
    });

    test("Options tab links to Settings", async ({ page }) => {
      await page.getByRole("button", { name: "Options" }).click();
      await expect(page.getByRole("button", { name: "Open Settings" })).toBeVisible();
    });

    test("Environments tab handles the empty case", async ({ page }) => {
      await page.getByRole("button", { name: "Scenes" }).click();
      // No environments in our mock game; the tab should still render
      // without throwing.
      await expect(page.getByRole("heading", { name: "Story" })).toBeVisible();
    });
  });
});
