/**
 * Reproduces the user-reported bug: "Adding a character during new
 * game grays out the field, but doesn't add them anywhere."
 *
 * The Add button uses ``useOptimisticAction(sideCharacters.length)``
 * — the field disables on click and re-enables when the side-
 * characters list grows by one (the marker comparison releases the
 * guard). If the snapshot's side_characters never gains the new
 * entry, the field stays disabled forever AND the character is
 * absent from the visible list.
 *
 * Two flows tested here:
 *   1. Happy path: snapshot rev 1 has empty list, user clicks Add,
 *      snapshot rev 2 has the new entry → field re-enables, name
 *      renders, c2s/new_game/add_side_character was sent.
 *   2. Stuck path: backend never echoes the patch (snapshot stays
 *      empty) → field stays disabled (matches user's symptom). This
 *      one exists to lock in the symptom; the underlying fix is in
 *      the backend handler / patch path, not the component.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, cleanup, fireEvent } from "@testing-library/react";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({ on: () => () => undefined })),
}));

import { send } from "../../src/app/client";
import { ConfirmStep } from "../../src/screens/NewGameInterview/ConfirmStep";

afterEach(() => cleanup());

describe("ConfirmStep — Add side character round-trip", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
  });

  it("dispatches add_side_character with the typed description", () => {
    const { container } = render(
      <ConfirmStep
        snapshot={{ setting: "x", side_characters: {} }}
        onConfirmed={() => undefined}
      />,
    );
    const input = container.querySelector(
      '[data-testid="side-character-description"]',
    ) as HTMLInputElement;
    fireEvent.change(input, { target: { value: "the keeper" } });
    const addBtn = Array.from(container.querySelectorAll("button"))
      .find((b) => b.textContent?.trim() === "Add") as HTMLButtonElement;
    fireEvent.click(addBtn);

    expect(send).toHaveBeenCalledWith("c2s/new_game/add_side_character", {
      description: "the keeper",
    });
    // Field is grayed pending backend echo.
    expect(input.disabled).toBe(true);
  });

  it(
    "field re-enables AND name renders once snapshot.side_characters " +
      "gains the entry",
    () => {
      const onConfirmed = () => undefined;
      const { container, rerender } = render(
        <ConfirmStep
          snapshot={{ setting: "x", side_characters: {} }}
          onConfirmed={onConfirmed}
        />,
      );
      const input = container.querySelector(
        '[data-testid="side-character-description"]',
      ) as HTMLInputElement;
      fireEvent.change(input, { target: { value: "the keeper" } });
      const addBtn = Array.from(container.querySelectorAll("button"))
        .find((b) => b.textContent?.trim() === "Add") as HTMLButtonElement;
      fireEvent.click(addBtn);
      expect(input.disabled).toBe(true);

      // Backend echoes the patch — snapshot's side_characters now
      // contains the new entry. Re-render with the updated snapshot.
      rerender(
        <ConfirmStep
          snapshot={{
            setting: "x",
            side_characters: {
              sc1: { id: "sc1", name: "Hale Stone" },
            },
          }}
          onConfirmed={onConfirmed}
        />,
      );

      const inputAfter = container.querySelector(
        '[data-testid="side-character-description"]',
      ) as HTMLInputElement;
      expect(inputAfter.disabled).toBe(false);
      // Name renders in the visible side-characters list. The row
      // is editable now, so the name lives as the input's value
      // rather than as plain text in the chip.
      const nameInput = container.querySelector(
        '[data-testid="side-character-row-sc1"] input',
      ) as HTMLInputElement;
      expect(nameInput.value).toBe("Hale Stone");
    },
  );

  it(
    "field stays disabled AND name is absent when snapshot never echoes " +
      "(reproduces the reported stuck-state)",
    () => {
      const { container, rerender } = render(
        <ConfirmStep
          snapshot={{ setting: "x", side_characters: {} }}
          onConfirmed={() => undefined}
        />,
      );
      const input = container.querySelector(
        '[data-testid="side-character-description"]',
      ) as HTMLInputElement;
      fireEvent.change(input, { target: { value: "the keeper" } });
      const addBtn = Array.from(container.querySelectorAll("button"))
        .find((b) => b.textContent?.trim() === "Add") as HTMLButtonElement;
      fireEvent.click(addBtn);

      // Simulate the backend NOT echoing — re-render with the same
      // empty snapshot (mirrors the 200ms tick re-render in the
      // parent, which keeps reading the same snapshot).
      rerender(
        <ConfirmStep
          snapshot={{ setting: "x", side_characters: {} }}
          onConfirmed={() => undefined}
        />,
      );

      const inputAfter = container.querySelector(
        '[data-testid="side-character-description"]',
      ) as HTMLInputElement;
      expect(inputAfter.disabled).toBe(true);
      // No row gets rendered when the snapshot has no entries.
      expect(
        container.querySelector('[data-testid^="side-character-row-"]'),
      ).toBeNull();
    },
  );
});
