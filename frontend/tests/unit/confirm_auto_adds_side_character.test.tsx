/**
 * On the Review step the player can type a side-character
 * description and click Add to enqueue them. If they type a
 * description but click Begin without first clicking Add, the
 * UI now auto-adds the typed description before sending
 * confirm — same expectation as a form that submits its
 * pending textbox. Without this, a player who typed "the
 * keeper of the lighthouse" and forgot to press Add would
 * start their game without that NPC.
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

describe("ConfirmStep auto-add", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
  });

  it("auto-adds typed-but-unconfirmed side character before confirming", () => {
    const onConfirmed = vi.fn();
    const { container } = render(
      <ConfirmStep
        snapshot={{
          setting: "harbor",
          genre: "mystery",
          visual_style: "ink",
          character_description: "archivist",
          character_name: "Iris",
          side_characters: {},
        }}
        onConfirmed={onConfirmed}
      />,
    );
    const input = container.querySelector(
      '[data-testid="side-character-description"]',
    ) as HTMLInputElement;
    expect(input).not.toBeNull();
    fireEvent.change(input, {
      target: { value: "the keeper of the lighthouse" },
    });
    const beginBtn = Array.from(container.querySelectorAll("button"))
      .find((b) => /^Begin$/i.test(b.textContent ?? ""));
    expect(beginBtn).toBeTruthy();
    (beginBtn as HTMLButtonElement).click();

    // Both messages must have fired, in order.
    const calls = vi.mocked(send).mock.calls.map((c) => c[0]);
    const addIdx = calls.indexOf("c2s/new_game/add_side_character");
    const confirmIdx = calls.indexOf("c2s/new_game/confirm");
    expect(addIdx).toBeGreaterThanOrEqual(0);
    expect(confirmIdx).toBeGreaterThan(addIdx);
    expect(send).toHaveBeenCalledWith("c2s/new_game/add_side_character", {
      description: "the keeper of the lighthouse",
    });
  });

  it("Begin with no pending text just confirms (no spurious add)", () => {
    const onConfirmed = vi.fn();
    const { container } = render(
      <ConfirmStep
        snapshot={{ setting: "x", genre: "y", visual_style: "z" }}
        onConfirmed={onConfirmed}
      />,
    );
    const beginBtn = Array.from(container.querySelectorAll("button"))
      .find((b) => /^Begin$/i.test(b.textContent ?? ""));
    (beginBtn as HTMLButtonElement).click();
    const calls = vi.mocked(send).mock.calls.map((c) => c[0]);
    expect(calls).toContain("c2s/new_game/confirm");
    expect(calls).not.toContain("c2s/new_game/add_side_character");
  });

  it("whitespace-only typed text doesn't trigger an add", () => {
    const onConfirmed = vi.fn();
    const { container } = render(
      <ConfirmStep
        snapshot={{ setting: "x" }}
        onConfirmed={onConfirmed}
      />,
    );
    const input = container.querySelector(
      '[data-testid="side-character-description"]',
    ) as HTMLInputElement;
    fireEvent.change(input, { target: { value: "   " } });
    const beginBtn = Array.from(container.querySelectorAll("button"))
      .find((b) => /^Begin$/i.test(b.textContent ?? ""));
    (beginBtn as HTMLButtonElement).click();
    const calls = vi.mocked(send).mock.calls.map((c) => c[0]);
    expect(calls).not.toContain("c2s/new_game/add_side_character");
    expect(calls).toContain("c2s/new_game/confirm");
  });
});
