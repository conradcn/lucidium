/**
 * ConfirmStep — Go back button.
 *
 * Pins:
 *   * Clicking Go back dispatches ``c2s/new_game/go_back`` with an
 *     empty payload (the backend wipes its InterviewState + cancels
 *     every prefetch) AND invokes ``onCancel`` so the parent screen
 *     routes the player back to the main menu.
 *   * The button is disabled once Begin has been clicked, so a
 *     stray double-click can't rewind a turn that's already
 *     committing.
 *   * Pending debounced edits are flushed BEFORE the cancel fires,
 *     mirroring the Begin flow, so an in-flight edit can't race
 *     the cancel and re-populate ``session.interview`` after the
 *     reset.
 *   * Begin is the primary action (``confirm-actions__begin`` class
 *     for the highlighted styling). Go back is the secondary action
 *     (``confirm-actions__back`` — no accent border, subdued color)
 *     so the eye lands on Begin first.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, cleanup, fireEvent, act } from "@testing-library/react";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({ on: () => () => undefined })),
}));

import { send } from "../../src/app/client";
import { ConfirmStep } from "../../src/screens/NewGameInterview/ConfirmStep";

afterEach(() => cleanup());

describe("ConfirmStep — Go back", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
    vi.useRealTimers();
  });

  it("dispatches c2s/new_game/go_back AND invokes onCancel", () => {
    const cancelSpy = vi.fn();
    const { container } = render(
      <ConfirmStep
        snapshot={{ setting: "x", side_characters: {} }}
        onConfirmed={() => undefined}
        onCancel={cancelSpy}
      />,
    );
    const backBtn = container.querySelector(
      '[data-testid="confirm-go-back"]',
    ) as HTMLButtonElement;
    expect(backBtn).not.toBeNull();
    fireEvent.click(backBtn);

    expect(send).toHaveBeenCalledWith("c2s/new_game/go_back", {});
    expect(cancelSpy).toHaveBeenCalledTimes(1);
  });

  it("works without an onCancel callback (no-op fallback)", () => {
    const { container } = render(
      <ConfirmStep
        snapshot={{ setting: "x", side_characters: {} }}
        onConfirmed={() => undefined}
      />,
    );
    const backBtn = container.querySelector(
      '[data-testid="confirm-go-back"]',
    ) as HTMLButtonElement;
    // Must not throw even with onCancel undefined.
    fireEvent.click(backBtn);
    expect(send).toHaveBeenCalledWith("c2s/new_game/go_back", {});
  });

  it("disables once Begin has been clicked", () => {
    const { container } = render(
      <ConfirmStep
        snapshot={{ setting: "x", side_characters: {} }}
        onConfirmed={() => undefined}
        onCancel={() => undefined}
      />,
    );
    const beginBtn = container.querySelector(
      '[data-testid="confirm-begin"]',
    ) as HTMLButtonElement;
    const backBtn = container.querySelector(
      '[data-testid="confirm-go-back"]',
    ) as HTMLButtonElement;
    expect(backBtn.disabled).toBe(false);
    fireEvent.click(beginBtn);
    expect(backBtn.disabled).toBe(true);
  });

  it("Begin carries the highlighted-primary class; Go back carries the subdued class", () => {
    const { container } = render(
      <ConfirmStep
        snapshot={{ setting: "x", side_characters: {} }}
        onConfirmed={() => undefined}
        onCancel={() => undefined}
      />,
    );
    const beginBtn = container.querySelector(
      '[data-testid="confirm-begin"]',
    ) as HTMLButtonElement;
    const backBtn = container.querySelector(
      '[data-testid="confirm-go-back"]',
    ) as HTMLButtonElement;
    // Pin the class hooks so a future styling refactor that drops
    // the primary/secondary distinction fails this test.
    expect(beginBtn.classList.contains("confirm-actions__begin")).toBe(true);
    expect(backBtn.classList.contains("confirm-actions__back")).toBe(true);
    // Begin must NOT carry the secondary class and vice versa.
    expect(beginBtn.classList.contains("confirm-actions__back")).toBe(false);
    expect(backBtn.classList.contains("confirm-actions__begin")).toBe(false);
  });

  it("flushes pending debounced edits before sending go_back", () => {
    vi.useFakeTimers();
    const { container } = render(
      <ConfirmStep
        snapshot={{ setting: "stone harbor", side_characters: {} }}
        onConfirmed={() => undefined}
        onCancel={() => undefined}
      />,
    );
    const settingInput = container.querySelector(
      '[data-testid="review-edit-setting"]',
    ) as HTMLTextAreaElement;
    // Type a new value; the save is debounced (350ms) so without a
    // flush the dispatch order would be:
    //   1. go_back fires
    //   2. (350ms later) the stale edit fires
    // which means the Name step pops back with the OLD setting on
    // screen.
    fireEvent.change(settingInput, {
      target: { value: "storm-shadowed harbor" },
    });

    const backBtn = container.querySelector(
      '[data-testid="confirm-go-back"]',
    ) as HTMLButtonElement;
    act(() => {
      fireEvent.click(backBtn);
    });

    const calls = vi.mocked(send).mock.calls;
    // Edit must precede go_back so the backend writes the new value
    // before the step bounce lands on the renderer.
    const editIdx = calls.findIndex(
      ([type]) => type === "c2s/new_game/edit_review",
    );
    const backIdx = calls.findIndex(
      ([type]) => type === "c2s/new_game/go_back",
    );
    expect(editIdx).toBeGreaterThanOrEqual(0);
    expect(backIdx).toBeGreaterThanOrEqual(0);
    expect(editIdx).toBeLessThan(backIdx);
    const editCall = calls[editIdx];
    expect(editCall).toBeDefined();
    expect(editCall![1]).toEqual({
      field: "setting",
      value: "storm-shadowed harbor",
    });
    vi.useRealTimers();
  });
});
