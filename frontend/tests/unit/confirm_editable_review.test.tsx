/**
 * The Review step's previously read-only fields are now inline-
 * editable until the player clicks Begin. Each edit fires a
 * debounced ``c2s/new_game/edit_review`` patch; clicking Begin
 * flushes any pending edits before sending ``c2s/new_game/confirm``
 * so the world_init prefetch sees the latest values.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render } from "@testing-library/react";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({ on: () => () => undefined })),
}));

import { send } from "../../src/app/client";
import { ConfirmStep } from "../../src/screens/NewGameInterview/ConfirmStep";

afterEach(() => cleanup());

const SNAPSHOT = {
  setting: "harbor",
  genre: "mystery",
  visual_style: "ink",
  character_description: "archivist",
  character_name: "Iris",
  side_characters: {},
};

describe("ConfirmStep — editable review fields", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(send).mockClear();
  });

  it("renders each answered step as an editable input pre-populated with the snapshot value", () => {
    const { getByTestId } = render(
      <ConfirmStep snapshot={SNAPSHOT} onConfirmed={vi.fn()} />,
    );
    expect((getByTestId("review-edit-setting") as HTMLTextAreaElement).value)
      .toBe("harbor");
    expect((getByTestId("review-edit-genre") as HTMLInputElement).value)
      .toBe("mystery");
    expect((getByTestId("review-edit-visual_style") as HTMLTextAreaElement).value)
      .toBe("ink");
    expect((getByTestId("review-edit-character_description") as HTMLTextAreaElement).value)
      .toBe("archivist");
    expect((getByTestId("review-edit-character_name") as HTMLInputElement).value)
      .toBe("Iris");
  });

  it("debounce-sends c2s/new_game/edit_review on input change", () => {
    const { getByTestId } = render(
      <ConfirmStep snapshot={SNAPSHOT} onConfirmed={vi.fn()} />,
    );
    fireEvent.change(getByTestId("review-edit-character_name"), {
      target: { value: "Marigold" },
    });
    // Before the debounce fires, no send yet.
    expect(vi.mocked(send).mock.calls.find(
      (c) => c[0] === "c2s/new_game/edit_review",
    )).toBeUndefined();
    // Advance past the 350ms debounce.
    vi.advanceTimersByTime(400);
    const call = vi.mocked(send).mock.calls.find(
      (c) => c[0] === "c2s/new_game/edit_review",
    );
    expect(call).toBeTruthy();
    expect(call?.[1]).toEqual({
      field: "character_name",
      value: "Marigold",
    });
  });

  it("coalesces rapid edits to a single send (last-write-wins)", () => {
    const { getByTestId } = render(
      <ConfirmStep snapshot={SNAPSHOT} onConfirmed={vi.fn()} />,
    );
    fireEvent.change(getByTestId("review-edit-setting"), {
      target: { value: "h" },
    });
    fireEvent.change(getByTestId("review-edit-setting"), {
      target: { value: "harvest" },
    });
    fireEvent.change(getByTestId("review-edit-setting"), {
      target: { value: "harvest moon valley" },
    });
    vi.advanceTimersByTime(400);
    const editCalls = vi.mocked(send).mock.calls.filter(
      (c) => c[0] === "c2s/new_game/edit_review",
    );
    expect(editCalls.length).toBe(1);
    expect(editCalls[0]?.[1]).toEqual({
      field: "setting",
      value: "harvest moon valley",
    });
  });

  it("flushes pending edits IMMEDIATELY when Begin is clicked", () => {
    const { getByTestId, container } = render(
      <ConfirmStep snapshot={SNAPSHOT} onConfirmed={vi.fn()} />,
    );
    fireEvent.change(getByTestId("review-edit-character_name"), {
      target: { value: "Hale" },
    });
    // Click Begin BEFORE the 350ms debounce fires.
    const beginBtn = Array.from(container.querySelectorAll("button"))
      .find((b) => /^Begin$/i.test(b.textContent ?? ""));
    expect(beginBtn).toBeTruthy();
    (beginBtn as HTMLButtonElement).click();
    // The debounce was flushed: edit_review fires synchronously.
    const editCall = vi.mocked(send).mock.calls.find(
      (c) => c[0] === "c2s/new_game/edit_review",
    );
    expect(editCall?.[1]).toEqual({
      field: "character_name",
      value: "Hale",
    });
    // And confirm fires AFTER the edit (same tick, but ordered).
    const calls = vi.mocked(send).mock.calls.map((c) => c[0]);
    const editIdx = calls.indexOf("c2s/new_game/edit_review");
    const confirmIdx = calls.indexOf("c2s/new_game/confirm");
    expect(editIdx).toBeGreaterThanOrEqual(0);
    expect(confirmIdx).toBeGreaterThan(editIdx);
  });

  it("locks fields once Begin is clicked", () => {
    const { getByTestId, container } = render(
      <ConfirmStep snapshot={SNAPSHOT} onConfirmed={vi.fn()} />,
    );
    const beginBtn = Array.from(container.querySelectorAll("button"))
      .find((b) => /^Begin$/i.test(b.textContent ?? "")) as HTMLButtonElement;
    // fireEvent wraps in act() so the state update from
    // useOptimisticAction's run() is reflected in the next
    // disabled-prop check.
    fireEvent.click(beginBtn);
    // Every editable field is now disabled.
    for (const key of [
      "setting", "genre", "visual_style",
      "character_description", "character_name",
    ]) {
      const el = getByTestId(`review-edit-${key}`) as
        | HTMLInputElement
        | HTMLTextAreaElement;
      expect(el.disabled).toBe(true);
    }
  });
});
