/**
 * Settings + Story panel autosave: every field change fires a
 * debounced patch through the existing ``c2s/settings/update``
 * (or ``c2s/edit/*``) message rather than waiting on a Save
 * button. Pin the contract for the global Settings screen and
 * the per-tab story panels so a refactor that re-introduces a
 * Save button trips the test.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, cleanup, fireEvent } from "@testing-library/react";

import { useLucidiumStore } from "../../src/state/store";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({ on: () => () => undefined })),
}));

import { send } from "../../src/app/client";
import { SettingsScreen } from "../../src/settings/SettingsScreen";

function baseSettings(): Record<string, unknown> {
  return {
    llm: { base_url: "x", model: "y", api_key: "", temperature: 0.8, max_tokens: 1024 },
    image: { backend: "embedded", embedded_models_dir: "" },
    music: { enabled: false },
    audio: { master_volume: 0.7, music_volume: 0.4 },
    typewriter_speed_chars_per_sec: 60,
    mature_content: false,
    learn_user_profile: true,
    user_profile: {},
  };
}

afterEach(() => cleanup());

describe("Settings autosave", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
  });

  it("has no Save button", () => {
    useLucidiumStore.setState({ settings: baseSettings() });
    const { container } = render(<SettingsScreen onBack={() => undefined} />);
    const labels = Array.from(container.querySelectorAll("button")).map(
      (b) => b.textContent?.trim() ?? "",
    );
    expect(labels).not.toContain("Save");
    expect(labels).not.toContain("Cancel");
  });

  it("offers a Done button that returns without saving (autosave handles persistence)", () => {
    const onBack = vi.fn();
    useLucidiumStore.setState({ settings: baseSettings() });
    const { container } = render(<SettingsScreen onBack={onBack} />);
    const done = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent?.trim() === "Done",
    ) as HTMLButtonElement;
    expect(done).toBeTruthy();
    fireEvent.click(done);
    expect(onBack).toHaveBeenCalled();
  });

  it("toggles Mature content checkbox → autosaves within debounce window", async () => {
    useLucidiumStore.setState({ settings: baseSettings() });
    const { container } = render(<SettingsScreen onBack={() => undefined} />);
    const checkbox = Array.from(
      container.querySelectorAll('input[type="checkbox"]'),
    ).find((el) => {
      const label = el.closest("label")?.textContent ?? "";
      return /mature content/i.test(label);
    }) as HTMLInputElement | undefined;
    expect(checkbox).toBeTruthy();
    fireEvent.click(checkbox!);
    // Autosave debounces ~350 ms.
    await new Promise((r) => setTimeout(r, 450));
    const call = vi.mocked(send).mock.calls.find(
      (c) =>
        c[0] === "c2s/settings/update"
        && (c[1] as { patch: { mature_content?: boolean } }).patch.mature_content === true,
    );
    expect(call).toBeTruthy();
  });

  it("flushes a pending patch immediately on unmount", async () => {
    useLucidiumStore.setState({ settings: baseSettings() });
    const { container, unmount } = render(
      <SettingsScreen onBack={() => undefined} />,
    );
    const styleInput = container.querySelector(
      'input[placeholder="conversational"]',
    ) as HTMLInputElement;
    fireEvent.change(styleInput, { target: { value: "pirate" } });
    // Unmount BEFORE the debounce fires — the cleanup hook
    // must flush the pending patch so the player's last edit
    // doesn't get dropped if they navigate away.
    unmount();
    const call = vi.mocked(send).mock.calls.find(
      (c) =>
        c[0] === "c2s/settings/update"
        && (c[1] as { patch: { narrator_style?: string } }).patch.narrator_style === "pirate",
    );
    expect(call).toBeTruthy();
  });
});
