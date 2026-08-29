/**
 * The Settings panel exposes a free-text "Narrator style" field
 * with quick-pick suggestion pills. Saving sends the value
 * through ``c2s/settings/update`` so the storyteller's next
 * prompt picks it up.
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

function settingsWith(narratorStyle?: string): Record<string, unknown> {
  return {
    llm: { base_url: "", model: "", api_key: "", temperature: 0.8, max_tokens: 1024 },
    image: { backend: "embedded", embedded_models_dir: "" },
    music: { enabled: false },
    audio: { master_volume: 0.7, music_volume: 0.4 },
    typewriter_speed_chars_per_sec: 60,
    mature_content: false,
    learn_user_profile: true,
    user_profile: {},
    ...(narratorStyle !== undefined ? { narrator_style: narratorStyle } : {}),
  };
}

afterEach(() => cleanup());

describe("Narrator style setting", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
  });

  it("hydrates from settings.narrator_style", () => {
    useLucidiumStore.setState({ settings: settingsWith("hard-boiled noir") });
    const { container } = render(<SettingsScreen onBack={() => undefined} />);
    const input = container.querySelector(
      'input[placeholder="conversational"]',
    ) as HTMLInputElement;
    expect(input).not.toBeNull();
    expect(input.value).toBe("hard-boiled noir");
  });

  it("defaults to 'conversational' when settings doesn't set the field", () => {
    useLucidiumStore.setState({ settings: settingsWith(undefined) });
    const { container } = render(<SettingsScreen onBack={() => undefined} />);
    const input = container.querySelector(
      'input[placeholder="conversational"]',
    ) as HTMLInputElement;
    expect(input.value).toBe("conversational");
  });

  it("clicking a suggestion pill updates the input", () => {
    useLucidiumStore.setState({ settings: settingsWith() });
    const { container } = render(<SettingsScreen onBack={() => undefined} />);
    const pirate = Array.from(container.querySelectorAll("button"))
      .find((b) => b.textContent?.trim() === "pirate") as HTMLButtonElement;
    expect(pirate).toBeTruthy();
    fireEvent.click(pirate);
    const input = container.querySelector(
      'input[placeholder="conversational"]',
    ) as HTMLInputElement;
    expect(input.value).toBe("pirate");
  });

  it("typing in the narrator_style input autosaves the patch", async () => {
    useLucidiumStore.setState({ settings: settingsWith("conversational") });
    const { container } = render(<SettingsScreen onBack={() => undefined} />);
    const input = container.querySelector(
      'input[placeholder="conversational"]',
    ) as HTMLInputElement;
    fireEvent.change(input, { target: { value: "literary" } });
    // Autosave debounces 350 ms — wait it out plus a small
    // buffer so the queued setTimeout fires.
    await new Promise((r) => setTimeout(r, 450));
    const updateCall = vi.mocked(send).mock.calls.find(
      (c) => c[0] === "c2s/settings/update"
        && (c[1] as { patch: Record<string, unknown> }).patch.narrator_style === "literary",
    );
    expect(updateCall).toBeTruthy();
  });

  it("offers the canonical suggestion list", () => {
    useLucidiumStore.setState({ settings: settingsWith() });
    const { container } = render(<SettingsScreen onBack={() => undefined} />);
    const labels = Array.from(container.querySelectorAll("button"))
      .map((b) => b.textContent?.trim() ?? "");
    for (const expected of [
      "conversational",
      "literary",
      "informal",
      "pirate",
    ]) {
      expect(labels).toContain(expected);
    }
  });
});
