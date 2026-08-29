/**
 * Both the first-time-setup wizard and the global Settings
 * screen must steer the user toward a supported checkpoint.
 *
 *   * Primary recommendation (Settings): Z-Image-Turbo —
 *     pre-loaded via CPU offload so a single 24 GiB card can
 *     render both character (832×1216) and background
 *     (1536×1024) without OOM.
 *   * Wizard: a Civitai search URL filtered to the three base
 *     models the embedded pipeline supports (SDXL / Pony /
 *     Z-Image) — broader than a single-model link, but still
 *     constrained so the user lands on something the loader
 *     can parse.
 *
 * Pin the wording so a future settings refactor doesn't
 * accidentally drop either recommendation — without the
 * Z-Image / Civitai link, players grab the alphabetically-
 * first checkpoint they have (which may be a non-SDXL
 * Qwen-Rapid model that the SDXL loader can't parse).
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { render, cleanup } from "@testing-library/react";

import { useLucidiumStore } from "../../src/state/store";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({ on: () => () => undefined })),
}));

import { SettingsScreen } from "../../src/settings/SettingsScreen";

afterEach(() => cleanup());

describe("SDXL guidance + civitai pointer", () => {
  it("first-time-setup source carries SDXL guidance + civitai pointer", async () => {
    // The wizard is multi-step state-machine driven by store
    // settings round-trips; reliably advancing it in vitest
    // would require mocking the WS round-trip. Instead, verify
    // the guidance lives in the FirstTimeSetup source — a
    // refactor that drops the lines is what we actually want
    // to catch. The Settings-screen test below covers the live
    // render path against a populated store.
    const fs = await import("node:fs/promises");
    const path = await import("node:path");
    const file = path.join(
      __dirname, "..", "..", "src", "screens", "FirstTimeSetup.tsx",
    );
    const source = await fs.readFile(file, "utf-8");
    expect(source).toMatch(/SDXL/i);
    expect(source).toMatch(/civitai\.com/i);
    // The wizard now points at a base-model-filtered Civitai search
    // (SDXL + Pony + Z-Image) instead of any one specific model.
    // Pin the filter params so a refactor that drops the Pony or
    // Z-Image filters narrows the search back to SDXL-only without
    // anyone noticing in review.
    expect(source).toMatch(/baseModels=SDXL/i);
    expect(source).toMatch(/baseModels=Pony/i);
    expect(source).toMatch(/baseModels=Z-Image/i);
    // The one-click hardware-recommended download: the copy, the
    // highlighted model name, the button, and the WS round-trips that
    // feed them. Pin these so a refactor can't quietly drop the
    // auto-download path back to manual-only.
    expect(source).toMatch(/dramatically change Lucidium/i);
    expect(source).toMatch(/download the recommend base model/i);
    expect(source).toMatch(/data-testid="download-recommended-model"/);
    expect(source).toMatch(/data-testid="recommended-model"/);
    expect(source).toMatch(/c2s\/embedded\/recommend_model/);
    expect(source).toMatch(/c2s\/embedded\/download_model/);
  });

  it("settings screen recommends Z-Image-Turbo first, SDXL as fallback", () => {
    useLucidiumStore.setState({
      settings: {
        llm: { base_url: "", model: "", api_key: "", temperature: 0.8, max_tokens: 1024 },
        image: { backend: "embedded", embedded_models_dir: "" },
        music: { enabled: false },
        audio: { master_volume: 0.7, music_volume: 0.4 },
        user_profile: {},
      },
    });
    const { container } = render(
      <SettingsScreen onBack={() => undefined} />,
    );

    // Z-Image-Turbo must surface as the primary recommendation with
    // a direct download link to the HF repo Lucidium loads from.
    const zImageLink = container.querySelector(
      'a[href*="Tongyi-MAI/Z-Image-Turbo"]',
    ) as HTMLAnchorElement | null;
    expect(
      zImageLink,
      "Settings screen must link to Z-Image-Turbo on HuggingFace",
    ).not.toBeNull();

    // SDXL fallback guidance + civitai pointer still present.
    expect(container.textContent ?? "").toMatch(/SDXL/i);
    const civitaiLink = container.querySelector(
      'a[href*="civitai.com"]',
    ) as HTMLAnchorElement | null;
    expect(civitaiLink).not.toBeNull();
  });
});
