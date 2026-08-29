/**
 * The Settings panel's "Inferred by the engine" section shows
 * ONLY tags whose summarizer-side hit count crosses
 * ``CERTAINTY_THRESHOLD``. Tags with a lower count stay tracked
 * internally — the storyteller's merged-profile prompt still
 * sees them — but they don't surface to the player until
 * reinforced.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { render, cleanup } from "@testing-library/react";

import { useLucidiumStore } from "../../src/state/store";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({ on: () => () => undefined })),
}));

import { SettingsScreen } from "../../src/settings/SettingsScreen";

function settingsWithInferred(opts: {
  likes: string[];
  scores: Record<string, number>;
}): Record<string, unknown> {
  return {
    llm: { base_url: "", model: "", api_key: "", temperature: 0.8, max_tokens: 1024 },
    image: { backend: "embedded", embedded_models_dir: "" },
    music: { enabled: false },
    audio: { master_volume: 0.7, music_volume: 0.4 },
    typewriter_speed_chars_per_sec: 60,
    mature_content: false,
    learn_user_profile: true,
    user_profile: {
      likes: [],
      dislikes: [],
      notes: [],
      summarizer_likes: opts.likes,
      summarizer_dislikes: [],
      summarizer_notes: [],
      summarizer_likes_scores: opts.scores,
      summarizer_dislikes_scores: {},
      summarizer_notes_scores: {},
      dismissed_likes: [],
      dismissed_dislikes: [],
      dismissed_notes: [],
    },
  };
}

afterEach(() => cleanup());

/** Read the values of the "Likes (inferred)" bucket's input
 *  rows. The tags render as ``<input value="...">`` for editing,
 *  not as text nodes, so ``textContent`` doesn't see them — we
 *  query inputs whose aria-label points at the inferred-likes
 *  bucket. */
function inferredLikesValues(container: HTMLElement): string[] {
  const inputs = Array.from(
    container.querySelectorAll('input[aria-label^="Likes (inferred) entry"]'),
  ) as HTMLInputElement[];
  return inputs.map((el) => el.value);
}

describe("Inferred-tags strength threshold (post-rework)", () => {
  it("hides tags whose strength is below the surface threshold (1.0)", () => {
    // The new scheme: tags accumulate strength from 0.1
    // (initial inference) by 0.1 per re-inference, surface
    // gate at 1.0. A tag at 0.4 is still speculative — the
    // engine tracks it but the player doesn't see it yet.
    useLucidiumStore.setState({
      settings: settingsWithInferred({
        likes: ["confident-thing", "fresh-guess"],
        scores: { "confident-thing": 1.0, "fresh-guess": 0.4 },
      }),
    });
    const { container } = render(<SettingsScreen onBack={() => undefined} />);
    const visible = inferredLikesValues(container);
    expect(visible).toContain("confident-thing");
    expect(visible).not.toContain("fresh-guess");
  });

  it("shows tags whose strength meets the surface threshold", () => {
    useLucidiumStore.setState({
      settings: settingsWithInferred({
        likes: ["just-surfaced", "very-strong"],
        scores: { "just-surfaced": 1.0, "very-strong": 1.8 },
      }),
    });
    const { container } = render(<SettingsScreen onBack={() => undefined} />);
    const visible = inferredLikesValues(container);
    expect(visible).toContain("just-surfaced");
    expect(visible).toContain("very-strong");
  });

  it("hides a tag at initial strength (0.1) — one inference is invisible", () => {
    // Pin the user-requested behaviour: a brand-new inference
    // (one observation, strength 0.1) does NOT surface. Tags
    // need ~9 reinforcements (0.1 + 9*0.1 = 1.0) to cross the
    // gate. This is what the user meant by "Learning feature
    // introduces contamination" — the old scheme surfaced
    // single-shot inferences.
    useLucidiumStore.setState({
      settings: settingsWithInferred({
        likes: ["one-shot-inference"],
        scores: { "one-shot-inference": 0.1 },
      }),
    });
    const { container } = render(<SettingsScreen onBack={() => undefined} />);
    expect(inferredLikesValues(container)).not.toContain("one-shot-inference");
  });

  it("treats tags missing from scores dict as above threshold (legacy save)", () => {
    // Pre-strength saves have populated lists but empty score
    // dicts. The default-to-threshold rule keeps those entries
    // visible — load doesn't silently hide everything.
    useLucidiumStore.setState({
      settings: settingsWithInferred({
        likes: ["heritage-tag", "another-old-tag"],
        scores: {},
      }),
    });
    const { container } = render(<SettingsScreen onBack={() => undefined} />);
    const visible = inferredLikesValues(container);
    expect(visible).toContain("heritage-tag");
    expect(visible).toContain("another-old-tag");
  });

  it("accepts integer strengths from old saves (pydantic int→float coerce)", () => {
    // Pydantic's int→float coercion means an old save
    // serialised with ``"x": 2`` (int) loads as ``"x": 2.0``
    // (float). Under the new threshold of 1.0 that tag stays
    // visible — matches old-system behaviour for
    // already-surfaced tags.
    useLucidiumStore.setState({
      settings: settingsWithInferred({
        likes: ["legacy-surfaced", "legacy-borderline"],
        scores: { "legacy-surfaced": 2, "legacy-borderline": 1 },
      }),
    });
    const { container } = render(<SettingsScreen onBack={() => undefined} />);
    const visible = inferredLikesValues(container);
    expect(visible).toContain("legacy-surfaced");
    expect(visible).toContain("legacy-borderline");
  });

  it("uses case-insensitive lookup for scores", () => {
    useLucidiumStore.setState({
      settings: settingsWithInferred({
        likes: ["Slow-Burn Investigation"],
        scores: { "slow-burn investigation": 1.5 },
      }),
    });
    const { container } = render(<SettingsScreen onBack={() => undefined} />);
    const visible = inferredLikesValues(container);
    expect(visible).toContain("Slow-Burn Investigation");
  });
});
