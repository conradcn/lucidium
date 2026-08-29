/**
 * Rapid-click regression: when the player clicks Continue (or
 * anywhere on the stage) faster than the LLM is producing the next
 * beat, EVERY line that DOES eventually get committed must reach
 * the DOM in full at some point. The renderer must never silently
 * skip a line — that drops player-visible plot.
 *
 * Setup:
 *   - Mock ``send`` so c2s/play/advance is observed and a deferred
 *     "LLM" updates the store with the next beat after a delay.
 *   - Drive a sequence of clicks at human speed (faster than the
 *     LLM delay) and observe that the typewriter eventually lands
 *     on every beat's full text.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, waitFor } from "@testing-library/react";

import { useLucidiumStore, type Game } from "../../src/state/store";
import { gameFixture, settingsFixture, type DeepPartial } from "../fixtures";
import type { DialogNode } from "../../src/shared/generated/Game.schema";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({ on: () => () => undefined })),
}));

import { send } from "../../src/app/client";
import { MainView } from "../../src/screens/MainView";

interface Beat {
  id: string;
  text: string;
}

function makeNode(beat: Beat, parentId: string | null, hasChild: boolean): DeepPartial<DialogNode> {
  return {
    id: beat.id,
    parent_id: parentId,
    chosen_option_id: null,
    speaker_id: null,
    text: beat.text,
    options: [],
    entering_character_ids: [],
    leaving_character_ids: [],
    new_characters: [],
    location_id: null,
    location_prompt: null,
    location_lighting: "",
    character_changes: [],
    state: hasChild ? "committed" : "committed",
    premise_hash: "h",
    generation_metadata: {
      model: null,
      prompt_hash: null,
      seed_parameters: {},
      tokens_in: 0,
      tokens_out: 0,
      latency_ms: 0,
    },
  };
}

function makeGame(beats: Beat[], currentIndex: number): Game {
  const nodes: Record<string, ReturnType<typeof makeNode>> = {};
  for (let i = 0; i <= currentIndex; i++) {
    const beat = beats[i]!;
    const parent = i === 0 ? null : beats[i - 1]!.id;
    const hasChild = i < currentIndex;
    nodes[beat.id] = makeNode(beat, parent, hasChild);
  }
  return gameFixture({
    id: "g1",
    schema_version: 1,
    current_node_id: beats[currentIndex]!.id,
    world: {
      game_name: "Test",
      setting: "harbor",
      genre: "mystery",
      visual_style: "ink",
      active_plot_threads: [],
      dropped_plot_threads: [],
      overall_plot_direction: "",
      summarizer_assessment: "",
      prompt_history_clamp_chars: 12000,
      player_intent: {
        pace_preference: "same",
        tone_preference: "unspecified",
        direction_signal: "none",
        weighted_evidence: [],
      },
    },
    characters: {},
    dialog_tree: {
      nodes,
      root_id: beats[0]!.id,
      committed_path: beats.slice(0, currentIndex + 1).map((b) => b.id),
    },
    environments: {},
    on_stage: [],
    cost_telemetry: {
      tokens_in: 0,
      tokens_out: 0,
      image_calls: 0,
      llm_calls: 0,
      latency_ms_total: 0,
      dollar_estimate: 0,
    },
  });
}

afterEach(() => {
  cleanup();
});

describe("rapid Continue clicks present every line", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
  });

  it("five rapid clicks (clicks ahead of mocked LLM delay) — every beat lands fully", async () => {
    const beats: Beat[] = [
      { id: "n1", text: "first beat lands clearly." },
      { id: "n2", text: "second beat opens a door." },
      { id: "n3", text: "third beat, a stranger turns." },
      { id: "n4", text: "fourth beat, the lights dim." },
      { id: "n5", text: "fifth beat, silence settles." },
    ];

    // Start at beat 0. Each c2s/play/advance triggers a delayed
    // store update that "lands" the next beat — simulating the
    // backend LLM round trip. The mocked delay is short but
    // non-zero so several clicks queue ahead of it.
    let currentIdx = 0;
    useLucidiumStore.setState({
      game: makeGame(beats, currentIdx),
      settings: settingsFixture({ typewriter_speed_chars_per_sec: 5000 }),
      hasSave: true,
      status: "open",
    });

    const sentTimes: number[] = [];
    vi.mocked(send).mockImplementation((msg: string) => {
      if (msg !== "c2s/play/advance") return;
      sentTimes.push(Date.now());
      if (currentIdx >= beats.length - 1) return;
      currentIdx += 1;
      const next = currentIdx;
      // Simulate a 50 ms LLM/backend delay before the next beat
      // appears in the store. The renderer keeps clicking through
      // this period; the test asserts every landed beat appears.
      setTimeout(() => {
        useLucidiumStore.setState({ game: makeGame(beats, next) });
      }, 50);
    });

    const { container } = render(
      <MainView
        onOpenStory={() => undefined}
        onOpenMenu={() => undefined}
        onOpenSettings={() => undefined}
      />,
    );

    // Wait for the first beat's text to land in the typewriter
    // before driving subsequent clicks.
    await waitFor(() =>
      expect(container.querySelector(".interaction .text")?.textContent).toBe(
        beats[0]!.text,
      ),
    );

    // Track every beat's full text being visible at SOME render.
    const seen = new Set<string>([beats[0]!.text]);
    const observer = new MutationObserver(() => {
      const txt = container.querySelector(".interaction .text")?.textContent ?? "";
      for (const beat of beats) {
        if (txt === beat.text) seen.add(beat.text);
      }
    });
    observer.observe(container, {
      subtree: true,
      childList: true,
      characterData: true,
    });

    try {
      // Drive four rapid clicks on the background — faster than the
      // mocked LLM delay so the renderer's optimistic-action guard
      // and the in-flight beat ordering both get exercised. Each
      // click is a real DOM event, not a programmatic call.
      for (let i = 0; i < 4; i++) {
        const bg = container.querySelector(".background") as HTMLElement;
        await act(async () => {
          fireEvent.click(bg);
        });
        // Wait briefly between clicks — enough for the optimistic
        // guard to release on the next state update but less than
        // the typewriter would need to fully land the beat.
        await new Promise((r) => setTimeout(r, 80));
      }

      // Give the system time to finish typing the last beat AND
      // for the MutationObserver to capture every beat's full text
      // along the way.
      await waitFor(
        () => {
          const txt = container.querySelector(".interaction .text")?.textContent ?? "";
          return txt === beats[beats.length - 1]!.text;
        },
        { timeout: 5000 },
      );
      // Final scrape in case the MutationObserver missed the very
      // last beat (it lands AFTER waitFor resolves).
      const txt = container.querySelector(".interaction .text")?.textContent ?? "";
      for (const beat of beats) {
        if (txt === beat.text) seen.add(beat.text);
      }
    } finally {
      observer.disconnect();
    }

    // Every beat that the backend committed must have appeared in
    // the typewriter at full length at some point.
    for (const beat of beats) {
      expect(
        seen.has(beat.text),
        `beat "${beat.text}" was never fully presented`,
      ).toBe(true);
    }
  });
});
