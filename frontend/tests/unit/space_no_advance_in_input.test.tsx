/**
 * Spacebar in the "Or do something else..." free-text input must
 * type a space character into the field, NOT advance the dialog.
 *
 * Regression: the window-level keydown listener that wires
 * spacebar → c2s/play/advance had a focus-detection gap — it
 * read ``document.activeElement.tagName``, which can briefly be
 * ``BODY`` mid-keystroke (e.g. on disabled-input transitions),
 * letting space leak through to the advance path while the user
 * was actively typing.
 */

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { render, cleanup, fireEvent } from "@testing-library/react";

import { useLucidiumStore } from "../../src/state/store";
import { gameFixture, settingsFixture } from "../fixtures";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({ on: () => () => undefined })),
}));

import { send } from "../../src/app/client";
import { MainView } from "../../src/screens/MainView";

function makeGameState() {
  return {
    game: gameFixture({
      id: "g1",
      schema_version: 1,
      current_node_id: "n1",
      world: {
        game_name: "T",
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
        nodes: {
          n1: {
            id: "n1",
            parent_id: null,
            chosen_option_id: null,
            speaker_id: null,
            text: "The harbor wakes slow.",
            options: [],
            entering_character_ids: [],
            leaving_character_ids: [],
            new_characters: [],
            location_id: null,
            location_prompt: null,
            character_changes: [],
            state: "committed",
            premise_hash: "h0",
            generation_metadata: {
              model: null, prompt_hash: null, seed_parameters: {},
              tokens_in: 0, tokens_out: 0, latency_ms: 0,
            },
          },
        },
        root_id: "n1",
        committed_path: ["n1"],
      },
      environments: {},
      on_stage: [],
      cost_telemetry: {
        tokens_in: 0, tokens_out: 0, image_calls: 0, llm_calls: 0,
        latency_ms_total: 0, dollar_estimate: 0,
      },
    }),
    settings: settingsFixture({ typewriter_speed_chars_per_sec: 1000 }),
    hasSave: true,
    status: "open" as const,
  };
}

// jsdom's ``document.hasFocus()`` defaults to ``false`` — the new
// focus gate in MainView's keydown handler bails on
// !document.hasFocus(), which would make every test in this
// suite a no-op without an override. Stub to ``true`` for the
// happy paths; the alt-tab test below switches it back to false
// to verify the gate.
const _originalHasFocus = document.hasFocus.bind(document);
beforeAll(() => {
  document.hasFocus = () => true;
});
afterAll(() => {
  document.hasFocus = _originalHasFocus;
});

afterEach(() => cleanup());

describe("Space key in free-text input", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
  });

  it("does not dispatch c2s/play/advance when space is pressed inside the free-text input", () => {
    useLucidiumStore.setState(makeGameState());
    const { container } = render(
      <MainView
        onOpenStory={() => undefined}
        onOpenMenu={() => undefined}
        onOpenSettings={() => undefined}
      />,
    );

    const input = container.querySelector(
      ".free-text input",
    ) as HTMLInputElement | null;
    expect(input).not.toBeNull();
    input!.focus();
    expect(document.activeElement).toBe(input);

    // Press space inside the input. Two paths must hold:
    //   1. The input's onKeyDown stops propagation so the
    //      window-level listener never sees the event.
    //   2. The window-level listener also re-checks the focus
    //      target as a defensive backstop.
    fireEvent.keyDown(input!, { key: " ", code: "Space" });
    expect(send).not.toHaveBeenCalledWith(
      "c2s/play/advance",
      expect.anything(),
    );
  });

  it("space anywhere ELSE on the page still advances when there are no options", () => {
    useLucidiumStore.setState(makeGameState());
    render(
      <MainView
        onOpenStory={() => undefined}
        onOpenMenu={() => undefined}
        onOpenSettings={() => undefined}
      />,
    );

    // Don't focus an input — fire on the body. The advance must
    // still go through, otherwise we've broken the convenience
    // hotkey.
    fireEvent.keyDown(document.body, { key: " ", code: "Space" });
    expect(send).toHaveBeenCalledWith(
      "c2s/play/advance",
      { option_id: null },
    );
  });

  it("space does NOT advance when the window has lost OS focus", () => {
    useLucidiumStore.setState(makeGameState());
    render(
      <MainView
        onOpenStory={() => undefined}
        onOpenMenu={() => undefined}
        onOpenSettings={() => undefined}
      />,
    );

    // Stub document.hasFocus to simulate an alt-tabbed renderer.
    // Without this gate, an Electron child window or DevTools
    // panel could still route keystrokes to the main window's
    // listener and silently advance the dialog while the player
    // typed somewhere else entirely.
    const originalHasFocus = document.hasFocus.bind(document);
    document.hasFocus = () => false;
    try {
      fireEvent.keyDown(document.body, { key: " ", code: "Space" });
      expect(send).not.toHaveBeenCalledWith(
        "c2s/play/advance",
        expect.anything(),
      );
    } finally {
      document.hasFocus = originalHasFocus;
    }
  });
});
