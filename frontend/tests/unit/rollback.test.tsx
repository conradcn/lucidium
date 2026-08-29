/**
 * Ren'Py-style rollback. Wheel-up over the main view replaces the
 * dialog text with the previous committed beat, suppresses
 * options + free-text + Continue, and shows a banner. Wheel-down
 * walks back toward the live beat; clicking or pressing Space
 * during rollback exits to the live beat instead of advancing.
 *
 * Pin both layers: the InteractionPanel renders the historical
 * beat at offset > 0 (no controls), and the MainView wires
 * wheel/space/PageUp into that offset.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, cleanup, fireEvent } from "@testing-library/react";

import { useLucidiumStore } from "../../src/state/store";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({ on: () => () => undefined })),
}));

import { send } from "../../src/app/client";
import { InteractionPanel } from "../../src/screens/MainView/InteractionPanel";
import { MainView } from "../../src/screens/MainView";

function gameWithThreeBeats(): Record<string, unknown> {
  return {
    id: "g1",
    schema_version: 1,
    current_node_id: "n3",
    world: {
      game_name: "T", setting: "harbor", genre: "mystery", visual_style: "ink",
      active_plot_threads: [], dropped_plot_threads: [],
      overall_plot_direction: "find the keeper",
      summarizer_assessment: "",
      prompt_history_clamp_chars: 12000,
      player_intent: {
        pace_preference: "same", tone_preference: "unspecified",
        direction_signal: "none", weighted_evidence: [],
      },
    },
    characters: {
      hale: {
        id: "hale", is_player: false, name: "Hale", description: "",
        kind: "human", physical_description: "",
        gender: "male", age: 40, ethnicity: "local",
        skin: "tan", hair_color: "grey", hairstyle: "short",
        eye_color: "brown", build: "stocky", bust: "n/a",
        outfit: "oilskin", pose: "leaning", expression: "watchful",
        facts: [], images: [], seed: 1,
      },
    },
    dialog_tree: {
      nodes: {
        n1: {
          id: "n1", parent_id: null, chosen_option_id: null,
          speaker_id: null, text: "You step onto the wet stones.",
          options: [], state: "committed",
          entering_character_ids: [], leaving_character_ids: [],
          new_characters: [], location_id: null, location_prompt: null,
          character_changes: [], premise_hash: "h1",
          generation_metadata: {
            model: null, prompt_hash: null, seed_parameters: {},
            tokens_in: 0, tokens_out: 0, latency_ms: 0,
          },
        },
        n2: {
          id: "n2", parent_id: "n1", chosen_option_id: null,
          speaker_id: "hale", text: "I've been waiting.",
          options: [], state: "committed",
          entering_character_ids: [], leaving_character_ids: [],
          new_characters: [], location_id: null, location_prompt: null,
          character_changes: [], premise_hash: "h2",
          generation_metadata: {
            model: null, prompt_hash: null, seed_parameters: {},
            tokens_in: 0, tokens_out: 0, latency_ms: 0,
          },
        },
        n3: {
          id: "n3", parent_id: "n2", chosen_option_id: null,
          speaker_id: null,
          text: "He gestures toward the lighthouse.",
          options: [
            { id: "follow", text: "Follow him." },
            { id: "wait", text: "Stay where you are." },
          ],
          state: "committed",
          entering_character_ids: [], leaving_character_ids: [],
          new_characters: [], location_id: null, location_prompt: null,
          character_changes: [], premise_hash: "h3",
          generation_metadata: {
            model: null, prompt_hash: null, seed_parameters: {},
            tokens_in: 0, tokens_out: 0, latency_ms: 0,
          },
        },
      },
      committed_path: ["n1", "n2", "n3"],
      root_id: "n1",
    },
    on_stage: ["hale"],
    cost_telemetry: {
      tokens_in: 0, tokens_out: 0, image_calls: 0, llm_calls: 0,
      latency_ms_total: 0, dollar_estimate: 0,
    },
  };
}

function setupGame(): void {
  useLucidiumStore.setState({
    game: gameWithThreeBeats() as never,
    settings: { typewriter_speed_chars_per_sec: 1000 } as never,
    status: "open" as never,
  });
}

afterEach(() => cleanup());

describe("InteractionPanel rollback rendering", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
    setupGame();
  });

  it("renders the live beat with options at offset 0", () => {
    const { container } = render(<InteractionPanel rollbackOffset={0} flushSignal={1} />);
    const text = container.querySelector(".text");
    expect(text?.textContent).toContain("He gestures toward the lighthouse.");
    const optionButtons = Array.from(
      container.querySelectorAll(".options button"),
    );
    expect(optionButtons.length).toBe(2);
    expect(container.querySelector('[data-testid="rollback-banner"]')).toBeNull();
    expect(container.querySelector(".free-text")).toBeTruthy();
  });

  it("offset=1 renders the previous committed beat", () => {
    const { container } = render(<InteractionPanel rollbackOffset={1} />);
    expect(container.querySelector(".text")?.textContent).toContain(
      "I've been waiting.",
    );
    expect(container.querySelector(".text")?.textContent).not.toContain(
      "He gestures toward the lighthouse.",
    );
  });

  it("offset=2 renders two beats back", () => {
    const { container } = render(<InteractionPanel rollbackOffset={2} />);
    expect(container.querySelector(".text")?.textContent).toContain(
      "You step onto the wet stones.",
    );
  });

  it("rollback mode hides options + Continue + free-text and shows the banner", () => {
    const { container } = render(<InteractionPanel rollbackOffset={1} />);
    expect(container.querySelector('[data-testid="rollback-banner"]')).toBeTruthy();
    expect(container.querySelector(".options")).toBeNull();
    expect(container.querySelector(".free-text")).toBeNull();
    expect(container.querySelector(".continue-glyph")).toBeNull();
  });

  it("banner pluralisation honours singular vs plural", () => {
    const { container, rerender } = render(<InteractionPanel rollbackOffset={1} />);
    expect(
      container.querySelector('[data-testid="rollback-banner"]')?.textContent,
    ).toContain("1 beat ");
    rerender(<InteractionPanel rollbackOffset={2} />);
    expect(
      container.querySelector('[data-testid="rollback-banner"]')?.textContent,
    ).toContain("2 beats ");
  });
});

describe("MainView wheel rollback", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
    setupGame();
  });

  it("wheel up rolls back; wheel down rolls forward", () => {
    const { container } = render(
      <MainView
        onOpenStory={() => undefined}
        onOpenMenu={() => undefined}
        onOpenSettings={() => undefined}
      />,
    );
    const view = container.querySelector(
      '[data-testid="main-view"]',
    ) as HTMLDivElement;
    expect(container.querySelector('[data-testid="rollback-banner"]')).toBeNull();
    fireEvent.wheel(view, { deltaY: -100 });
    expect(container.querySelector('[data-testid="rollback-banner"]')).toBeTruthy();
    expect(
      container.querySelector(".text")?.textContent,
    ).toContain("I've been waiting.");
    fireEvent.wheel(view, { deltaY: 100 });
    expect(container.querySelector('[data-testid="rollback-banner"]')).toBeNull();
    expect(
      container.querySelector(".text")?.textContent,
    ).toContain("He gestures toward the lighthouse.");
  });

  it("wheel up clamps at the start of the committed path", () => {
    const { container } = render(
      <MainView
        onOpenStory={() => undefined}
        onOpenMenu={() => undefined}
        onOpenSettings={() => undefined}
      />,
    );
    const view = container.querySelector(
      '[data-testid="main-view"]',
    ) as HTMLDivElement;
    // 3 beats → max offset is 2 (rolled back to the first beat).
    fireEvent.wheel(view, { deltaY: -100 });
    fireEvent.wheel(view, { deltaY: -100 });
    fireEvent.wheel(view, { deltaY: -100 });
    fireEvent.wheel(view, { deltaY: -100 });
    expect(
      container.querySelector(".text")?.textContent,
    ).toContain("You step onto the wet stones.");
  });

  it("clicking the stage during rollback exits to the live beat (no advance)", () => {
    const { container } = render(
      <MainView
        onOpenStory={() => undefined}
        onOpenMenu={() => undefined}
        onOpenSettings={() => undefined}
      />,
    );
    const view = container.querySelector(
      '[data-testid="main-view"]',
    ) as HTMLDivElement;
    fireEvent.wheel(view, { deltaY: -100 });
    expect(container.querySelector('[data-testid="rollback-banner"]')).toBeTruthy();
    // Click the background (a stage region without buttons).
    const stage = container.querySelector(".stage") as HTMLDivElement;
    fireEvent.click(stage);
    // Banner gone — back at the live beat.
    expect(container.querySelector('[data-testid="rollback-banner"]')).toBeNull();
    // No advance was sent (we exited rollback rather than moving forward).
    const advanceCalls = vi.mocked(send).mock.calls.filter(
      (c) => c[0] === "c2s/play/advance",
    );
    expect(advanceCalls.length).toBe(0);
  });
});
