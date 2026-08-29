/**
 * Click-anywhere-to-continue: when the current dialog node has no
 * explicit options, clicking on the stage area (background, character
 * portrait, the empty space above the dialog box) MUST dispatch
 * c2s/play/advance with option_id=null. Visual-novel convention.
 *
 * Regression: clicking anywhere on the window did NOT trigger continue
 * because (a) the canImplicitContinue gate fired in the wrong cases or
 * (b) the click target's closest() match swallowed the event.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, cleanup, fireEvent } from "@testing-library/react";

import { useLucidiumStore } from "../../src/state/store";
import { gameFixture, settingsFixture } from "../fixtures";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({ on: () => () => undefined })),
}));

import { send } from "../../src/app/client";
import { MainView } from "../../src/screens/MainView";

function makeGameState({
  options = [] as { id: string; text: string }[],
  speakerId = null as string | null,
} = {}) {
  return {
    game: gameFixture({
      id: "g1",
      schema_version: 1,
      current_node_id: "n1",
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
      characters: {
        c1: {
          id: "c1",
          is_player: false,
          name: "Hale",
          description: "keeper",
          gender: "male",
          age: 50,
          ethnicity: "local",
          skin: "tan",
          hair_color: "grey",
          hairstyle: "short",
          eye_color: "brown",
          build: "stocky",
          bust: "n/a",
          outfit: "oilskin",
          pose: "standing",
          expression: "watchful",
          facts: [],
          images: [],
          seed: 7,
        },
      },
      dialog_tree: {
        nodes: {
          n1: {
            id: "n1",
            parent_id: null,
            chosen_option_id: null,
            speaker_id: speakerId,
            text: "The harbor wakes slow.",
            options,
            entering_character_ids: [],
            leaving_character_ids: [],
            new_characters: [],
            location_id: null,
            location_prompt: null,
            character_changes: [],
            state: "committed",
            premise_hash: "h0",
            generation_metadata: {
              model: null,
              prompt_hash: null,
              seed_parameters: {},
              tokens_in: 0,
              tokens_out: 0,
              latency_ms: 0,
            },
          },
        },
        root_id: "n1",
        committed_path: ["n1"],
      },
      environments: {},
      on_stage: ["c1"],
      cost_telemetry: {
        tokens_in: 0,
        tokens_out: 0,
        image_calls: 0,
        llm_calls: 0,
        latency_ms_total: 0,
        dollar_estimate: 0,
      },
    }),
    settings: settingsFixture({ typewriter_speed_chars_per_sec: 1000 }),
    hasSave: true,
    status: "open" as const,
  };
}

afterEach(() => cleanup());

describe("MainView click-anywhere-to-continue", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
  });

  it("clicking the background dispatches c2s/play/advance when there are no options", () => {
    useLucidiumStore.setState(makeGameState({ options: [] }));
    const { container } = render(
      <MainView onOpenStory={() => undefined} onOpenMenu={() => undefined} onOpenSettings={() => undefined} />,
    );

    const background = container.querySelector(".background") as HTMLElement;
    expect(background).not.toBeNull();
    fireEvent.click(background);
    expect(send).toHaveBeenCalledWith("c2s/play/advance", { option_id: null });
  });

  it("clicking on the character portrait area also advances", () => {
    useLucidiumStore.setState(makeGameState({ options: [] }));
    const { container } = render(
      <MainView onOpenStory={() => undefined} onOpenMenu={() => undefined} onOpenSettings={() => undefined} />,
    );
    // The placeholder div renders when there's no portrait yet — click it.
    const placeholder = container.querySelector(".character .placeholder");
    if (placeholder) {
      fireEvent.click(placeholder);
      expect(send).toHaveBeenCalledWith("c2s/play/advance", { option_id: null });
    } else {
      // If a portrait <img> rendered instead, click that.
      const img = container.querySelector(".character img");
      expect(img).not.toBeNull();
      fireEvent.click(img!);
      expect(send).toHaveBeenCalledWith("c2s/play/advance", { option_id: null });
    }
  });

  it("clicking the stage area (between characters) advances", () => {
    useLucidiumStore.setState(makeGameState({ options: [] }));
    const { container } = render(
      <MainView onOpenStory={() => undefined} onOpenMenu={() => undefined} onOpenSettings={() => undefined} />,
    );
    const stage = container.querySelector(".stage") as HTMLElement;
    expect(stage).not.toBeNull();
    fireEvent.click(stage);
    expect(send).toHaveBeenCalledWith("c2s/play/advance", { option_id: null });
  });

  it("clicking on the dialog text itself advances — VN convention", () => {
    useLucidiumStore.setState(makeGameState({ options: [] }));
    const { container } = render(
      <MainView onOpenStory={() => undefined} onOpenMenu={() => undefined} onOpenSettings={() => undefined} />,
    );
    const text = container.querySelector(".interaction .text") as HTMLElement;
    expect(text).not.toBeNull();
    fireEvent.click(text);
    // Clicking the text is the most natural "I read it, next" gesture.
    expect(send).toHaveBeenCalledWith("c2s/play/advance", { option_id: null });
  });

  it("clicking the empty area inside .interaction (between text and free-text input) advances", () => {
    useLucidiumStore.setState(makeGameState({ options: [] }));
    const { container } = render(
      <MainView onOpenStory={() => undefined} onOpenMenu={() => undefined} onOpenSettings={() => undefined} />,
    );
    const interaction = container.querySelector(".interaction") as HTMLElement;
    expect(interaction).not.toBeNull();
    fireEvent.click(interaction);
    expect(send).toHaveBeenCalledWith("c2s/play/advance", { option_id: null });
  });

  it("clicking the free-text input does NOT advance (typing target)", () => {
    useLucidiumStore.setState(makeGameState({ options: [] }));
    const { container } = render(
      <MainView onOpenStory={() => undefined} onOpenMenu={() => undefined} onOpenSettings={() => undefined} />,
    );
    const input = container.querySelector(
      ".free-text input",
    ) as HTMLInputElement | null;
    expect(input).not.toBeNull();
    fireEvent.click(input!);
    expect(send).not.toHaveBeenCalled();
  });

  it("does NOT auto-advance when there are explicit options to choose from", () => {
    useLucidiumStore.setState(
      makeGameState({
        options: [
          { id: "o1", text: "Walk to the archive." },
          { id: "o2", text: "Stay a moment." },
        ],
      }),
    );
    const { container } = render(
      <MainView onOpenStory={() => undefined} onOpenMenu={() => undefined} onOpenSettings={() => undefined} />,
    );
    const background = container.querySelector(".background") as HTMLElement;
    fireEvent.click(background);
    // Choices on the table — silently advancing would lose the
    // player's pick. Click must be a no-op.
    expect(send).not.toHaveBeenCalled();
  });

  it("clicking the top-bar area (away from the game stage) does NOT advance", () => {
    useLucidiumStore.setState(makeGameState({ options: [] }));
    const { container } = render(
      <MainView onOpenStory={() => undefined} onOpenMenu={() => undefined} onOpenSettings={() => undefined} />,
    );
    // Click the top-bar container itself (not a button inside it).
    // The click-advance whitelist must NOT include the top-bar
    // chrome — accidentally advancing because the player aimed
    // for a button and missed by a few pixels is the bug we're
    // protecting against here.
    const topBar = container.querySelector(".top-bar") as HTMLElement;
    expect(topBar).not.toBeNull();
    fireEvent.click(topBar);
    expect(send).not.toHaveBeenCalled();
  });

  it("clicking on the bare main-view (outside background/stage/dialog) does NOT advance", () => {
    useLucidiumStore.setState(makeGameState({ options: [] }));
    const { container } = render(
      <MainView onOpenStory={() => undefined} onOpenMenu={() => undefined} onOpenSettings={() => undefined} />,
    );
    // The outermost main-view container fills the window. Any
    // click landing on bare main-view chrome (between the
    // top-bar and the background, or in margin space if a future
    // layout adds one) must be a no-op — only the three
    // whitelisted regions count as "the game screen".
    const mainView = container.querySelector(
      "[data-testid='main-view']",
    ) as HTMLElement;
    expect(mainView).not.toBeNull();
    fireEvent.click(mainView);
    expect(send).not.toHaveBeenCalled();
  });

  it("clicking the top-bar Story button does NOT also fire continue", () => {
    useLucidiumStore.setState(makeGameState({ options: [] }));
    const { container } = render(
      <MainView onOpenStory={() => undefined} onOpenMenu={() => undefined} onOpenSettings={() => undefined} />,
    );
    const storyBtn = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent?.trim() === "Story",
    ) as HTMLButtonElement | undefined;
    expect(storyBtn).toBeDefined();
    fireEvent.click(storyBtn!);
    expect(send).not.toHaveBeenCalled();
  });
});
