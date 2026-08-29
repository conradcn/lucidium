/**
 * The Cast tab carries a ``kind`` selector (human / nonhuman)
 * for each character. Flipping it dispatches a
 * ``c2s/edit/character`` patch with field=kind so the backend's
 * portrait pipeline switches between the structured-anatomy
 * (human) and freeform-physical_description (nonhuman) paths.
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
import { CharactersTab } from "../../src/screens/MainView/StoryPanel/CharactersTab";

function makeGameState() {
  return {
    game: gameFixture({
      id: "g1",
      schema_version: 1,
      current_node_id: "n1",
      world: {
        game_name: "T", setting: "harbor", genre: "mystery", visual_style: "ink",
        active_plot_threads: [], dropped_plot_threads: [],
        overall_plot_direction: "", summarizer_assessment: "",
        prompt_history_clamp_chars: 12000,
        player_intent: {
          pace_preference: "same", tone_preference: "unspecified",
          direction_signal: "none", weighted_evidence: [],
        },
      },
      characters: {
        c2: {
          id: "c2",
          is_player: false,
          name: "Hale",
          description: "the keeper",
          kind: "human",
          physical_description: "",
          gender: "male", age: 50, ethnicity: "local",
          skin: "tan", hair_color: "grey", hairstyle: "short",
          eye_color: "brown", build: "stocky", bust: "n/a",
          outfit: "oilskin", pose: "leaning", expression: "watchful",
          facts: [], images: [], seed: 7,
        },
      },
      dialog_tree: {
        nodes: { n1: {
          id: "n1", parent_id: null, chosen_option_id: null,
          speaker_id: null, text: "x", options: [],
          entering_character_ids: [], leaving_character_ids: [],
          new_characters: [], location_id: null, location_prompt: null,
          character_changes: [], state: "committed", premise_hash: "h",
          generation_metadata: {
            model: null, prompt_hash: null, seed_parameters: {},
            tokens_in: 0, tokens_out: 0, latency_ms: 0,
          },
        }},
        root_id: "n1", committed_path: ["n1"],
      },
      environments: {}, on_stage: ["c2"],
      cost_telemetry: {
        tokens_in: 0, tokens_out: 0, image_calls: 0, llm_calls: 0,
        latency_ms_total: 0, dollar_estimate: 0,
      },
    }),
    settings: settingsFixture({}),
    hasSave: true, status: "open" as const,
  };
}

afterEach(() => cleanup());

// NPC cards default-collapse on the cast tab — tests have to
// click the header before any of the editor controls render.
function expandHaleCard(container: HTMLElement): void {
  const header = Array.from(container.querySelectorAll("h3"))
    .find((h) => h.textContent?.includes("Hale"));
  if (!header) throw new Error("Hale card header not found");
  fireEvent.click(header);
}

describe("Character kind editor", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
  });

  it("renders a kind selector populated from the character's kind", () => {
    useLucidiumStore.setState(makeGameState());
    const { container } = render(<CharactersTab />);
    expandHaleCard(container);
    const select = Array.from(container.querySelectorAll("select"))
      .find((s) => Array.from(s.options).some((o) => o.value === "nonhuman"));
    expect(select).toBeTruthy();
    expect((select as HTMLSelectElement).value).toBe("human");
  });

  it("dispatches c2s/edit/character with field=kind on selection change", () => {
    useLucidiumStore.setState(makeGameState());
    const { container } = render(<CharactersTab />);
    expandHaleCard(container);
    const select = Array.from(container.querySelectorAll("select"))
      .find((s) => Array.from(s.options).some((o) => o.value === "nonhuman"))!;
    fireEvent.change(select, { target: { value: "nonhuman" } });
    expect(send).toHaveBeenCalledWith("c2s/edit/character", {
      character_id: "c2",
      field: "kind",
      value: "nonhuman",
    });
  });
});
