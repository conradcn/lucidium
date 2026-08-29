/**
 * "Apply as backdrop" button in EnvironmentsTab.
 *
 * Pin: a non-active env with a rendered image gets the button;
 * the active env doesn't (already applied); an env with no
 * image_path doesn't (no backdrop to apply). Clicking sends
 * c2s/edit/environment/apply with the right id.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render } from "@testing-library/react";

import { useLucidiumStore } from "../../src/state/store";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({ on: () => () => undefined })),
}));

import { send } from "../../src/app/client";
import { EnvironmentsTab } from "../../src/screens/MainView/StoryPanel/EnvironmentsTab";

function setupGame(opts: {
  envs: Record<
    string,
    { id: string; location_label: string; prompt: string; image_path: string | null }
  >;
  currentLocation: string | null;
}): void {
  useLucidiumStore.setState({
    game: {
      id: "g1",
      schema_version: 1,
      current_node_id: "n1",
      world: {
        game_name: "T", setting: "harbor", genre: "mystery", visual_style: "ink",
        active_plot_threads: [], dropped_plot_threads: [],
        overall_plot_direction: "",
        summarizer_assessment: "",
        prompt_history_clamp_chars: 12000,
        player_intent: {
          pace_preference: "same", tone_preference: "unspecified",
          direction_signal: "none", weighted_evidence: [],
        },
      },
      characters: {},
      dialog_tree: {
        nodes: {
          n1: {
            id: "n1", parent_id: null, chosen_option_id: null,
            speaker_id: null, text: "x", options: [],
            entering_character_ids: [], leaving_character_ids: [],
            new_characters: [], location_id: opts.currentLocation,
            location_prompt: null, character_changes: [],
            state: "committed", premise_hash: "h",
            generation_metadata: {
              model: null, prompt_hash: null, seed_parameters: {},
              tokens_in: 0, tokens_out: 0, latency_ms: 0,
            },
          },
        },
        root_id: "n1", committed_path: ["n1"],
      },
      environments: opts.envs,
      on_stage: [],
      cost_telemetry: {
        tokens_in: 0, tokens_out: 0, image_calls: 0, llm_calls: 0,
        latency_ms_total: 0, dollar_estimate: 0,
      },
    } as never,
  });
}

afterEach(() => cleanup());

describe("EnvironmentsTab — Apply as backdrop button", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
  });

  it("renders the Apply button on a non-active env that has an image", () => {
    setupGame({
      envs: {
        a: {
          id: "a", location_label: "harbor",
          prompt: "harbor", image_path: "/saves/a.png",
        },
        b: {
          id: "b", location_label: "tavern",
          prompt: "tavern", image_path: "/saves/b.png",
        },
      },
      currentLocation: "a",
    });
    const { queryByTestId } = render(<EnvironmentsTab />);
    // Active env (a) — no Apply button (already applied).
    expect(queryByTestId("apply-env-a")).toBeNull();
    // Non-active env (b) with an image — Apply button present.
    expect(queryByTestId("apply-env-b")).toBeTruthy();
  });

  it("hides the Apply button on an env with no rendered image", () => {
    setupGame({
      envs: {
        a: {
          id: "a", location_label: "harbor",
          prompt: "harbor", image_path: "/saves/a.png",
        },
        b: {
          id: "b", location_label: "tavern",
          prompt: "tavern", image_path: null,  // not rendered yet
        },
      },
      currentLocation: "a",
    });
    const { queryByTestId } = render(<EnvironmentsTab />);
    // No image -> nothing to apply.
    expect(queryByTestId("apply-env-b")).toBeNull();
  });

  it("sends c2s/edit/environment/apply when clicked", () => {
    setupGame({
      envs: {
        a: {
          id: "a", location_label: "harbor",
          prompt: "harbor", image_path: "/saves/a.png",
        },
        b: {
          id: "b", location_label: "tavern",
          prompt: "tavern", image_path: "/saves/b.png",
        },
      },
      currentLocation: "a",
    });
    const { getByTestId } = render(<EnvironmentsTab />);
    fireEvent.click(getByTestId("apply-env-b"));
    const call = vi.mocked(send).mock.calls.find(
      (c) => c[0] === "c2s/edit/environment/apply",
    );
    expect(call).toBeTruthy();
    expect((call?.[1] as { environment_id: string }).environment_id).toBe("b");
  });

  it("Rerender button still works alongside Apply", () => {
    setupGame({
      envs: {
        a: {
          id: "a", location_label: "harbor",
          prompt: "harbor", image_path: "/saves/a.png",
        },
        b: {
          id: "b", location_label: "tavern",
          prompt: "tavern", image_path: "/saves/b.png",
        },
      },
      currentLocation: "a",
    });
    const { container } = render(<EnvironmentsTab />);
    const rerenderButtons = Array.from(container.querySelectorAll("button"))
      .filter((b) => /^Rerender$/i.test(b.textContent?.trim() ?? ""));
    // One per env.
    expect(rerenderButtons.length).toBe(2);
  });
});
