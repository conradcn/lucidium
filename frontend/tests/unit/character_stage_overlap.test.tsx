/**
 * The CharacterStage exposes the on-stage character count to
 * CSS via the ``--character-count`` custom property. The stage's
 * stylesheet uses it to compute negative ``margin-left`` for
 * each non-first character so 3+ figures overlap slightly
 * instead of overflowing the viewport.
 *
 * jsdom doesn't compute CSS calc / clamp / vw values, so this
 * test pins the React-side contract: the count React passes is
 * the count of VISIBLE NPCs (player excluded, missing
 * characters skipped). The layout math itself lives in the
 * stylesheet and is exercised by the e2e suite + manual play.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { render, cleanup } from "@testing-library/react";

import { useLucidiumStore } from "../../src/state/store";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({ on: () => () => undefined })),
}));

import { CharacterStage } from "../../src/screens/MainView/CharacterStage";

function _char(id: string, isPlayer: boolean): unknown {
  return {
    id, name: id.toUpperCase(), is_player: isPlayer,
    description: "x", gender: "male", age: 30, ethnicity: "local",
    skin: "fair", hair_color: "dark", hairstyle: "short",
    eye_color: "brown", build: "average", bust: "n/a",
    outfit: "wool coat", pose: "standing", expression: "alert",
    images: [],
    facts: [], seed: 1,
  };
}

function setupGame(npcCount: number, includePlayer: boolean): void {
  const characters: Record<string, unknown> = {};
  const onStage: string[] = [];
  if (includePlayer) {
    characters["p"] = _char("p", true);
    onStage.push("p");
  }
  for (let i = 0; i < npcCount; i++) {
    const id = `c${i}`;
    characters[id] = _char(id, false);
    onStage.push(id);
  }
  useLucidiumStore.setState({
    game: {
      characters,
      on_stage: onStage,
      // Minimal world / dialog stubs so the store doesn't yell.
      world: { active_plot_threads: [], dropped_plot_threads: [] },
      dialog_tree: { nodes: {}, root_id: null, committed_path: [] },
      environments: {},
      current_node_id: null,
      cost_telemetry: {
        tokens_in: 0, tokens_out: 0, image_calls: 0, llm_calls: 0,
        latency_ms_total: 0, dollar_estimate: 0,
      },
    } as never,
  });
}

afterEach(() => cleanup());

describe("CharacterStage --character-count", () => {
  it("sets count to 0 when no NPCs are on stage", () => {
    setupGame(0, false);
    const { container } = render(<CharacterStage />);
    const stage = container.querySelector(".stage") as HTMLElement;
    expect(stage.style.getPropertyValue("--character-count")).toBe("0");
  });

  it("sets count to 1 for a single NPC", () => {
    setupGame(1, false);
    const { container } = render(<CharacterStage />);
    const stage = container.querySelector(".stage") as HTMLElement;
    expect(stage.style.getPropertyValue("--character-count")).toBe("1");
  });

  it("excludes the player from the count", () => {
    // Player + 2 NPCs → count = 2 (player not rendered as actor).
    setupGame(2, true);
    const { container } = render(<CharacterStage />);
    const stage = container.querySelector(".stage") as HTMLElement;
    expect(stage.style.getPropertyValue("--character-count")).toBe("2");
    // The DOM should also only render 2 character cards.
    expect(container.querySelectorAll(".character").length).toBe(2);
  });

  it("sets count to 5 for a five-NPC scene (the overlap path)", () => {
    setupGame(5, false);
    const { container } = render(<CharacterStage />);
    const stage = container.querySelector(".stage") as HTMLElement;
    expect(stage.style.getPropertyValue("--character-count")).toBe("5");
    expect(container.querySelectorAll(".character").length).toBe(5);
  });

  it("skips NPCs whose id is in on_stage but not in characters", () => {
    // Backend can transiently emit an on_stage entry for a
    // character that hasn't materialised yet; the count must
    // reflect what actually renders, not the raw on_stage
    // length.
    useLucidiumStore.setState({
      game: {
        characters: { c0: _char("c0", false) },
        on_stage: ["c0", "ghost"],
        world: { active_plot_threads: [], dropped_plot_threads: [] },
        dialog_tree: { nodes: {}, root_id: null, committed_path: [] },
        environments: {},
        current_node_id: null,
        cost_telemetry: {
          tokens_in: 0, tokens_out: 0, image_calls: 0, llm_calls: 0,
          latency_ms_total: 0, dollar_estimate: 0,
        },
      } as never,
    });
    const { container } = render(<CharacterStage />);
    const stage = container.querySelector(".stage") as HTMLElement;
    expect(stage.style.getPropertyValue("--character-count")).toBe("1");
  });
});
