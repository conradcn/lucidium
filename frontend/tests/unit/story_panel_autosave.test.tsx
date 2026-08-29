/**
 * Story-panel tabs autosave their edits via the existing
 * ``c2s/edit/*`` messages — no per-field Save buttons. Pin
 * the CharactersTab + WorldInfoTab + EnvironmentsTab paths so
 * a refactor that re-adds a Save button trips the test.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, cleanup, fireEvent } from "@testing-library/react";

import { useLucidiumStore } from "../../src/state/store";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({ on: () => () => undefined })),
}));

import { send } from "../../src/app/client";
import { CharactersTab } from "../../src/screens/MainView/StoryPanel/CharactersTab";
import { WorldInfoTab } from "../../src/screens/MainView/StoryPanel/WorldInfoTab";
import { EnvironmentsTab } from "../../src/screens/MainView/StoryPanel/EnvironmentsTab";

function setupGameWithChar(): void {
  useLucidiumStore.setState({
    game: {
      id: "g1",
      schema_version: 1,
      current_node_id: "n1",
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
        c1: {
          id: "c1", is_player: false, name: "Hale", description: "the keeper",
          kind: "human", physical_description: "",
          gender: "male", age: 50, ethnicity: "local",
          skin: "tan", hair_color: "grey", hairstyle: "short",
          eye_color: "brown", build: "stocky", bust: "n/a",
          outfit: "oilskin", pose: "leaning", expression: "watchful",
          facts: [], images: [], seed: 1,
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
      environments: {
        e1: { id: "e1", location_label: "harbor", prompt: "stone harbor at dawn", image_path: null, seed: 1 },
      },
      on_stage: ["c1"],
      cost_telemetry: {
        tokens_in: 0, tokens_out: 0, image_calls: 0, llm_calls: 0,
        latency_ms_total: 0, dollar_estimate: 0,
      },
    } as never,
  });
}

afterEach(() => cleanup());

// NPC cards default-collapse on the cast tab — tests must click
// the header to reveal the editor fields.
function expandHaleCard(container: HTMLElement): void {
  const header = Array.from(container.querySelectorAll("h3"))
    .find((h) => h.textContent?.includes("Hale"));
  if (!header) throw new Error("Hale card header not found");
  fireEvent.click(header);
}

describe("Story-panel autosave", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
  });

  it("CharactersTab field edits autosave (no Save buttons in the field rows)", async () => {
    setupGameWithChar();
    const { container } = render(<CharactersTab />);
    expandHaleCard(container);
    const outfitInput = Array.from(
      container.querySelectorAll("input[type='text']"),
    ).find((el) => (el as HTMLInputElement).value === "oilskin") as
      | HTMLInputElement
      | undefined;
    expect(outfitInput).toBeTruthy();
    fireEvent.change(outfitInput!, { target: { value: "weathered grey oilskin" } });
    // Per-field Save button must be gone from the field rows.
    const fieldSaveButtons = Array.from(container.querySelectorAll("button"))
      .filter((b) => /^Save$/i.test(b.textContent?.trim() ?? ""));
    expect(fieldSaveButtons.length).toBe(0);
    // Autosave debounces 400 ms.
    await new Promise((r) => setTimeout(r, 500));
    const call = vi.mocked(send).mock.calls.find(
      (c) =>
        c[0] === "c2s/edit/character"
        && (c[1] as { field: string; value: string }).field === "outfit"
        && (c[1] as { value: string }).value === "weathered grey oilskin",
    );
    expect(call).toBeTruthy();
  });

  it("CharactersTab Rerender flushes a pending pose edit BEFORE sending the rerender", async () => {
    // Repro for "rerenders don't always trigger an actual rerender,
    // even when a pose has been changed manually": the pose edit sits
    // in the 400 ms autosave debounce. Clicking Rerender immediately
    // used to fire ``c2s/edit/character/rerender`` first; the backend
    // rendered the STALE pose, the prompt_hash was unchanged, and the
    // render was short-circuited. The button must flush the pending
    // edit first so the edit reaches the backend before the rerender.
    setupGameWithChar();
    const { container } = render(<CharactersTab />);
    expandHaleCard(container);
    const poseInput = Array.from(
      container.querySelectorAll("input[type='text']"),
    ).find((el) => (el as HTMLInputElement).value === "leaning") as
      | HTMLInputElement
      | undefined;
    expect(poseInput).toBeTruthy();
    // Type a new pose, then click Rerender WELL WITHIN the 400 ms
    // debounce so the pose edit is still pending.
    fireEvent.change(poseInput!, { target: { value: "arms crossed" } });
    const rerenderBtn = Array.from(container.querySelectorAll("button"))
      .find((b) => /^Rerender$/i.test(b.textContent?.trim() ?? ""));
    expect(rerenderBtn).toBeTruthy();
    fireEvent.click(rerenderBtn!);

    const calls = vi.mocked(send).mock.calls;
    const editIdx = calls.findIndex(
      (c) =>
        c[0] === "c2s/edit/character"
        && (c[1] as { field: string; value: string }).field === "pose"
        && (c[1] as { value: string }).value === "arms crossed",
    );
    const rerenderIdx = calls.findIndex(
      (c) => c[0] === "c2s/edit/character/rerender",
    );
    // Both must have been sent, and the pose edit must come FIRST.
    expect(editIdx).toBeGreaterThanOrEqual(0);
    expect(rerenderIdx).toBeGreaterThanOrEqual(0);
    expect(editIdx).toBeLessThan(rerenderIdx);
  });

  it("WorldInfoTab edits autosave", async () => {
    setupGameWithChar();
    const { container } = render(<WorldInfoTab />);
    const settingTextarea = Array.from(container.querySelectorAll("textarea"))
      .find((t) => (t as HTMLTextAreaElement).value === "harbor") as
      | HTMLTextAreaElement
      | undefined;
    expect(settingTextarea).toBeTruthy();
    fireEvent.change(settingTextarea!, { target: { value: "salt-bleached harbor" } });
    await new Promise((r) => setTimeout(r, 500));
    const call = vi.mocked(send).mock.calls.find(
      (c) =>
        c[0] === "c2s/edit/world"
        && (c[1] as { field: string; value: string }).field === "setting"
        && (c[1] as { value: string }).value === "salt-bleached harbor",
    );
    expect(call).toBeTruthy();
  });

  it("CharactersTab keeps the typed draft on screen after the autosave fires (so the cursor doesn't reset to the end)", async () => {
    // Repro for the "can only type at the end" bug: when the
    // 400 ms autosave fired, the previous implementation wiped
    // the local draft. The input ``value`` then snapped from the
    // typed string back to the OLD canonical value (the
    // state/patch from the backend hasn't arrived yet), which
    // forced React to set ``input.value`` programmatically and
    // moved the cursor to the end. The fix is to keep the draft
    // until canonical catches up — pinned here.
    setupGameWithChar();
    const { container } = render(<CharactersTab />);
    expandHaleCard(container);
    const outfitInput = Array.from(
      container.querySelectorAll("input[type='text']"),
    ).find((el) => (el as HTMLInputElement).value === "oilskin") as
      | HTMLInputElement
      | undefined;
    expect(outfitInput).toBeTruthy();
    // Simulate typing in the middle: change "oilskin" → "oilXskin"
    fireEvent.change(outfitInput!, { target: { value: "oilXskin" } });
    expect(outfitInput!.value).toBe("oilXskin");
    // Let the 400 ms debounce fire. State/patch hasn't arrived;
    // canonical is still "oilskin". The draft must persist so
    // the input still shows the typed string.
    await new Promise((r) => setTimeout(r, 500));
    expect(outfitInput!.value).toBe("oilXskin");
  });

  it("CharactersTab clears the draft once canonical state catches up", async () => {
    setupGameWithChar();
    const { container, rerender } = render(<CharactersTab />);
    expandHaleCard(container);
    const outfitInput = Array.from(
      container.querySelectorAll("input[type='text']"),
    ).find((el) => (el as HTMLInputElement).value === "oilskin") as
      | HTMLInputElement
      | undefined;
    expect(outfitInput).toBeTruthy();
    fireEvent.change(outfitInput!, { target: { value: "weathered grey oilskin" } });
    await new Promise((r) => setTimeout(r, 500));
    // Backend echoes back via state/patch — push the new
    // canonical value into the store. Drafts should clear.
    const game = useLucidiumStore.getState().game as unknown as {
      characters: Record<string, Record<string, unknown>>;
    };
    useLucidiumStore.setState({
      game: {
        ...(game as object),
        characters: {
          ...game.characters,
          c1: { ...game.characters.c1, outfit: "weathered grey oilskin" },
        },
      } as never,
    });
    rerender(<CharactersTab />);
    // After canonical catches up, the input should still show
    // the new value (now sourced from canonical, not the draft).
    const refreshedInput = Array.from(
      container.querySelectorAll("input[type='text']"),
    ).find(
      (el) => (el as HTMLInputElement).value === "weathered grey oilskin",
    );
    expect(refreshedInput).toBeTruthy();
  });

  it("EnvironmentsTab edits autosave", async () => {
    setupGameWithChar();
    const { container } = render(<EnvironmentsTab />);
    const labelInput = Array.from(
      container.querySelectorAll("input[type='text']"),
    ).find((el) => (el as HTMLInputElement).value === "harbor") as
      | HTMLInputElement
      | undefined;
    expect(labelInput).toBeTruthy();
    fireEvent.change(labelInput!, { target: { value: "stone harbor at dawn" } });
    await new Promise((r) => setTimeout(r, 500));
    const call = vi.mocked(send).mock.calls.find(
      (c) =>
        c[0] === "c2s/edit/environment"
        && (c[1] as { field: string; value: string }).field === "location_label"
        && (c[1] as { value: string }).value === "stone harbor at dawn",
    );
    expect(call).toBeTruthy();
  });
});
