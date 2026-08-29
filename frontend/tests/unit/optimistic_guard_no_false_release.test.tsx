/**
 * Regression: when a choice click is in flight, background events
 * (async asset state/full, summarizer state/full, scheduler status)
 * MUST NOT release the optimistic-action guard. The earlier
 * implementation reset on any change to the game-ref, which let
 * routine background updates "complete" the pending click and re-
 * enable the option buttons WITHOUT the advance actually landing.
 * Players reported this as "the dialog reset to the last choice"
 * after some delay.
 *
 * The fix: the guard releases ONLY when (a) ``current_node_id``
 * changes from the value at click time, OR (b) an ``s2c/error``
 * lands. This file exercises both paths.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render } from "@testing-library/react";

import { useLucidiumStore, type Game } from "../../src/state/store";
import { gameFixture, settingsFixture } from "../fixtures";

// Fake WS client: lets the test fire arbitrary events into the
// listeners, simulating backend pushes without real sockets.
const wsListeners = new Map<string, Set<(payload: unknown) => void>>();
const fakeClient = {
  on: (type: string, listener: (p: unknown) => void) => {
    let set = wsListeners.get(type);
    if (!set) {
      set = new Set();
      wsListeners.set(type, set);
    }
    set.add(listener);
    return () => set?.delete(listener);
  },
};

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => fakeClient),
}));

import { send } from "../../src/app/client";
import { InteractionPanel } from "../../src/screens/MainView/InteractionPanel";

function fireWsEvent(type: string, payload: unknown): void {
  const listeners = wsListeners.get(type);
  if (!listeners) return;
  for (const listener of listeners) listener(payload);
}

function makeGameAt(nodeId: string, options: { id: string; text: string }[]): Game {
  return gameFixture({
    current_node_id: nodeId,
    characters: {},
    dialog_tree: {
      nodes: {
        [nodeId]: {
          id: nodeId,
          parent_id: null,
          chosen_option_id: null,
          speaker_id: null,
          text: "Two paths.",
          options,
          entering_character_ids: [],
          leaving_character_ids: [],
          new_characters: [],
          location_id: null,
          location_prompt: null,
          location_lighting: "",
          character_changes: [],
          state: "committed",
          premise_hash: "h",
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
      root_id: nodeId,
      committed_path: [nodeId],
    },
    environments: {},
    on_stage: [],
  });
}

afterEach(() => {
  cleanup();
  wsListeners.clear();
});

describe("useOptimisticAction guard release semantics", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
  });

  it("does NOT release the guard on background state/full while the same node is current", () => {
    useLucidiumStore.setState({
      game: makeGameAt("n1", [
        { id: "o1", text: "Walk left." },
        { id: "o2", text: "Walk right." },
      ]),
      settings: settingsFixture({ typewriter_speed_chars_per_sec: 1000 }),
      hasSave: true,
      status: "open",
    });

    const { container } = render(<InteractionPanel />);
    // Click the first option — guard locks, send fires.
    const left = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent === "Walk left.",
    ) as HTMLButtonElement;
    expect(left).toBeDefined();
    act(() => {
      fireEvent.click(left);
    });
    expect(send).toHaveBeenCalledTimes(1);
    expect(left.disabled).toBe(true);

    // Background event lands: a stale state/full from an asset task
    // arrives. It updates the SAME node id (no advance happened).
    // The earlier impl released on game-ref change here — incorrect.
    act(() => {
      useLucidiumStore.setState({
        game: makeGameAt("n1", [
          { id: "o1", text: "Walk left." },
          { id: "o2", text: "Walk right." },
        ]),
      });
    });

    // Guard must STILL be locked — the advance hasn't completed.
    const stillLeft = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent === "Walk left.",
    ) as HTMLButtonElement;
    expect(stillLeft.disabled).toBe(true);
  });

  it("releases the guard when current_node_id actually changes", () => {
    useLucidiumStore.setState({
      game: makeGameAt("n1", [{ id: "o1", text: "Go." }]),
      settings: settingsFixture({ typewriter_speed_chars_per_sec: 1000 }),
      hasSave: true,
      status: "open",
    });

    const { container } = render(<InteractionPanel />);
    const go = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent === "Go.",
    ) as HTMLButtonElement;
    act(() => {
      fireEvent.click(go);
    });
    expect(go.disabled).toBe(true);

    // Real advance: backend lands a new node.
    act(() => {
      useLucidiumStore.setState({
        game: makeGameAt("n2", [{ id: "o3", text: "Continue." }]),
      });
    });

    // The first option button is gone (its node is no longer
    // current); the new button's pending should be false.
    const cont = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent === "Continue.",
    ) as HTMLButtonElement;
    expect(cont.disabled).toBe(false);
  });

  it("releases the guard when s2c/error lands (failed advance can be retried)", () => {
    useLucidiumStore.setState({
      game: makeGameAt("n1", [{ id: "o1", text: "Try." }]),
      settings: settingsFixture({ typewriter_speed_chars_per_sec: 1000 }),
      hasSave: true,
      status: "open",
    });

    const { container } = render(<InteractionPanel />);
    const tryBtn = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent === "Try.",
    ) as HTMLButtonElement;
    act(() => {
      fireEvent.click(tryBtn);
    });
    expect(tryBtn.disabled).toBe(true);

    // Backend rejected the command — release the guard so the
    // player can retry instead of staring at a permanent disable.
    act(() => {
      fireWsEvent("s2c/error", {
        code: "provider_unreachable",
        message: "LLM provider failed",
        recoverable: true,
      });
    });

    const after = Array.from(container.querySelectorAll("button")).find(
      (b) => b.textContent === "Try.",
    ) as HTMLButtonElement;
    expect(after.disabled).toBe(false);
  });
});
