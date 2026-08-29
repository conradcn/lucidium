/**
 * The Continue button at the end of a beat batch shows a spinner
 * (not the static triangle) when no pre-rendered next beat sits
 * ready. The button stays clickable EITHER WAY — the spinner is
 * just a hint that the engine is generating. Without this hint
 * the player saw a triangle that "didn't respond" because the
 * LLM round-trip was in flight.
 *
 * Pin the wiring so a future refactor that drops the spinner
 * (or always renders the triangle) trips this test.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render } from "@testing-library/react";

import { useLucidiumStore } from "../../src/state/store";
import { InteractionPanel } from "../../src/screens/MainView/InteractionPanel";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({ on: () => () => undefined })),
}));

afterEach(() => cleanup());

interface NodeShape {
  id: string;
  parent_id: string | null;
  chosen_option_id: string | null;
  options: { id: string; text: string }[];
  state?: string;
  text?: string;
  speaker_id?: string | null;
}

function setupGame(opts: {
  current: NodeShape;
  others?: NodeShape[];
}): void {
  const nodes: Record<string, NodeShape> = {
    [opts.current.id]: opts.current,
  };
  for (const n of opts.others ?? []) {
    nodes[n.id] = n;
  }
  useLucidiumStore.setState({
    settings: { typewriter_speed_chars_per_sec: 60 } as never,
    game: {
      id: "g1",
      schema_version: 1,
      current_node_id: opts.current.id,
      world: {} as never,
      characters: {},
      dialog_tree: {
        nodes,
        committed_path: [opts.current.id],
        root_id: opts.current.id,
      },
      environments: {},
      on_stage: [],
    } as never,
  });
}

function makeNode(over: Partial<NodeShape> = {}): NodeShape {
  return {
    id: "n1",
    parent_id: null,
    chosen_option_id: null,
    options: [],
    state: "committed",
    text: "x",
    speaker_id: null,
    ...over,
  };
}

describe("Continue glyph — spinner vs ready triangle", () => {
  it("renders a spinner when there's no pre-rendered next beat", () => {
    // Current node has no options AND no continue child in the
    // tree. The player has hit the end of a batch; the engine is
    // generating the next beat; show the spinner so the player
    // knows we're working.
    setupGame({
      current: makeNode({ id: "n1", options: [] }),
    });
    const { container, queryByTestId } = render(<InteractionPanel />);
    // Spinner present.
    expect(queryByTestId("continue-glyph-spinner")).toBeTruthy();
    // No triangle.
    expect(container.querySelector(".continue-glyph__triangle")).toBeNull();
    // Button still clickable (NOT disabled by the loading state).
    const btn = container.querySelector("button.continue-glyph") as HTMLButtonElement;
    expect(btn.disabled).toBe(false);
  });

  it("renders the triangle when a pre-rendered child sits ready", () => {
    // Pre-rendered child exists: parent_id matches current,
    // chosen_option_id is null (it's a continuation, not a
    // chosen branch), state isn't invalidated.
    setupGame({
      current: makeNode({ id: "n1", options: [] }),
      others: [
        {
          id: "n2",
          parent_id: "n1",
          chosen_option_id: null,
          options: [],
          state: "ready",
        },
      ],
    });
    const { container, queryByTestId } = render(<InteractionPanel />);
    // Triangle present.
    expect(container.querySelector(".continue-glyph__triangle")).toBeTruthy();
    // No spinner.
    expect(queryByTestId("continue-glyph-spinner")).toBeNull();
  });

  it("renders neither glyph when the node has option buttons", () => {
    setupGame({
      current: makeNode({
        id: "n1",
        options: [{ id: "o1", text: "go north" }],
      }),
    });
    const { container, queryByTestId } = render(<InteractionPanel />);
    // Option buttons replace the continue glyph entirely.
    expect(container.querySelector(".continue-glyph")).toBeNull();
    expect(queryByTestId("continue-glyph-spinner")).toBeNull();
  });
});
