import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";

import { useLucidiumStore } from "../../src/state/store";
import { gameFixture, settingsFixture } from "../fixtures";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(),
}));

import { send } from "../../src/app/client";
import { InteractionPanel } from "../../src/screens/MainView/InteractionPanel";

describe("InteractionPanel", () => {
  beforeEach(() => {
    useLucidiumStore.setState({
      game: gameFixture({
        current_node_id: "n1",
        dialog_tree: {
          nodes: {
            n1: {
              id: "n1",
              text: "The harbor wakes slow.",
              options: [
                { id: "o1", text: "Walk to the archive." },
                { id: "o2", text: "Stay a moment." },
              ],
            },
          },
        },
      }),
      settings: settingsFixture({ typewriter_speed_chars_per_sec: 1000 }),
      hasSave: false,
      status: "open",
    });
    vi.mocked(send).mockClear();
  });
  afterEach(() => cleanup());

  it("renders option buttons and dispatches advance on click", () => {
    render(<InteractionPanel />);
    fireEvent.click(screen.getByText("Walk to the archive."));
    expect(send).toHaveBeenCalledWith("c2s/play/advance", { option_id: "o1" });
  });

  it("dispatches free-text submission", () => {
    render(<InteractionPanel />);
    const input = screen.getByPlaceholderText("Or do something else...");
    fireEvent.change(input, { target: { value: "Iris kneels by the door." } });
    fireEvent.click(screen.getByText("Submit"));
    expect(send).toHaveBeenCalledWith("c2s/play/free_text", { text: "Iris kneels by the door." });
  });

  it("falls back to a Continue glyph button when there are no options", () => {
    // The Continue control renders as an arrow glyph (when the
    // next beat is pre-generated and walkable) or a spinner (while
    // the LLM is drafting). The arrow case must dispatch advance;
    // the spinner case must be disabled.
    useLucidiumStore.setState({
      game: gameFixture({
        current_node_id: "n1",
        dialog_tree: {
          nodes: {
            n1: {
              id: "n1",
              text: "...",
              options: [],
              parent_id: null,
              chosen_option_id: null,
            },
            // Pre-generated continue child — flips the glyph to
            // the ready/arrow state.
            n2: {
              id: "n2",
              text: "next beat",
              options: [],
              parent_id: "n1",
              chosen_option_id: null,
            },
          },
        },
      }),
    });
    render(<InteractionPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(send).toHaveBeenCalledWith("c2s/play/advance", { option_id: null });
  });

  it("shows the continue arrow when no child exists yet, and stays clickable", () => {
    useLucidiumStore.setState({
      game: gameFixture({
        current_node_id: "n1",
        dialog_tree: {
          nodes: {
            n1: {
              id: "n1",
              text: "...",
              options: [],
              parent_id: null,
              chosen_option_id: null,
            },
          },
        },
      }),
    });
    render(<InteractionPanel />);
    // The triangle is the player's affordance — "you can advance".
    // The earlier disabled-spinner pattern displayed an in-flight
    // LLM call as if it were a hard block, even though the
    // click-anywhere-to-continue path was advancing through it.
    // ``data-state="loading"`` is still set so CSS could choose to
    // hint at the pre-generation status, but the button label is
    // always "Continue" and the button itself is enabled.
    const btn = screen.getByRole("button", {
      name: "Continue",
    }) as HTMLButtonElement;
    expect(btn.disabled).toBe(false);
    expect(btn.dataset.state).toBe("loading");
  });

  it("shows a speaker tag when the current node has a speaker_id", () => {
    useLucidiumStore.setState({
      game: gameFixture({
        current_node_id: "n1",
        characters: {
          c1: { id: "c1", name: "Hale" },
        },
        dialog_tree: {
          nodes: {
            n1: {
              id: "n1",
              text: "The harbor wakes slow.",
              speaker_id: "c1",
              options: [],
            },
          },
        },
      }),
      settings: settingsFixture({ typewriter_speed_chars_per_sec: 1000 }),
    });
    render(<InteractionPanel />);
    const tag = screen.getByTestId("speaker-tag");
    expect(tag.textContent).toBe("Hale");
  });

  it("renders no speaker tag for narration (speaker_id null)", () => {
    useLucidiumStore.setState({
      game: gameFixture({
        current_node_id: "n1",
        characters: {},
        dialog_tree: {
          nodes: {
            n1: {
              id: "n1",
              text: "The harbor wakes slow.",
              speaker_id: null,
              options: [],
            },
          },
        },
      }),
      settings: settingsFixture({ typewriter_speed_chars_per_sec: 1000 }),
    });
    render(<InteractionPanel />);
    expect(screen.queryByTestId("speaker-tag")).toBeNull();
  });
});
