import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

import { useLucidiumStore } from "../../src/state/store";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(),
}));

import { StartScreen } from "../../src/screens/StartScreen";

describe("accessibility — StartScreen", () => {
  beforeEach(() => {
    useLucidiumStore.setState({ hasSave: true, status: "open", game: null, settings: null });
  });
  afterEach(() => cleanup());

  it("offers actionable buttons in tab order", () => {
    render(
      <StartScreen
        cycle={{
          bg: { current: 0, pair: [0, 0], bump: 0, position: 0 },
          guide: { current: 0, pair: [0, 0], bump: 0, position: 0 },
        }}
        onNewGame={() => undefined}
        onLoadGame={() => undefined}
        onSettings={() => undefined}
      />,
    );
    const buttons = screen.getAllByRole("button");
    // Each button's accessible name comes from its aria-label —
    // the visible text is rendered as TWO stacked spans (current +
    // previous font) for the cross-fade, but a11y trees collapse to
    // the explicit aria-label so screen readers don't double-speak.
    const names = buttons.map((b) => b.getAttribute("aria-label"));
    expect(names).toEqual([
      "Continue", "New Game", "Surprise Me", "Load Game", "Options", "Exit",
    ]);
  });

  it("hides the Continue button when no save exists", () => {
    useLucidiumStore.setState({ hasSave: false, status: "open", game: null, settings: null });
    render(
      <StartScreen
        cycle={{
          bg: { current: 0, pair: [0, 0], bump: 0, position: 0 },
          guide: { current: 0, pair: [0, 0], bump: 0, position: 0 },
        }}
        onNewGame={() => undefined}
        onLoadGame={() => undefined}
        onSettings={() => undefined}
      />,
    );
    const buttons = screen.getAllByRole("button");
    const names = buttons.map((b) => b.getAttribute("aria-label"));
    expect(names).not.toContain("Continue");
  });
});
