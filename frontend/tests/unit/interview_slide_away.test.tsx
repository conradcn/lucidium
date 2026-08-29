/**
 * InterviewStepShell — when ``options`` change (the LLM-driven
 * personalised set arrives and replaces the hardcoded defaults),
 * the OLD options briefly slide out before the NEW options slide
 * in. Without this, the player can be bait-and-switched: they
 * click toward an option that vanishes mid-click.
 *
 * Pin the React behaviour: a ``--leaving`` class appears on the
 * grid for the hold window, then the grid re-mounts under a new
 * key with ``--entering`` so the slide-in animation runs.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, waitFor } from "@testing-library/react";

import { InterviewStepShell } from "../../src/screens/NewGameInterview/InterviewStepShell";

afterEach(() => cleanup());

beforeEach(() => {
  vi.useRealTimers();
});

describe("InterviewStepShell — slide-away on options change", () => {
  it("renders the initial options without a leaving class", () => {
    const { container } = render(
      <InterviewStepShell
        title="Setting"
        options={["A harbor", "A ruined castle"]}
        onAnswer={vi.fn()}
      />,
    );
    const grid = container.querySelector(".option-grid");
    expect(grid).toBeTruthy();
    expect(grid?.getAttribute("data-leaving")).toBe("false");
    expect(grid?.classList.contains("option-grid--entering")).toBe(true);
  });

  it("flips to leaving=true when options change, then back to entering", async () => {
    const { container, rerender } = render(
      <InterviewStepShell
        title="Setting"
        options={["A harbor"]}
        onAnswer={vi.fn()}
      />,
    );
    rerender(
      <InterviewStepShell
        title="Setting"
        options={["A different setting"]}
        onAnswer={vi.fn()}
      />,
    );
    // Immediately after the props change, the old grid is in its
    // leaving state. The OLD options content is still in the DOM —
    // the React-side hold prevents the abrupt swap.
    const leavingGrid = container.querySelector(".option-grid");
    expect(leavingGrid?.getAttribute("data-leaving")).toBe("true");
    expect(leavingGrid?.textContent).toContain("A harbor");
    // After the leave window, the NEW options take over with
    // entering animation.
    await waitFor(
      () => {
        const grid = container.querySelector(".option-grid");
        expect(grid?.getAttribute("data-leaving")).toBe("false");
        expect(grid?.textContent).toContain("A different setting");
      },
      { timeout: 1000 },
    );
  });

  it("does NOT flip to leaving when options stay identical across renders", () => {
    const { container, rerender } = render(
      <InterviewStepShell
        title="Setting"
        options={["A", "B"]}
        onAnswer={vi.fn()}
      />,
    );
    // Re-render with the SAME options array reference content.
    rerender(
      <InterviewStepShell
        title="Setting"
        options={["A", "B"]}
        onAnswer={vi.fn()}
      />,
    );
    const grid = container.querySelector(".option-grid");
    // Same contents → no animation kick.
    expect(grid?.getAttribute("data-leaving")).toBe("false");
  });

  it("disables option buttons during the leaving window", () => {
    const { container, rerender } = render(
      <InterviewStepShell
        title="Setting"
        options={["A"]}
        onAnswer={vi.fn()}
      />,
    );
    rerender(
      <InterviewStepShell
        title="Setting"
        options={["B"]}
        onAnswer={vi.fn()}
      />,
    );
    // While leaving, every option button is disabled so the
    // player can't click on a button that's about to vanish.
    const buttons = container.querySelectorAll(".option-grid button");
    buttons.forEach((b) => {
      expect((b as HTMLButtonElement).disabled).toBe(true);
    });
  });
});
