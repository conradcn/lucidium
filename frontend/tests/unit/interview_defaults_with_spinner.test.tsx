/**
 * The character-description and name interview steps used to
 * block the player on a "Generating options…" splash while the
 * LLM round-trip ran. They now render with built-in defaults
 * the moment the player arrives + a small spinner in the
 * title — picking a default is a valid answer, OR the player
 * can wait a moment for tailored suggestions to swap in.
 *
 * Pin both halves: defaults render immediately, and the
 * spinner badge is visible while defaults are showing. Once
 * real LLM options arrive (snapshot fields populate), the
 * spinner disappears.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { render, cleanup } from "@testing-library/react";

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({ on: () => () => undefined })),
}));

import { CharacterStep } from "../../src/screens/NewGameInterview/CharacterStep";
import { NameStep } from "../../src/screens/NewGameInterview/NameStep";

afterEach(() => cleanup());

describe("Interview defaults + spinner", () => {
  it("CharacterStep with loading=true shows a spinner badge", () => {
    const { container } = render(
      <CharacterStep
        options={["wry archivist", "retired bounty hunter"]}
        loading
        onAnswer={() => undefined}
      />,
    );
    expect(
      container.querySelector("[data-testid='interview-options-loading']"),
    ).not.toBeNull();
    // Defaults still rendered as buttons.
    const buttonTexts = Array.from(container.querySelectorAll("button"))
      .map((b) => b.textContent?.trim() ?? "");
    expect(buttonTexts).toContain("wry archivist");
    expect(buttonTexts).toContain("retired bounty hunter");
  });

  it("CharacterStep with loading=false has NO spinner", () => {
    const { container } = render(
      <CharacterStep
        options={["wry archivist", "retired bounty hunter"]}
        loading={false}
        onAnswer={() => undefined}
      />,
    );
    expect(
      container.querySelector("[data-testid='interview-options-loading']"),
    ).toBeNull();
  });

  it("NameStep with loading=true shows a spinner badge", () => {
    const { container } = render(
      <NameStep
        options={["Iris Vale", "Vance Reed"]}
        loading
        onAnswer={() => undefined}
      />,
    );
    expect(
      container.querySelector("[data-testid='interview-options-loading']"),
    ).not.toBeNull();
  });

  it("loading=true shows a hint about the defaults being placeholders", () => {
    const { container } = render(
      <CharacterStep
        options={["x", "y"]}
        loading
        onAnswer={() => undefined}
      />,
    );
    // The shell renders a "Showing defaults while the engine
    // thinks" subtitle so the player understands the buttons
    // are placeholders.
    expect(container.textContent).toMatch(/defaults while the engine thinks/i);
  });
});
