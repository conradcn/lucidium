/**
 * InterviewStepShell keyboard + multiline behaviour:
 *  - Enter alone submits the free-text answer.
 *  - Shift+Enter inserts a newline (does NOT submit).
 *  - The free-text field is a textarea so multi-line answers fit.
 *  - The shell carries the always-white interview-step-content class.
 */

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render } from "@testing-library/react";

import { InterviewStepShell } from "../../src/screens/NewGameInterview/InterviewStepShell";

describe("InterviewStepShell — keyboard and multiline", () => {
  it("submits the free-text answer when the player presses Enter", () => {
    const onAnswer = vi.fn();
    const { getByTestId } = render(
      <InterviewStepShell
        title="Setting"
        options={["A harbor at dawn"]}
        onAnswer={onAnswer}
      />,
    );
    const ta = getByTestId("interview-free-text") as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: "stone keep on a hilltop" } });
    fireEvent.keyDown(ta, { key: "Enter", shiftKey: false });
    expect(onAnswer).toHaveBeenCalledWith("stone keep on a hilltop", true);
  });

  it("does NOT submit on Shift+Enter (newline insertion)", () => {
    const onAnswer = vi.fn();
    const { getByTestId } = render(
      <InterviewStepShell
        title="Setting"
        options={["A"]}
        onAnswer={onAnswer}
      />,
    );
    const ta = getByTestId("interview-free-text") as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: "line one" } });
    fireEvent.keyDown(ta, { key: "Enter", shiftKey: true });
    expect(onAnswer).not.toHaveBeenCalled();
  });

  it("does NOT submit empty / whitespace-only Enter presses", () => {
    const onAnswer = vi.fn();
    const { getByTestId } = render(
      <InterviewStepShell
        title="Setting"
        options={["A"]}
        onAnswer={onAnswer}
      />,
    );
    const ta = getByTestId("interview-free-text") as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: "   " } });
    fireEvent.keyDown(ta, { key: "Enter", shiftKey: false });
    expect(onAnswer).not.toHaveBeenCalled();
  });

  it("renders the free-text field as a textarea (multiline-capable)", () => {
    const { getByTestId } = render(
      <InterviewStepShell
        title="Setting"
        options={["A"]}
        onAnswer={vi.fn()}
      />,
    );
    const ta = getByTestId("interview-free-text");
    expect(ta.tagName.toLowerCase()).toBe("textarea");
  });

  it("attaches the interview-step-content class for always-white text styling", () => {
    const { container } = render(
      <InterviewStepShell
        title="Setting"
        options={["A"]}
        onAnswer={vi.fn()}
      />,
    );
    const content = container.querySelector(".interview-step-content");
    expect(content).toBeTruthy();
  });

  it("clicking an option button still submits with isFreeText=false", () => {
    const onAnswer = vi.fn();
    const { getByText } = render(
      <InterviewStepShell
        title="Setting"
        options={["A harbor"]}
        onAnswer={onAnswer}
      />,
    );
    fireEvent.click(getByText("A harbor"));
    expect(onAnswer).toHaveBeenCalledWith("A harbor", false);
  });
});
