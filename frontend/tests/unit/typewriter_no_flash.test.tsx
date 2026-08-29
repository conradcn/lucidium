/**
 * Regression: when the dialog node transitions to a new beat, the
 * typewriter must not flash the new beat's full text for a frame
 * before starting the per-character animation. The earlier impl
 * reset ``shown`` inside useEffect, which fires AFTER render — so the
 * first frame after a text-prop change would render
 * ``newText.slice(0, oldShown)`` and expose the entire incoming line.
 *
 * The fix: slice from a ref'd ``committedText`` so the render between
 * the prop change and the reset effect shows the OLD text for one
 * extra frame instead.
 */

import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, waitFor } from "@testing-library/react";

import { Typewriter } from "../../src/screens/MainView/Typewriter";

afterEach(() => cleanup());

describe("Typewriter no-flash on text change", () => {
  it("never renders the incoming new text on the frame the prop changes", async () => {
    const { rerender, container } = render(
      <Typewriter text="alpha" speedCharsPerSec={5000} />,
    );
    await waitFor(() => expect(container.textContent).toBe("alpha"));

    // Capture content immediately after the prop swap — this is the
    // "first frame" the user would see. Without the fix, this would
    // contain "beta" because the old impl rendered
    // newText.slice(0, oldShown=5) = "beta-".
    rerender(<Typewriter text="beta-something" speedCharsPerSec={5000} />);
    expect(container.textContent ?? "").not.toContain("beta");

    // The typewriter eventually catches up to the new text in full.
    await waitFor(
      () => expect(container.textContent).toBe("beta-something"),
      { timeout: 1000 },
    );
  });

  it("rapid beat swaps still finish each beat at some render", async () => {
    const beats = [
      "first beat lands.",
      "second beat lands.",
      "third beat lands.",
      "fourth beat lands.",
      "fifth and final beat.",
    ];
    const { rerender, container } = render(
      <Typewriter text={beats[0]!} speedCharsPerSec={5000} />,
    );
    for (const beat of beats) {
      rerender(<Typewriter text={beat} speedCharsPerSec={5000} />);
      await waitFor(
        () => expect(container.textContent).toBe(beat),
        { timeout: 1500 },
      );
    }
  });

  it("a swap mid-typing finishes the new text completely", async () => {
    const { rerender, container } = render(
      <Typewriter text="aaaaaaaaaa" speedCharsPerSec={50} />,
    );
    // Wait long enough for ~3 chars then swap.
    await new Promise((r) => setTimeout(r, 60));
    rerender(<Typewriter text="bbbbbbbbbb" speedCharsPerSec={5000} />);
    await waitFor(
      () => expect(container.textContent).toBe("bbbbbbbbbb"),
      { timeout: 2000 },
    );
  });
});
