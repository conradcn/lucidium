/**
 * The loading screen carries a short UX nudge — "Remember that
 * this is your dream. If you want something to be true, act as
 * though it is." — to help first-time players understand that
 * the free-text box can declare facts, not just describe
 * actions. Pin the wording so a future style refactor doesn't
 * silently drop the line.
 */

import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";

import { LoadingScreen } from "../../src/screens/LoadingScreen";

describe("LoadingScreen", () => {
  it("renders the dream-framing hint", () => {
    const { getByText } = render(<LoadingScreen />);
    expect(
      getByText(
        /Remember that this is your dream\. If you want something to be true, act as though it is\./i,
      ),
    ).toBeTruthy();
  });

  it("still renders the spinner alongside the hint", () => {
    const { container } = render(<LoadingScreen />);
    expect(container.querySelector(".loading-overlay__spinner")).not.toBeNull();
  });
});
