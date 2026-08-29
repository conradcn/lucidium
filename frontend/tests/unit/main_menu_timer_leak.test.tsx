/**
 * Profiler-style regression test for the main-menu timer leak.
 *
 * The carousel ticks every 10 s; each tick fires the four-phase
 * ``useMenuTransition`` orchestrator which in turn drives
 * ``OrchestatedText`` (title) and ``StaggeredText`` (every menu
 * button). Both text components sleep between letter steps via
 * ``setTimeout``. The original ``sleep`` helper checked the
 * ``cancelled`` flag only at sleep() CALL time — so when a parent
 * unmount fired ``cancelled = true``, the in-flight ``setTimeout``
 * was NOT cleared. Across a long sit on the start screen those
 * stranded timeouts dribbled into the event loop every cycle,
 * each holding a closure over the unmounted component.
 *
 * vitest's ``vi.getTimerCount()`` exposes pending fake timers, so we
 * can profile the leak directly: bump the source repeatedly to
 * trigger many transitions, interrupt them mid-flight by
 * unmounting, and assert no timers remain. This test fails against
 * the old code path; the fix lands when the unmount cleanup clears
 * the in-flight timeout id.
 */

import type { JSX } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render } from "@testing-library/react";
import { useState } from "react";

import { OrchestatedText } from "../../src/screens/OrchestatedText";
import { StaggeredText } from "../../src/screens/StaggeredText";

describe("main-menu text components: timer cleanup", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("StaggeredText leaks no pending timers when unmounted mid-cycle", () => {
    const baseline = vi.getTimerCount();
    const { unmount, rerender } = render(
      <StaggeredText
        text="LUCIDIUM"
        styleCurrent={{}}
        stylePrevious={{}}
        bumpKey={0}
      />,
    );

    // Mid-cycle interruptions: bump the key several times without
    // letting any animation complete. Each bump starts a fresh
    // backspace+rewrite chain; under the old code path, the prior
    // chain's in-flight ``setTimeout`` would survive into the new
    // cycle and accumulate over time.
    for (let i = 1; i <= 5; i++) {
      rerender(
        <StaggeredText
          text="LUCIDIUM"
          styleCurrent={{}}
          stylePrevious={{}}
          bumpKey={i}
        />,
      );
      // Advance only PART of a step so we always interrupt a sleep
      // in-flight rather than letting it resolve.
      act(() => {
        vi.advanceTimersByTime(20);
      });
    }

    // Unmount mid-cycle — this is the moment the leak fires under
    // the old code. The cleanup MUST clear the pending timeout.
    unmount();

    // After unmount, advance a long way so any stranded timer that
    // wasn't cleared would fire and we'd see its callback's setState
    // attempt. With a proper cleanup the count returns to baseline.
    expect(vi.getTimerCount()).toBe(baseline);
  });

  it("OrchestatedText leaks no pending timers when unmounted mid-cycle", () => {
    const baseline = vi.getTimerCount();
    function Harness(): JSX.Element {
      const [erase, setErase] = useState(0);
      const [rewrite, setRewrite] = useState(0);
      return (
        <>
          <button
            data-testid="bump-erase"
            onClick={() => setErase((v) => v + 1)}
          />
          <button
            data-testid="bump-rewrite"
            onClick={() => setRewrite((v) => v + 1)}
          />
          <OrchestatedText
            text="LUCIDIUM"
            styleCurrent={{}}
            stylePrevious={{}}
            eraseBump={erase}
            rewriteBump={rewrite}
          />
        </>
      );
    }
    const { unmount, getByTestId } = render(<Harness />);

    // Cycle the erase / rewrite triggers a few times to fan multiple
    // in-flight sleep chains. Each act() runs synchronously; the
    // first triggers OrchestatedText's effect which queues the first
    // sleep timeout — it remains pending until vi.advanceTimers.
    for (let i = 0; i < 3; i++) {
      act(() => {
        getByTestId("bump-erase").click();
      });
      act(() => {
        vi.advanceTimersByTime(20);
      });
      act(() => {
        getByTestId("bump-rewrite").click();
      });
      act(() => {
        vi.advanceTimersByTime(20);
      });
    }

    unmount();

    expect(vi.getTimerCount()).toBe(baseline);
  });

  it("StaggeredText sustained cycling stays bounded (10× heavier load)", () => {
    // Stress shape: keep the component mounted, bump bumpKey rapidly
    // with mid-step advances, and confirm the pending-timer count
    // tracks AT MOST one in-flight sleep per active component — i.e.
    // bounded by the components, not by the number of cycles. Under
    // the old leak this count grew unboundedly across cycles.
    const baseline = vi.getTimerCount();
    const { rerender, unmount } = render(
      <StaggeredText
        text="LUCIDIUM"
        styleCurrent={{}}
        stylePrevious={{}}
        bumpKey={0}
      />,
    );

    let maxObserved = 0;
    for (let i = 1; i <= 50; i++) {
      rerender(
        <StaggeredText
          text="LUCIDIUM"
          styleCurrent={{}}
          stylePrevious={{}}
          bumpKey={i}
        />,
      );
      act(() => {
        vi.advanceTimersByTime(30);
      });
      const live = vi.getTimerCount() - baseline;
      if (live > maxObserved) maxObserved = live;
    }
    // Worst-case in-flight for one component is 1 sleep — the
    // backspace/rewrite chain is sequential per-letter. Loosen to
    // ≤ 2 to absorb React's transitional render windows. The old
    // leak path drove this monotonically up past 50.
    expect(maxObserved).toBeLessThanOrEqual(2);

    unmount();
    expect(vi.getTimerCount()).toBe(baseline);
  });
});
