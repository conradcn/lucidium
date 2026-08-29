/**
 * Menu cycle hook: bg and guide now tick on the same frame, every
 * PAIR_DURATION_MS. The earlier 1 s "guide leads bg" lag was the
 * transition orchestrator in CSS form; with ``useMenuTransition``
 * driving the four-phase animation in React, both layers re-key
 * together and CSS ``animation-delay`` sequences the visible order
 * (phase A guide wipe-down → phase B bg radial → phase C guide
 * wipe-up).
 *
 * Hook lives at App level so the cycle survives start ↔ interview
 * phase transitions; the StartScreen no longer owns the timer, so
 * the test targets the hook directly via renderHook.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";

vi.mock("../../src/app/menuManifest.generated", () => ({
  MENU_BASE: "/main-menu/",
  MENU_PAIRS: [
    { slug: "alpha", label: "Alpha" },
    { slug: "beta", label: "Beta" },
    { slug: "gamma", label: "Gamma" },
  ],
}));

import { useMenuCycle } from "../../src/app/useMenuCycle";

describe("useMenuCycle — main-menu carousel timing", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("starts with both stacks at position 0 of the shuffled order", () => {
    const { result } = renderHook(() => useMenuCycle());
    // Position is the index into the per-session shuffled order;
    // it always begins at 0 even though the *pair index* it maps to
    // is randomised per session.
    expect(result.current.bg.position).toBe(0);
    expect(result.current.guide.position).toBe(0);
    // Both stacks initialised to the same pair so the first
    // cross-fade has something to fade FROM.
    expect(result.current.bg.pair[result.current.bg.current]).toBe(
      result.current.guide.pair[result.current.guide.current],
    );
  });

  it("advances guide and background on the same tick, every PAIR_DURATION_MS", () => {
    const { result } = renderHook(() => useMenuCycle());

    // 10 s mark: both stacks tick together (position 0 → 1).
    act(() => {
      vi.advanceTimersByTime(10_000);
    });
    expect(result.current.guide.position).toBe(1);
    expect(result.current.bg.position).toBe(1);

    // Another full cycle.
    act(() => {
      vi.advanceTimersByTime(10_000);
    });
    expect(result.current.guide.position).toBe(2);
    expect(result.current.bg.position).toBe(2);
  });

  it("wraps around after the last pair", () => {
    const { result } = renderHook(() => useMenuCycle());
    // Walk three full guide ticks (10 s each) — three positions
    // ahead of position 0 in a 3-pair order means wrap back to 0.
    for (let i = 0; i < 3; i++) {
      act(() => {
        vi.advanceTimersByTime(10_000);
      });
    }
    expect(result.current.guide.position).toBe(0);
  });

  it("walks every pair exactly once before wrapping (no repeats)", () => {
    const { result } = renderHook(() => useMenuCycle());
    const seen = new Set<number>();
    seen.add(result.current.guide.pair[result.current.guide.current]!);
    for (let i = 0; i < 2; i++) {
      act(() => {
        vi.advanceTimersByTime(10_000);
      });
      seen.add(result.current.guide.pair[result.current.guide.current]!);
    }
    // Three ticks total, three distinct pairs visited from the
    // 3-pair mocked manifest — proves the shuffled order is a
    // permutation, not a sample-with-replacement.
    expect(seen.size).toBe(3);
  });
});
