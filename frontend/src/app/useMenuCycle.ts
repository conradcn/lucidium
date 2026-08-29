import { useEffect, useMemo, useRef, useState } from "react";

import { MENU_PAIRS } from "./menuManifest.generated";

export const PAIR_DURATION_MS = 10_000;
// Both halves of the carousel now tick on the SAME frame because
// the four-phase transition orchestrator (``useMenuTransition``)
// drives the visible animation order via CSS ``animation-delay`` on
// each layer. The earlier 1 s "guide leads bg" lag was the orchestrator
// itself in CSS form; with the React-side hook scheduling the phases,
// the layers can re-key at the same moment and let CSS run them in
// the right order.
export const GUIDE_LEAD_MS = 0;

export interface CarouselSlot {
  // Two-slot ring buffer. ``current`` is the slot whose mask is
  // animating from 0 → big right now (the new image growing in over
  // the previous one). ``pair`` holds the PAIR INDEX (into MENU_PAIRS)
  // at each slot — NOT the position within the shuffled order.
  current: 0 | 1;
  pair: [number, number];
  // Bumps every tick so React re-keys the current layer and the CSS
  // animation re-fires — even if the same slot index hits twice.
  bump: number;
  // Position within the per-session shuffled order. Internal — the
  // pair index for the rendered image is derived as
  // ``order[position % order.length]``.
  position: number;
}

export interface MenuCycleState {
  bg: CarouselSlot;
  guide: CarouselSlot;
}

function nextSlot(prev: CarouselSlot, order: number[]): CarouselSlot {
  const otherSlot: 0 | 1 = prev.current === 0 ? 1 : 0;
  const newPosition = (prev.position + 1) % order.length;
  const newIndex = order[newPosition]!;
  const newPair: [number, number] = [...prev.pair] as [number, number];
  newPair[otherSlot] = newIndex;
  return {
    current: otherSlot,
    pair: newPair,
    bump: prev.bump + 1,
    position: newPosition,
  };
}

function initialState(order: number[]): MenuCycleState {
  const first = order[0] ?? 0;
  return {
    bg: { current: 0, pair: [first, first], bump: 0, position: 0 },
    guide: { current: 0, pair: [first, first], bump: 0, position: 0 },
  };
}

/**
 * Fisher-Yates shuffle. Returns a new array; doesn't mutate input.
 * Pure ``Math.random`` — the order is non-reproducible across
 * sessions, which is the point: every session opens on a different
 * pair instead of always landing on the first manifest entry.
 */
export function shuffledIndices(total: number): number[] {
  const out = Array.from({ length: total }, (_, i) => i);
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}

/**
 * Drive the menu carousel: every PAIR_DURATION_MS the dream guide
 * ticks first; GUIDE_LEAD_MS later the background ticks. Two-slot
 * ring buffers so the previous image stays in the DOM under the
 * new one (used by the radial-reveal masks in CSS).
 *
 * The walk order is RANDOM per session — a Fisher-Yates permutation
 * of MENU_PAIRS' indices, computed once on mount and held stable
 * for the rest of the run. Without this every session opens on the
 * same first pair (e.g. ``harbor-noir``); with it the player sees a
 * different starting pair every time and works through every pair
 * exactly once before any wrap.
 *
 * The state lives at the App level so the carousel keeps its place
 * across phase transitions (e.g. start → interview → start).
 */
export function useMenuCycle(): MenuCycleState {
  // ``useMemo`` with an empty dep list = computed once per mount;
  // stable for the lifetime of the App component.
  const order = useMemo(() => shuffledIndices(MENU_PAIRS.length), []);
  const [state, setState] = useState<MenuCycleState>(() => initialState(order));
  const bgTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (order.length < 2) return;
    let cancelled = false;
    const tick = (): void => {
      if (cancelled) return;
      // Bump bg and guide on the same frame; CSS animation-delay
      // sequences phase A (guide wipe-down) → phase B (bg radial
      // reveal) → phase C (guide wipe-up).
      setState((prev) => ({
        guide: nextSlot(prev.guide, order),
        bg: nextSlot(prev.bg, order),
      }));
      if (GUIDE_LEAD_MS > 0) {
        // Legacy lag — kept reachable so a configuration change can
        // re-introduce the older overlapping behaviour without
        // touching this hook again.
        bgTimerRef.current = setTimeout(() => {
          if (cancelled) return;
          setState((prev) => ({ ...prev, bg: nextSlot(prev.bg, order) }));
        }, GUIDE_LEAD_MS);
      }
    };
    const interval = setInterval(tick, PAIR_DURATION_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
      if (bgTimerRef.current !== null) clearTimeout(bgTimerRef.current);
    };
  }, [order]);

  return state;
}
