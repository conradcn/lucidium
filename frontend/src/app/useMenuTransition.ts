import { useEffect, useRef, useState } from "react";

/**
 * Four-phase orchestrator for the main-menu pair transition.
 *
 * Timeline (relative to a ``sourceBump`` tick):
 *
 *   t=0     : phase A — title backspace begins (560 ms; 8 letters × 70 ms
 *             stepMs in OrchestatedText).
 *   t=560   : title is empty; guide radial erode begins (800 ms). Title
 *             deletion completes BEFORE the character starts fading so
 *             the eye lands on each step in sequence.
 *   t=1360  : guide finishes eroding; phase B starts — bg radial wipe
 *             begins (1500 ms duration). Menu accent begins crossfading
 *             to the new pair colour.
 *   t=2110  : guide upward bloom begins (halfway through the bg radial),
 *             so the new dream guide rises into view as the background
 *             is still settling. Duration 800 ms.
 *   t=2860  : bg radial finishes.
 *   t=2910  : guide bloom finishes (50 ms overhang). Phase C — title
 *             rewrites in the new style (800 ms).
 *   t=3710  : phase D — per-button backspace+rewrite cascade, each
 *             button staggered ``BUTTON_STAGGER_MS`` behind the previous.
 *
 * Each consumer subscribes to its own bump:
 *   - title's ``OrchestatedText`` reads ``titleEraseBump`` (phase A) and
 *     ``titleRewriteBump`` (phase C — fires AFTER both image transitions
 *     finish, per the design that wants the typing to land on a settled
 *     scene).
 *   - the menu carousel's CSS animations key on the React element's
 *     mount via the cycle bumps and ``animation-delay`` lining up with
 *     ``CHARACTER_FADE_DELAY_MS`` (guide erode) / phase B / mid-B
 *     (guide bloom).
 *   - ``showNewAccent`` flips at phase B start so ``--menu-accent``
 *     interpolates from the old pair colour to the new one.
 *   - per-button ``buttonStaggerBump`` ticks at phase D start.
 */

// Title erase finishes at this offset (8 letters × 70 ms stepMs in
// OrchestatedText). The character fadeout's CSS ``animation-delay``
// matches this so the guide doesn't start dissolving until the title
// has fully cleared.
export const TITLE_ERASE_MS = 560;
export const CHARACTER_FADE_DELAY_MS = TITLE_ERASE_MS;
export const GUIDE_WIPE_MS = 800;
// Phase A now runs the title erase AND THEN the character erode in
// sequence, not in parallel.
export const PHASE_A_MS = CHARACTER_FADE_DELAY_MS + GUIDE_WIPE_MS; // 1360
// Phase B holds the background radial AND the guide bloom. The guide
// starts at HALFWAY through phase B and runs an extra
// ``GUIDE_OVERHANG_MS`` past the bg's finish so the title rewrite
// (phase C) doesn't begin until BOTH images have settled.
export const PHASE_B_BG_MS = 1500;
export const GUIDE_OVERHANG_MS =
  PHASE_B_BG_MS / 2 + GUIDE_WIPE_MS - PHASE_B_BG_MS; // 50 ms
export const PHASE_B_MS = PHASE_B_BG_MS + GUIDE_OVERHANG_MS; // 1550
export const PHASE_C_MS = 800;
export const BUTTON_STAGGER_MS = 100;

export type TransitionPhase = "idle" | "A" | "B" | "C" | "D";

export interface MenuTransition {
  phase: TransitionPhase;
  titleEraseBump: number;
  titleRewriteBump: number;
  buttonStaggerBump: number;
  showNewAccent: boolean;
}

export function useMenuTransition(sourceBump: number): MenuTransition {
  const [state, setState] = useState<MenuTransition>({
    phase: "idle",
    titleEraseBump: sourceBump,
    titleRewriteBump: sourceBump,
    buttonStaggerBump: sourceBump,
    showNewAccent: true,
  });
  const lastSource = useRef(sourceBump);

  useEffect(() => {
    if (lastSource.current === sourceBump) return;
    lastSource.current = sourceBump;

    // Phase A: title backspace fires NOW. The CSS-driven guide wipe
    // also fires on this tick — its trigger is the React key change
    // on the new "current" guide slot, which the carousel re-keyed
    // off ``cycle.guide.bump``.
    setState((s) => ({
      ...s,
      phase: "A",
      titleEraseBump: sourceBump,
      // Hold the OLD accent through phase A so the still-visible
      // button borders / hover colour don't pre-empt the bg swap.
      showNewAccent: false,
    }));

    const tB = setTimeout(() => {
      setState((s) => ({
        ...s,
        phase: "B",
        // Flip the accent: the menu wrap's ``--menu-accent``
        // interpolates over the existing CSS transition, painting
        // the buttons in the new pair's colour while the background
        // radially wipes to the new image.
        showNewAccent: true,
      }));
    }, PHASE_A_MS);

    const tC = setTimeout(() => {
      setState((s) => ({
        ...s,
        phase: "C",
        titleRewriteBump: sourceBump,
      }));
    }, PHASE_A_MS + PHASE_B_MS);

    const tD = setTimeout(() => {
      setState((s) => ({
        ...s,
        phase: "D",
        buttonStaggerBump: sourceBump,
      }));
    }, PHASE_A_MS + PHASE_B_MS + PHASE_C_MS);

    return () => {
      clearTimeout(tB);
      clearTimeout(tC);
      clearTimeout(tD);
    };
  }, [sourceBump]);

  return state;
}
