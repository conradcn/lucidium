import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";

interface Props {
  text: string;
  styleCurrent: CSSProperties;
  stylePrevious?: CSSProperties | undefined;
  // Bumps every cycle to trigger the backspace-then-rewrite
  // animation. The component is otherwise static.
  bumpKey: number;
  // Step interval per letter — same value drives the backspace
  // cadence (right-to-left) and the rewrite cadence (left-to-right).
  stepMs?: number;
}

interface AnimState {
  visible: number;
  style: CSSProperties;
}

/**
 * Render ``text`` inside a SINGLE element whose visible substring
 * grows and shrinks like a typewriter. On every ``bumpKey`` change:
 *
 *   1. Backspace right-to-left in the PREVIOUS pair's style — the
 *      element's text content drops one character per ``stepMs``.
 *   2. At the empty moment, swap to the CURRENT pair's style.
 *   3. Rewrite left-to-right in the CURRENT style.
 *
 * Why the single-element shape: the element keeps its own font /
 * color / text-shadow as one continuous treatment over whatever
 * substring is currently visible. The earlier per-letter layered
 * version gave each glyph its own shadow box, which clipped the
 * shadow at letter boundaries; this approach lets the shadow flow
 * across the whole visible text the way a normal CSS shadow does.
 */
export function StaggeredText({
  text,
  styleCurrent,
  stylePrevious,
  bumpKey,
  stepMs = 60,
}: Props): JSX.Element {
  const [state, setState] = useState<AnimState>({
    visible: text.length,
    style: styleCurrent,
  });

  // Track the previous bump so the effect only fires on a real
  // pair-cycle (not on every parent re-render).
  const lastBumpRef = useRef<number>(bumpKey);
  // Hold the current visible count outside React state so the
  // animation closure can read the fresh count without re-running.
  const visibleRef = useRef<number>(state.visible);
  // Synced in an effect rather than during render: refs must not be
  // written while rendering. This effect is declared BEFORE the
  // animation effect below, so on the commit where ``bumpKey``
  // changes the ref is already up to date by the time the animation
  // reads it — identical timing to the old render-phase write.
  useEffect(() => {
    visibleRef.current = state.visible;
  }, [state.visible]);

  useEffect(() => {
    if (lastBumpRef.current === bumpKey) return;
    lastBumpRef.current = bumpKey;

    let cancelled = false;
    let pendingTimeout: ReturnType<typeof setTimeout> | null = null;
    // Track the in-flight timeout so the cleanup can clear it
    // immediately. The old ``if (cancelled) clearTimeout(id)``
    // pattern only checked at sleep() CALL time and left the
    // timeout running after unmount — across the menu carousel's
    // ~10 s ticks this dribbles unreachable closures into the
    // event loop. Each MenuButton runs its own chain (~10 sleeps
    // per cycle × 6 buttons) so the dribble adds up over a long
    // sit on the start screen.
    const sleep = (ms: number): Promise<void> =>
      new Promise((resolve) => {
        if (cancelled) { resolve(); return; }
        pendingTimeout = setTimeout(() => {
          pendingTimeout = null;
          resolve();
        }, ms);
      });

    const run = async (): Promise<void> => {
      // PHASE 1 — backspace in the PREVIOUS pair's style.
      // Snap to the previous style on the FIRST tick so the
      // disappearing letters look right. If there's no previous
      // (first cycle ever), skip straight to phase 3.
      if (stylePrevious !== undefined) {
        setState({ visible: visibleRef.current, style: stylePrevious });
        const startVisible = visibleRef.current;
        for (let v = startVisible - 1; v >= 0; v--) {
          await sleep(stepMs);
          if (cancelled) return;
          setState({ visible: v, style: stylePrevious });
        }
      }

      // PHASE 2 — empty moment. THIS is where the style swaps;
      // there's no visible text right now so the change can't be
      // seen as a font/color jump.
      setState({ visible: 0, style: styleCurrent });

      // PHASE 3 — rewrite in the CURRENT pair's style.
      for (let v = 1; v <= text.length; v++) {
        await sleep(stepMs);
        if (cancelled) return;
        setState({ visible: v, style: styleCurrent });
      }
    };

    void run();
    return () => {
      cancelled = true;
      if (pendingTimeout !== null) {
        clearTimeout(pendingTimeout);
        pendingTimeout = null;
      }
    };
    // Only re-run on bumpKey: styleCurrent / stylePrevious are
    // fresh inline objects every render, so listing them would
    // restart the animation on every parent render. ``bumpKey`` is
    // the single source of truth for "the pair just changed."
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bumpKey]);

  return (
    <span className="staggered" aria-label={text} style={state.style}>
      {text.slice(0, state.visible)}
    </span>
  );
}
