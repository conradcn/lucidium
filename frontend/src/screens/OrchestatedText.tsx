import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";

interface Props {
  text: string;
  styleCurrent: CSSProperties;
  stylePrevious?: CSSProperties | undefined;
  // Bumps independently — the parent transition orchestrator ticks
  // ``eraseBump`` at phase A start and ``rewriteBump`` at phase C
  // start, so the title's backspace animation runs while the guide
  // wipes down, then waits through phase B (background swap) before
  // re-typing in the new style alongside the guide wiping back up.
  eraseBump: number;
  rewriteBump: number;
  stepMs?: number;
}

interface AnimState {
  visible: number;
  style: CSSProperties;
}

/**
 * Like ``StaggeredText`` but split into two independently-triggered
 * phases. The earlier component coupled backspace and rewrite into a
 * single ``bumpKey`` change, which fired both back-to-back. The
 * orchestrated menu transition needs the empty moment to LINGER —
 * phase A backspaces, phases B + the radial bg wipe run while the
 * title is empty, then phase C rewrites in the new style.
 *
 * Initial state has the full text visible in the current style so a
 * fresh mount lands on the correct visual without any animation.
 */
export function OrchestatedText({
  text,
  styleCurrent,
  stylePrevious,
  eraseBump,
  rewriteBump,
  stepMs = 70,
}: Props): JSX.Element {
  const [state, setState] = useState<AnimState>({
    visible: text.length,
    style: styleCurrent,
  });

  const lastEraseRef = useRef<number>(eraseBump);
  const lastRewriteRef = useRef<number>(rewriteBump);
  const visibleRef = useRef<number>(state.visible);
  // Synced in an effect rather than during render: refs must not be
  // written while rendering. Declared BEFORE the animation effects
  // below, so on the commit where a bump changes the ref is already
  // current by the time an animation reads it — identical timing to
  // the old render-phase write.
  useEffect(() => {
    visibleRef.current = state.visible;
  }, [state.visible]);

  useEffect(() => {
    if (lastEraseRef.current === eraseBump) return;
    lastEraseRef.current = eraseBump;

    let cancelled = false;
    let pendingTimeout: ReturnType<typeof setTimeout> | null = null;
    // Each ``sleep`` registers its timeout id so the unmount cleanup
    // can clear it immediately, instead of letting the timeout fire
    // and the awaiter's downstream ``if (cancelled) return`` carry the
    // bail. The old shape (``if (cancelled) clearTimeout(id)`` checked
    // only at sleep() CALL time) left in-flight timeouts running for
    // the full ``stepMs`` after unmount, each holding a closure over
    // the component instance — minor on its own, but the menu carousel
    // re-triggers this chain every ~10 s, so the slow accumulation
    // shows up as "main menu memory grows" over a long sit.
    const sleep = (ms: number): Promise<void> =>
      new Promise((resolve) => {
        if (cancelled) { resolve(); return; }
        pendingTimeout = setTimeout(() => {
          pendingTimeout = null;
          resolve();
        }, ms);
      });

    const run = async (): Promise<void> => {
      // Snap to the previous style on the first tick — the
      // disappearing letters carry the old pair's font/colour so the
      // backspace reads as the OLD treatment vanishing rather than
      // the new one shrinking.
      const eraseStyle = stylePrevious ?? styleCurrent;
      setState({ visible: visibleRef.current, style: eraseStyle });
      const startVisible = visibleRef.current;
      for (let v = startVisible - 1; v >= 0; v--) {
        await sleep(stepMs);
        if (cancelled) return;
        setState({ visible: v, style: eraseStyle });
      }
      // At the empty moment, swap to the NEW style. The span carries
      // ``--menu-title-shade`` via its inline style; the text-shadow
      // / stroke / drop-shadow CSS rules (scoped to .start-screen__title
      // .staggered) read the variable at use-site, so the title's glow
      // colour silently updates the moment the letters are gone — not
      // earlier when the bump fired (which would crossfade the glow
      // under the still-visible old letters), and not later at phase C
      // (which would crossfade the glow under the freshly-typed new
      // letters).
      setState({ visible: 0, style: styleCurrent });
    };
    void run();
    return () => {
      cancelled = true;
      if (pendingTimeout !== null) {
        clearTimeout(pendingTimeout);
        pendingTimeout = null;
      }
    };
    // eraseBump is the only legitimate trigger — listing styles
    // would re-fire on every parent render with fresh inline-style
    // refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [eraseBump]);

  useEffect(() => {
    if (lastRewriteRef.current === rewriteBump) return;
    lastRewriteRef.current = rewriteBump;

    let cancelled = false;
    let pendingTimeout: ReturnType<typeof setTimeout> | null = null;
    const sleep = (ms: number): Promise<void> =>
      new Promise((resolve) => {
        if (cancelled) { resolve(); return; }
        pendingTimeout = setTimeout(() => {
          pendingTimeout = null;
          resolve();
        }, ms);
      });

    const run = async (): Promise<void> => {
      // Empty moment: snap to the new style with visible=0 BEFORE
      // the first letter appears so phase C's first letter is
      // already the new typography.
      setState({ visible: 0, style: styleCurrent });
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rewriteBump]);

  return (
    <span className="staggered" aria-label={text} style={state.style}>
      {text.slice(0, state.visible)}
    </span>
  );
}
