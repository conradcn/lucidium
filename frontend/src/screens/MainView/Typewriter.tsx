import type { JSX } from "react";
import { useEffect, useState } from "react";

interface Props {
  text: string;
  speedCharsPerSec: number;
  // Bumping this counter forces the typewriter to jump to the full
  // text immediately (without advancing the dialog). Wired to the
  // click-anywhere-to-continue handler so a rapid-clicker sees every
  // beat at full length for at least one render before the next
  // beat swaps in. Without this, fast clicks land their advances
  // BEFORE the previous beat finished typing — the line was
  // committed by the backend but the player never read it.
  flushSignal?: number | undefined;
  onComplete?: () => void;
}

export function Typewriter({
  text,
  speedCharsPerSec,
  flushSignal,
  onComplete,
}: Props): JSX.Element {
  // ``committedText`` is the text the typewriter has *seen* — the
  // span only ever slices from this, never from the incoming
  // ``text`` prop. The earlier impl reset ``shown`` inside
  // useEffect; the render between the prop change and the effect
  // would render ``newText.slice(0, oldShown)`` and flash the new
  // line in full for one frame. Slicing from a ref'd committed copy
  // avoids that.
  //
  // It is held in state and re-synced during render (the "adjust
  // state when a prop changes" pattern from
  // https://react.dev/learn/you-might-not-need-an-effect) rather than
  // in a ref: React re-runs this component immediately with the new
  // text and ``shown`` back at 0, before anything is painted, which
  // gives the same no-flash guarantee the ref gave without reading a
  // ref during render.
  const [committed, setCommitted] = useState(text);
  const [shown, setShown] = useState(0);

  if (committed !== text) {
    setCommitted(text);
    setShown(0);
  }

  // Flush: jump to full text on demand. ``flushSignal`` is an
  // imperative one-shot from the click-anywhere handler — an external
  // event, not a value derivable from props — and the jump must land
  // after the render-phase reset above, so this genuinely is a
  // setState from an effect. The extra render is the point: the
  // player has asked to see the whole line NOW.
  useEffect(() => {
    if (flushSignal === undefined) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setShown(text.length);
  }, [flushSignal, text]);

  useEffect(() => {
    if (shown >= text.length) {
      onComplete?.();
      return;
    }
    const interval = Math.max(1, Math.round(1000 / Math.max(1, speedCharsPerSec)));
    const id = setTimeout(
      () => setShown((n) => Math.min(text.length, n + 1)),
      interval,
    );
    return () => clearTimeout(id);
  }, [shown, text, speedCharsPerSec, onComplete]);

  return <span>{committed.slice(0, Math.min(shown, committed.length))}</span>;
}
