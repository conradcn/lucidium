import { useEffect, useState } from "react";

/**
 * Mirror ``value`` into local state, but defer each change by
 * ``delayMs``. Used by the menu transition to stagger per-button
 * animation triggers without each button needing its own timer
 * orchestration.
 */
export function useDelayedValue<T>(value: T, delayMs: number): T {
  const [delayed, setDelayed] = useState(value);
  useEffect(() => {
    // Always go through the timer, even for a non-positive delay: a
    // zero-delay timeout lands on the next macrotask, which is
    // indistinguishable from the old synchronous-in-effect assignment
    // for the stagger animation, and keeps the state update in a
    // callback rather than in the effect body.
    const id = setTimeout(() => setDelayed(value), Math.max(0, delayMs));
    return () => clearTimeout(id);
  }, [value, delayMs]);
  return delayed;
}
