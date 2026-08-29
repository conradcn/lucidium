import { useCallback, useEffect, useRef, useState } from "react";

import { optionalClient } from "./optionalClient";

/**
 * Hook that lets a button trigger its action ONCE per "open window",
 * disabling itself for the rest of the window. The window resets
 * when:
 *
 *   - the ``advanceMarker`` value changes from the value at click
 *     time (caller passes the current dialog node id; release fires
 *     when the renderer sees a NEW node land in the store), OR
 *   - an ``s2c/error`` lands on the WS bus (the backend rejected
 *     the command — re-enable the button so the player can retry
 *     instead of staring at a permanently-disabled choice).
 *
 * Calling ``useOptimisticAction()`` with no marker yields a one-
 * shot guard: the click locks the button and only an error or
 * unmount releases it. Used for "Begin" / "Confirm" buttons that
 * transition out of the current screen, where the screen itself
 * unmounting is the natural reset.
 *
 * Earlier this hook reset on ANY game-ref change so a backend that
 * returned a no-op wouldn't lock the buttons forever. That
 * introduced a false-release bug: any background ``state/full``
 * (assets, summarizer) bumped the game ref, releasing the guard
 * BEFORE the advance had actually completed. Players reported the
 * dialog "reset to the last choice" because the buttons re-enabled
 * while the same node was still on screen.
 */
export function useOptimisticAction(
  advanceMarker?: unknown,
): [boolean, (action: () => void) => void] {
  const [pending, setPending] = useState(false);
  const guard = useRef(false);
  const markerAtClick = useRef<unknown>(advanceMarker);

  // Subscribe to s2c/error so a failed advance releases the guard.
  // The release happens in the subscription CALLBACK — the WS bus is
  // the external system this effect synchronises with — rather than
  // bouncing through an "error tick" counter and a second effect,
  // which would be a setState in an effect body for no benefit.
  // Defensive: in test setups that mock ``ensureClient`` to return
  // a non-object (e.g. ``vi.fn()`` with no implementation) we silently
  // skip subscription rather than crashing the entire component tree
  // during render setup.
  useEffect(() => {
    const client = optionalClient();
    if (!client || typeof client.on !== "function") return;
    const off = client.on("s2c/error", () => {
      if (!guard.current) return;
      setPending(false);
      guard.current = false;
    });
    return off;
  }, []);

  useEffect(() => {
    if (!guard.current) return;
    if (advanceMarker !== markerAtClick.current) {
      setPending(false);
      guard.current = false;
    }
  }, [advanceMarker]);

  const run = useCallback(
    (action: () => void): void => {
      if (guard.current) return;
      guard.current = true;
      markerAtClick.current = advanceMarker;
      setPending(true);
      action();
    },
    [advanceMarker],
  );

  return [pending, run];
}
