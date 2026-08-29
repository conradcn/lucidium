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
 *   - the backend acknowledges that the turn finished — an
 *     ``s2c/text/complete`` (a beat was walked to and fully
 *     delivered) or an ``s2c/play/cancelled`` (the player pressed
 *     Stop), OR
 *   - an ``s2c/error`` lands on the WS bus (the backend rejected
 *     the command — re-enable the button so the player can retry
 *     instead of staring at a permanently-disabled choice).
 *
 * Calling ``useOptimisticAction()`` with no marker yields a one-
 * shot guard: the click locks the button and only a backend ack,
 * an error, or unmount releases it. Used for "Begin" / "Confirm"
 * buttons that transition out of the current screen, where the
 * screen itself unmounting is the natural reset.
 *
 * Earlier this hook reset on ANY game-ref change so a backend that
 * returned a no-op wouldn't lock the buttons forever. That
 * introduced a false-release bug: any background ``state/full``
 * (assets, summarizer) bumped the game ref, releasing the guard
 * BEFORE the advance had actually completed. Players reported the
 * dialog "reset to the last choice" because the buttons re-enabled
 * while the same node was still on screen.
 *
 * Narrowing the reset to ``current_node_id`` fixed that but left the
 * opposite hole: a reply that does NOT move the current node (the
 * backend walked nowhere, a beat arrived for the node already on
 * screen, the player cancelled) never releases the guard at all, so
 * the Continue glyph — and MainView's click-anywhere advance, which
 * shares this guard — stay dead until the screen unmounts. Hence the
 * ack-based releases: ``s2c/text/complete`` is emitted ONLY by the
 * foreground walk path (``handlers._emit_walk``), never by a
 * background asset or summarizer push, so it settles the turn
 * without reopening the false-release hole.
 */
export function useOptimisticAction(
  advanceMarker?: unknown,
): [boolean, (action: () => void) => void] {
  const [pending, setPending] = useState(false);
  const guard = useRef(false);
  const markerAtClick = useRef<unknown>(advanceMarker);

  const release = useCallback((): void => {
    if (!guard.current) return;
    guard.current = false;
    setPending(false);
  }, []);

  // Subscribe to the backend messages that end a turn. The release
  // happens in the subscription CALLBACK — the WS bus is the external
  // system this effect synchronises with — rather than bouncing
  // through a tick counter and a second effect, which would be a
  // setState in an effect body for no benefit.
  // Defensive: in test setups that mock ``ensureClient`` to return
  // a non-object (e.g. ``vi.fn()`` with no implementation) we silently
  // skip subscription rather than crashing the entire component tree
  // during render setup.
  useEffect(() => {
    const client = optionalClient();
    if (!client || typeof client.on !== "function") return;
    const offs = [
      client.on("s2c/error", release),
      client.on("s2c/text/complete", release),
      client.on("s2c/play/cancelled", release),
    ];
    return () => {
      for (const off of offs) {
        if (typeof off === "function") off();
      }
    };
  }, [release]);

  useEffect(() => {
    if (advanceMarker !== markerAtClick.current) release();
  }, [advanceMarker, release]);

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
