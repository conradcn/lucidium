import type { JSX } from "react";
import { useEffect, useRef } from "react";

/**
 * Single top-level music orchestrator. Mounted ONCE in App, owns
 * two ``HTMLAudioElement`` slots, and crossfades between any two
 * URLs the parent passes in — including ``null`` (fade to
 * silence). Replaces the previous MainView-only BackgroundMusic
 * so that menu ↔ in-game transitions also crossfade instead of
 * popping.
 *
 * Props are intentionally narrow:
 *   * ``url`` is the resolved fetchable URL (already through
 *     ``assetUrl`` for backend-served paths, or a literal
 *     ``/main-menu/MainTheme.mp3`` for bundled assets).
 *   * ``masterVolume`` × ``musicVolume`` become the active
 *     element's effective volume — both 0..1, multiplied so a
 *     master of zero silences everything regardless of the
 *     per-channel slider.
 *   * ``enabled`` gates playback entirely — when off, both slots
 *     pause and ``url`` is ignored. Used for the music
 *     ``enabled`` setting (model unloaded server-side, but the
 *     menu theme should still respect the toggle).
 *
 * Browser autoplay restriction: the first ``play()`` after page
 * load may reject if the user hasn't interacted yet. We arm a
 * one-shot listener on ``pointerdown``/``keydown`` that retries
 * play() on whichever slot has a src loaded. Plays-through-
 * gesture matches the rest of the engine's audio behaviour.
 */
interface Props {
  url: string | null;
  masterVolume: number;
  musicVolume: number;
  enabled: boolean;
  /** Crossfade duration in ms. 1500 ms reads as a deliberate
   *  scene change without feeling sluggish; the previous
   *  in-game-only component used the same value. */
  crossfadeMs?: number;
}

export function MusicPlayer({
  url, masterVolume, musicVolume, enabled, crossfadeMs = 1500,
}: Props): JSX.Element {
  const slotARef = useRef<HTMLAudioElement | null>(null);
  const slotBRef = useRef<HTMLAudioElement | null>(null);
  // ``activeSlot`` is the slot currently playing (or last played).
  // The next URL change writes to the OTHER slot so both can be
  // alive simultaneously through the crossfade.
  const activeSlotRef = useRef<"A" | "B">("A");
  const currentUrlRef = useRef<string | null>(null);
  // Cache the live volume so a settings-driven change can adjust
  // the active slot WITHOUT restarting playback. The crossfade
  // path also reads this so an incoming track ramps to the right
  // target rather than hard-coded 0.4.
  const targetVolumeRef = useRef<number>(0);
  // Per-slot fade intervals. Each new fade on a slot CANCELS the
  // previous one on that same slot — otherwise rapid url changes
  // (menu → game → menu, settings flip-flop, etc.) leave the old
  // 30 ms-tick intervals running until they self-complete, each
  // holding a closure over the audio element. The pile-up is the
  // most likely cause of the "main menu memory grows over time"
  // symptom, since menu navigations are the main path that swaps
  // the music url mid-fade.
  const fadeIntervalsRef = useRef<{ A: number | null; B: number | null }>({
    A: null, B: null,
  });

  // Compute the effective per-element volume from master ×
  // channel × enabled. Held in a ref the URL effect can read
  // without re-firing every time the sliders move.
  const effectiveVolume = enabled
    ? Math.max(0, Math.min(1, masterVolume * musicVolume))
    : 0;
  // Synced in an effect, not during render: refs must not be written
  // while rendering. This effect is declared BEFORE every effect that
  // reads the ref, so on any commit the ref is current by the time a
  // consumer runs — identical timing to the old render-phase write.
  useEffect(() => {
    targetVolumeRef.current = effectiveVolume;
  }, [effectiveVolume]);

  // Apply live volume changes to whichever slot is playing —
  // moving the master/music slider should react instantly, not
  // wait for the next URL change.
  useEffect(() => {
    const active = activeSlotRef.current === "A"
      ? slotARef.current : slotBRef.current;
    if (!active) return;
    if (effectiveVolume <= 0) {
      // Pause both slots — there's nothing audible to render
      // and pausing avoids decoding cycles in the background.
      slotARef.current?.pause();
      slotBRef.current?.pause();
    } else {
      // Apply directly — no ramp on slider drags. The active
      // element resumes if it was previously paused (e.g. when
      // re-enabling music or bumping master from zero).
      active.volume = effectiveVolume;
      if (active.src && active.paused) active.play().catch(() => {});
    }
  }, [effectiveVolume]);

  // URL change → crossfade. The previous slot fades out while
  // the new slot fades up to the live target volume; if either
  // ref is unavailable we silently skip — the next render
  // re-runs the effect against fresh refs.
  useEffect(() => {
    if (!enabled) {
      slotARef.current?.pause();
      slotBRef.current?.pause();
      currentUrlRef.current = null;
      return;
    }
    if (currentUrlRef.current === url) return;
    currentUrlRef.current = url;
    const incoming = activeSlotRef.current === "A"
      ? slotBRef.current : slotARef.current;
    const outgoing = activeSlotRef.current === "A"
      ? slotARef.current : slotBRef.current;
    if (!incoming) return;
    // No URL means "fade to silence" — keep the outgoing slot
    // ramping down but don't load anything new.
    if (!url) {
      const outgoingSlot = activeSlotRef.current;
      startFade(fadeIntervalsRef.current, outgoingSlot, outgoing, 0, crossfadeMs);
      return;
    }
    incoming.src = url;
    incoming.volume = 0;
    incoming.loop = true;
    incoming.play().catch(() => {});
    const incomingSlot = activeSlotRef.current === "A" ? "B" : "A";
    const outgoingSlot = activeSlotRef.current;
    startFade(fadeIntervalsRef.current, incomingSlot, incoming, targetVolumeRef.current, crossfadeMs);
    if (outgoing) startFade(fadeIntervalsRef.current, outgoingSlot, outgoing, 0, crossfadeMs);
    activeSlotRef.current = incomingSlot;
  }, [url, enabled, crossfadeMs]);

  // Clear any in-flight fade intervals on unmount. Without this,
  // hot-reloading or any code path that unmounts the player mid-
  // fade leaves the 30 ms ticks running for the rest of the page's
  // lifetime, each holding a closure over an audio element React
  // will otherwise have released.
  useEffect(() => {
    const intervals = fadeIntervalsRef.current;
    return () => {
      if (intervals.A !== null) window.clearInterval(intervals.A);
      if (intervals.B !== null) window.clearInterval(intervals.B);
      intervals.A = null;
      intervals.B = null;
    };
  }, []);

  // Arm autoplay on first gesture. Browsers block play() until
  // a user has interacted; once they have, both slots can be
  // resumed (the eager .play() in the URL effect either
  // succeeded or hit an AbortError-style rejection that we
  // swallowed, so re-trying is safe).
  useEffect(() => {
    const arm = (): void => {
      const a = slotARef.current;
      const b = slotBRef.current;
      if (a && a.src && a.paused) a.play().catch(() => {});
      if (b && b.src && b.paused) b.play().catch(() => {});
      window.removeEventListener("pointerdown", arm);
      window.removeEventListener("keydown", arm);
    };
    window.addEventListener("pointerdown", arm);
    window.addEventListener("keydown", arm);
    return () => {
      window.removeEventListener("pointerdown", arm);
      window.removeEventListener("keydown", arm);
    };
  }, []);

  return (
    <>
      <audio ref={slotARef} aria-hidden style={{ display: "none" }} />
      <audio ref={slotBRef} aria-hidden style={{ display: "none" }} />
    </>
  );
}

/** Start a linear ramp on the audio element's ``volume`` property,
 *  CANCELLING any previous fade on the same slot. Hits ``target``
 *  exactly at ``durationMs`` and pauses the element if the target
 *  was zero (silence-after-fadeout). Tick interval picked to feel
 *  smooth without stressing the main thread — 30 ms is well under
 *  perceptual threshold.
 *
 *  The cancel-first behaviour is the load-bearing part: rapid url
 *  changes (e.g. menu → game → menu before the 1.5 s crossfade
 *  completes) used to pile new intervals on top of in-flight ones,
 *  each holding a closure over the audio element. Storing the id
 *  per-slot lets the next call clear the previous one so at most
 *  one interval per slot exists at a time. */
function startFade(
  intervals: { A: number | null; B: number | null },
  slot: "A" | "B",
  el: HTMLAudioElement | null,
  target: number,
  durationMs: number,
): void {
  if (!el) return;
  const prior = intervals[slot];
  if (prior !== null) window.clearInterval(prior);
  const start = el.volume;
  const startedAt = performance.now();
  const id = window.setInterval(() => {
    const t = Math.min(1, (performance.now() - startedAt) / durationMs);
    el.volume = start + (target - start) * t;
    if (t >= 1) {
      window.clearInterval(id);
      if (intervals[slot] === id) intervals[slot] = null;
      if (target === 0) el.pause();
    }
  }, 30);
  intervals[slot] = id;
}
