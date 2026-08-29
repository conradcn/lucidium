/**
 * Automatic GPU-acceleration status surface.
 *
 * The backend now AUTO-PROVISIONS GPU acceleration: on startup it
 * probes the GPU and, if the active torch overlay doesn't match the
 * recommended flavor, it background-downloads + installs the right one
 * itself — streaming progress over ``s2c/torch_overlay/progress`` and
 * ending with a terminal ``s2c/torch_overlay/status`` carrying
 * ``activated`` (relaunch needed).
 *
 * This component is the UI reflection of that flow. It is NOT a
 * "Download?" prompt — the download is already happening server-side.
 * Instead it shows:
 *
 *   * an unobtrusive, NON-BLOCKING banner ("Setting up GPU
 *     acceleration… 42%") while provisioning is in flight, driven by
 *     the streamed progress events — the player can keep using the app
 *     on the CPU overlay the whole time; and
 *   * a dismissible "Restart Lucidium to enable GPU acceleration"
 *     notice once the terminal ``activated`` status lands.
 *
 * It renders NOTHING when no provisioning is happening (already on the
 * right overlay, or no GPU). The manual flavor switcher lives in
 * Settings (see ``GpuAccelerationSection``); this surface never asks
 * the user to initiate anything.
 *
 * Reuses the shared ``useTorchOverlay`` hook + the progress / relaunch
 * sub-components so the look matches the Settings control.
 */

import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";

import {
  RelaunchNotice,
  TorchOverlayProgress,
  flavorLabel,
  useTorchOverlay,
} from "../app/TorchOverlayPanel";

// Grace window before the auto-relaunch fires once the backend signals
// a wheel was activated. Just long enough for the player to read the
// banner ("Restarting to enable GPU acceleration…") so the window
// closing isn't a surprise; short enough that they're not stuck waiting.
const AUTO_RELAUNCH_DELAY_MS = 4000;

// How long ``bytes_done`` may sit unchanged before we call the download
// stalled in the UI. Comfortably under the backend's stall watchdog
// (120s by default) so the player sees "this looks stuck" BEFORE the
// backend gives up and the banner disappears — a slow-but-alive download
// keeps ticking bytes every few seconds and never trips this.
const STALL_HINT_MS = 30_000;

/**
 * True when the byte counter hasn't moved for {@link STALL_HINT_MS}.
 *
 * Without this, a stalled download and a slow one look identical: the
 * backend reports the same "downloading" stage either way, and before
 * the byte totals were plumbed through both rendered the same
 * indeterminate bar. Freshness is a UI-side observation — it needs a
 * clock, not another protocol field.
 */
function useStalled(bytesDone: number | null): boolean {
  const [stalled, setStalled] = useState(false);

  // Restarting the timer on every counter change is the whole mechanism:
  // a live download keeps clearing it, a dead one lets it fire.
  // Clearing the flag is a render-phase adjustment rather than part of
  // the effect body: it must land before the banner paints, so a
  // download that resumes never shows one frame of the stall hint.
  const [seenBytes, setSeenBytes] = useState(bytesDone);
  if (seenBytes !== bytesDone) {
    setSeenBytes(bytesDone);
    setStalled(false);
  }

  useEffect(() => {
    if (bytesDone === null) return;
    const timer = window.setTimeout(() => setStalled(true), STALL_HINT_MS);
    return () => window.clearTimeout(timer);
  }, [bytesDone]);

  return stalled;
}

export function GpuAccelStatus(): JSX.Element | null {
  const { state, requestStatus } = useTorchOverlay();
  // Schedule-guard ref (NOT useState): flipping to true on the first
  // activated status must not re-render — a re-render would re-run
  // the auto-relaunch effect and its cleanup would CLEAR the timer
  // we just queued, so the relaunch would never fire. A ref is the
  // standard idiom for "remember we did this without affecting
  // render output."
  const scheduledRef = useRef(false);

  useEffect(() => {
    // Probe once on mount so we learn the current overlay state. We do
    // NOT trigger any install here — the backend drives provisioning on
    // its own; this is purely so a status that already carries
    // ``activated`` (e.g. provisioning finished before the UI mounted)
    // surfaces the relaunch notice. The hook handles its own
    // (re)subscription, so re-probing every render would just spam the
    // backend.
    requestStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-relaunch: when the backend signals a wheel was activated,
  // schedule the Electron process to relaunch itself so the new torch
  // is on sys.path next launch. The grace window lets the player read
  // the banner. We expose the relaunch via the preload bridge; in test
  // runs (no preload) the call is a no-op so the banner just sits.
  useEffect(() => {
    if (!state.activatedFlavor) return;
    if (scheduledRef.current) return;
    scheduledRef.current = true;
    // Structural cast, justified: ``window.lucidium`` is injected by the
    // Electron preload at runtime — no ambient declaration to narrow to.
    const w = window as unknown as {
      lucidium?: { relaunchForOverlay?: () => void };
    };
    const relaunch = w.lucidium?.relaunchForOverlay;
    if (typeof relaunch !== "function") return;
    const timer = window.setTimeout(() => {
      relaunch();
    }, AUTO_RELAUNCH_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [state.activatedFlavor]);

  // Must be called before the early returns below (hooks order).
  const stalled = useStalled(state.progress?.bytes_done ?? null);

  // Provisioning finished and a relaunch is happening: show the
  // restart-imminent banner. No dismiss button — the window is about
  // to close on its own and a dismiss control would mislead the
  // player into thinking they can opt out of the restart.
  if (state.activatedFlavor) {
    return (
      <div
        className="gpu-accel-status gpu-accel-status--done"
        role="status"
        data-testid="gpu-accel-relaunch"
      >
        <div className="gpu-accel-status__body">
          <RelaunchNotice flavor={state.activatedFlavor} />
          <p className="settings-field-hint" style={{ margin: "0.4rem 0 0" }}>
            Restarting automatically in a few seconds…
          </p>
        </div>
      </div>
    );
  }

  // Auto-provisioning in flight: a non-blocking banner. ``installing``
  // flips true the moment progress starts streaming (the hook sets it
  // on the first ``s2c/torch_overlay/progress``), so this appears WITHOUT
  // any user action. Recommended flavor (when known) names what's being
  // set up; we fall back to generic copy before the first status lands.
  const provisioning = state.installing || state.progress !== null;
  if (provisioning) {
    const flavor =
      state.progress?.flavor ?? state.recommended ?? null;
    const label = flavor ? flavorLabel(flavor) : "GPU acceleration";
    return (
      <div
        className="gpu-accel-status gpu-accel-status--busy"
        role="status"
        data-testid="gpu-accel-status"
      >
        <div className="gpu-accel-status__body">
          <span className="gpu-accel-status__title">
            Setting up GPU acceleration ({label})… image generation is
            paused until this finishes. Dialog continues normally.
          </span>
          {state.progress ? (
            <TorchOverlayProgress progress={state.progress} />
          ) : (
            <p className="settings-field-hint" style={{ margin: 0 }}>
              Starting download…
            </p>
          )}
          {stalled ? (
            // The bar alone can't say this: a frozen counter and a slow
            // one look the same. The backend aborts a truly dead
            // transfer on its own watchdog, after which this banner
            // disappears and image generation resumes.
            <p
              className="settings-field-hint"
              data-testid="gpu-accel-stalled"
              style={{ margin: "0.4rem 0 0" }}
            >
              The download hasn't progressed for a while — the connection
              may be blocked. Lucidium will give up shortly and carry on
              without GPU acceleration.
            </p>
          ) : null}
        </div>
      </div>
    );
  }

  // Nothing to provision (already on the recommended overlay, or no GPU
  // detected): render nothing.
  return null;
}
