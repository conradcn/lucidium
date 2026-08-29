/**
 * GPU-acceleration (torch runtime overlay) UI.
 *
 * The frozen build ships only the ``cpu`` overlay; this panel drives
 * the ``c2s/torch_overlay/*`` round-trip that downloads + activates a
 * GPU-correct torch build (CUDA / ROCm / DirectML / XPU). It mirrors
 * the SDXL "Download" pattern: a deliberate user click triggers a
 * multi-hundred-MB fetch, streamed back as
 * ``s2c/torch_overlay/progress`` events and terminated by a final
 * ``s2c/torch_overlay/status`` carrying ``activated`` (relaunch
 * needed).
 *
 * KEY UX FACT: installing only flips the NEXT-launch overlay pointer —
 * torch is already imported in the running backend — so on success we
 * tell the player to relaunch the app for GPU acceleration to take
 * effect.
 *
 * Two consumers share this machinery:
 *   * ``GpuAccelStatus`` (auto): a non-blocking banner that REFLECTS
 *     the backend's automatic provisioning — progress while it
 *     downloads, then a relaunch notice — without ever asking the
 *     player to initiate a download.
 *   * ``SettingsScreen``: a full manual control with a flavor select +
 *     an install/switch/re-download button, plus the auto-provision
 *     toggle.
 *
 * Both reuse the hook + the progress + relaunch sub-components below
 * so the look stays consistent.
 */

import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";

import { ensureClient, send } from "./client";
import type { S2CTorchOverlayProgress } from "../shared/generated/S2CTorchOverlayProgress.schema";
import type { C2STorchOverlayStatus } from "../shared/generated/C2STorchOverlayStatus.schema";
import type { C2STorchOverlayInstall } from "../shared/generated/C2STorchOverlayInstall.schema";

// The five overlay flavors the backend understands. CPU is the
// always-available fallback the app ships with; the GPU flavors are
// what a fresh download fetches.
export const TORCH_FLAVORS = [
  "cuda",
  "rocm",
  "directml",
  "xpu",
  "cpu",
] as const;
export type TorchFlavor = (typeof TORCH_FLAVORS)[number];

const GPU_FLAVORS: ReadonlySet<string> = new Set([
  "cuda",
  "rocm",
  "directml",
  "xpu",
]);

export function isGpuFlavor(flavor: string | null | undefined): boolean {
  return flavor != null && GPU_FLAVORS.has(flavor);
}

// Human-friendly labels for the prompt copy + the select. Keyed by the
// raw flavor string the backend uses on the wire.
const FLAVOR_LABELS: Record<string, string> = {
  cuda: "NVIDIA GPU (CUDA)",
  rocm: "AMD GPU (ROCm)",
  directml: "DirectML GPU",
  xpu: "Intel GPU (XPU)",
  cpu: "CPU only",
};

export function flavorLabel(flavor: string): string {
  return FLAVOR_LABELS[flavor] ?? flavor;
}

// The detected-vendor phrasing for the first-run prompt headline.
const VENDOR_PHRASE: Record<string, string> = {
  cuda: "An NVIDIA GPU was detected",
  rocm: "An AMD GPU was detected",
  directml: "A DirectX 12 GPU was detected",
  xpu: "An Intel GPU was detected",
};

export function vendorPhrase(flavor: string): string {
  return VENDOR_PHRASE[flavor] ?? "A compatible GPU was detected";
}

export interface TorchOverlayState {
  /** The flavor the GPU probe recommends for this machine. */
  recommended: string | null;
  /** Flavors with an overlay already on disk. */
  installed: string[];
  /** Flavor the entry shim loads on next launch (null = fresh). */
  active: string | null;
  /** Resolved overlay root, for diagnostics / tooltips. */
  runtimeDir: string | null;
  /** Live download progress while an install is in flight. */
  progress: S2CTorchOverlayProgress | null;
  /** True between the install click and its terminal status/error. */
  installing: boolean;
  /**
   * The flavor whose post-install status carried ``activated: true`` —
   * a relaunch is needed for it to take effect. Null until that
   * terminal status lands.
   */
  activatedFlavor: string | null;
  /** Backend error message, if the last install failed. */
  error: string | null;
}

const INITIAL_STATE: TorchOverlayState = {
  recommended: null,
  installed: [],
  active: null,
  runtimeDir: null,
  progress: null,
  installing: false,
  activatedFlavor: null,
  error: null,
};

/**
 * Subscribes to the torch-overlay status + progress messages and
 * exposes a ``requestStatus`` / ``install`` action pair. Shared by the
 * first-run prompt and the Settings control so both render from one
 * source of truth.
 */
export function useTorchOverlay(): {
  state: TorchOverlayState;
  requestStatus: () => void;
  install: (flavor: string) => void;
} {
  const [state, setState] = useState<TorchOverlayState>(INITIAL_STATE);

  // Hold the flavor we're currently installing so the terminal status
  // (which doesn't necessarily echo it) can be attributed correctly
  // for the relaunch message. Lives in a ref so the message handlers
  // — registered once at mount — always read the latest value.
  const installingFlavorRef = useRef<string | null>(null);

  useEffect(() => {
    const client = ensureClient();

    const offStatus = client.on("s2c/torch_overlay/status", (p) => {
      setState((prev) => ({
        ...prev,
        recommended: p.recommended ?? null,
        installed: Array.isArray(p.installed) ? p.installed : [],
        active: p.active ?? null,
        runtimeDir: p.runtime_dir ?? null,
        // A status carrying ``activated`` is the terminal reply to an
        // install: clear the in-flight flags and surface the relaunch
        // message. A plain status (the initial probe) leaves any
        // existing activated flavor untouched.
        installing: p.activated ? false : prev.installing,
        progress: p.activated ? null : prev.progress,
        activatedFlavor: p.activated
          ? installingFlavorRef.current ?? p.active ?? prev.activatedFlavor
          : prev.activatedFlavor,
        error: p.activated ? null : prev.error,
      }));
      if (p.activated) installingFlavorRef.current = null;
    });

    const offProgress = client.on("s2c/torch_overlay/progress", (p) => {
      setState((prev) => ({ ...prev, progress: p, installing: true }));
    });

    // A backend error during an install (e.g. network failure) ends
    // the in-flight state and shows the message. We only claim the
    // error while we believe an install is running so we don't steal
    // unrelated s2c/error events (the global handler still logs them).
    const offError = client.on("s2c/error", (p) => {
      setState((prev) =>
        prev.installing
          ? {
              ...prev,
              installing: false,
              progress: null,
              error: p.message ?? "Install failed.",
            }
          : prev,
      );
    });

    return () => {
      offStatus();
      offProgress();
      offError();
    };
  }, []);

  const requestStatus = (): void => {
    const payload: C2STorchOverlayStatus = {};
    send("c2s/torch_overlay/status", payload);
  };

  const install = (flavor: string): void => {
    installingFlavorRef.current = flavor;
    setState((prev) => ({
      ...prev,
      installing: true,
      progress: null,
      activatedFlavor: null,
      error: null,
    }));
    const payload: C2STorchOverlayInstall = { flavor };
    send("c2s/torch_overlay/install", payload);
  };

  return { state, requestStatus, install };
}

/**
 * Streamed download-progress bar. Determinate when the backend knows
 * ``bytes_total``, indeterminate otherwise. Labels itself with the
 * backend's ``stage`` string so the bar reads "downloading" /
 * "unpacking" without the renderer parsing byte counts.
 */
export function TorchOverlayProgress({
  progress,
}: {
  progress: S2CTorchOverlayProgress;
}): JSX.Element {
  const total = progress.bytes_total ?? null;
  const done = progress.bytes_done ?? 0;
  const pct =
    total && total > 0 ? Math.min(100, Math.round((done / total) * 100)) : null;
  return (
    <div className="torch-overlay-progress" data-testid="torch-overlay-progress">
      <div className="torch-overlay-progress__label">
        <span>{progress.stage}</span>
        {pct !== null ? (
          <span>{pct}%</span>
        ) : (
          <span>{formatBytes(done)}</span>
        )}
      </div>
      <progress
        className="torch-overlay-progress__bar"
        // Omitting ``value`` renders an indeterminate bar — exactly
        // what we want when the total is unknown.
        {...(pct !== null ? { value: pct, max: 100 } : {})}
        data-testid="torch-overlay-progress-bar"
      />
    </div>
  );
}

/**
 * Post-install relaunch notice. The install only flips the next-launch
 * pointer, so GPU acceleration is inert until the app restarts.
 */
export function RelaunchNotice({
  flavor,
}: {
  flavor: string;
}): JSX.Element {
  return (
    <div
      className="settings-field-notice"
      role="status"
      data-testid="torch-overlay-relaunch"
      style={{
        marginTop: "0.5rem",
        padding: "0.6rem 0.75rem",
        borderLeft: "3px solid var(--accent, #7ec77e)",
        background: "rgba(126, 199, 126, 0.08)",
        fontSize: "0.85rem",
        lineHeight: 1.45,
      }}
    >
      <strong>{flavorLabel(flavor)} acceleration installed.</strong> Restart
      Lucidium for it to take effect — torch is already loaded in the running
      session, so the new runtime kicks in on the next launch.
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
}
