/**
 * Frontend coverage for the torch-overlay (GPU acceleration) UI.
 *
 * The backend now AUTO-PROVISIONS GPU acceleration: on startup it
 * detects the GPU and, if the active overlay is wrong, it
 * background-downloads + installs the right one itself, streaming
 * ``s2c/torch_overlay/progress`` and ending with a terminal
 * ``s2c/torch_overlay/status`` carrying ``activated``. This exercises
 * the React side that reflects that flow:
 *
 *   1. ``GpuAccelStatus`` shows a non-blocking progress banner WITHOUT
 *      any user click as soon as progress streams in.
 *   2. A terminal ``activated`` status surfaces a dismissible
 *      relaunch-needed notice.
 *   3. Nothing renders when no provisioning is happening (no progress,
 *      no activated status).
 *   4. The Settings GPU section still installs the selected flavor
 *      manually, and its ``auto_gpu_provision`` toggle autosaves the
 *      right settings update.
 *
 * The websocket client is mocked the same way the other unit tests do
 * (``confirm_add_side_character`` / ``sdxl_guidance``), but with a tiny
 * in-memory event bus on ``ensureClient().on`` so the test can emit
 * S2C payloads and assert the UI reacts.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render } from "@testing-library/react";

// In-memory fan-out matching WsClient.on's contract: register a
// listener, get an unsubscribe back. ``emit`` pushes a payload to all
// listeners of a type — the test's stand-in for an inbound S2C frame.
const listeners = new Map<string, Set<(p: unknown) => void>>();
function emit(type: string, payload: unknown): void {
  act(() => {
    for (const fn of listeners.get(type) ?? []) fn(payload);
  });
}
function clearBus(): void {
  listeners.clear();
}

vi.mock("../../src/app/client", () => ({
  send: vi.fn(),
  ensureClient: vi.fn(() => ({
    on: (type: string, fn: (p: unknown) => void) => {
      let set = listeners.get(type);
      if (!set) {
        set = new Set();
        listeners.set(type, set);
      }
      set.add(fn);
      return () => set?.delete(fn);
    },
  })),
}));

import { send } from "../../src/app/client";
import { GpuAccelStatus } from "../../src/screens/GpuAccelPrompt";
import { SettingsScreen } from "../../src/settings/SettingsScreen";
import { useLucidiumStore } from "../../src/state/store";

afterEach(() => {
  cleanup();
  clearBus();
});

describe("GpuAccelStatus — automatic provisioning surface", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
  });

  it("probes status on mount but never triggers an install itself", () => {
    render(<GpuAccelStatus />);
    expect(send).toHaveBeenCalledWith("c2s/torch_overlay/status", {});
    // The automatic surface MUST NOT initiate a download — the backend
    // drives provisioning on its own.
    const installCalls = vi
      .mocked(send)
      .mock.calls.filter((c) => c[0] === "c2s/torch_overlay/install");
    expect(installCalls).toHaveLength(0);
  });

  it("renders nothing when no provisioning is happening", () => {
    const { queryByTestId } = render(<GpuAccelStatus />);
    // A plain status (no progress, no activated) is not provisioning.
    emit("s2c/torch_overlay/status", {
      recommended: "cuda",
      installed: ["cpu"],
      active: "cpu",
    });
    expect(queryByTestId("gpu-accel-status")).toBeNull();
    expect(queryByTestId("gpu-accel-relaunch")).toBeNull();
  });

  it("shows the auto progress banner when progress streams without a user click", () => {
    const { queryByTestId, getByTestId } = render(<GpuAccelStatus />);
    // No click — the backend simply starts streaming progress.
    emit("s2c/torch_overlay/progress", {
      flavor: "cuda",
      stage: "downloading",
      bytes_done: 420,
      bytes_total: 1000,
    });
    expect(queryByTestId("gpu-accel-status")).not.toBeNull();
    const bar = getByTestId("torch-overlay-progress-bar") as HTMLProgressElement;
    expect(bar.value).toBe(42);
    expect(getByTestId("gpu-accel-status").textContent).toMatch(
      /setting up gpu acceleration/i,
    );
    // It is NOT a modal / does not present a Download choice.
    expect(queryByTestId("gpu-accel-download")).toBeNull();
  });

  it("surfaces a restart-imminent notice on the activated terminal status and triggers the auto-relaunch IPC", () => {
    vi.useFakeTimers();
    const relaunchSpy = vi.fn();
    (window as unknown as {
      lucidium: { relaunchForOverlay: () => void };
    }).lucidium = { relaunchForOverlay: relaunchSpy };
    try {
      const { queryByTestId } = render(<GpuAccelStatus />);
      emit("s2c/torch_overlay/progress", {
        flavor: "cuda",
        stage: "downloading",
        bytes_done: 1000,
        bytes_total: 1000,
      });
      emit("s2c/torch_overlay/status", {
        recommended: "cuda",
        installed: ["cpu", "cuda"],
        active: "cuda",
        activated: true,
      });
      const relaunch = queryByTestId("gpu-accel-relaunch");
      expect(relaunch).not.toBeNull();
      expect(relaunch?.textContent ?? "").toMatch(/restart/i);
      // Progress banner is gone once provisioning completes.
      expect(queryByTestId("gpu-accel-status")).toBeNull();
      // It is NOT dismissible — the auto-restart is imminent and a
      // dismiss control would mislead the player into thinking they
      // could opt out.
      expect(queryByTestId("gpu-accel-relaunch-dismiss")).toBeNull();
      // After the grace window, the preload bridge fires.
      expect(relaunchSpy).not.toHaveBeenCalled();
      vi.advanceTimersByTime(5000);
      expect(relaunchSpy).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
      delete (window as unknown as { lucidium?: unknown }).lucidium;
    }
  });

  it("shows an indeterminate bar when bytes_total is unknown", () => {
    const { getByTestId } = render(<GpuAccelStatus />);
    emit("s2c/torch_overlay/progress", {
      flavor: "cuda",
      stage: "resolving",
      bytes_done: 1024,
      bytes_total: null,
    });
    const bar = getByTestId("torch-overlay-progress-bar") as HTMLProgressElement;
    // No ``value`` attribute => indeterminate progress element.
    expect(bar.hasAttribute("value")).toBe(false);
  });
});

describe("SettingsScreen — GPU acceleration control", () => {
  beforeEach(() => {
    vi.mocked(send).mockClear();
    useLucidiumStore.setState({
      settings: {
        llm: { base_url: "", model: "", api_key: "", temperature: 0.8, max_tokens: 1024 },
        image: { backend: "embedded", embedded_models_dir: "", auto_gpu_provision: true },
        music: { enabled: false },
        audio: { master_volume: 0.7, music_volume: 0.4 },
        user_profile: {},
      },
    });
  });

  it("renders the GPU-acceleration section and probes status", () => {
    const { getByTestId } = render(<SettingsScreen onBack={() => undefined} />);
    expect(getByTestId("gpu-acceleration-section")).not.toBeNull();
    expect(send).toHaveBeenCalledWith("c2s/torch_overlay/status", {});
  });

  it("keeps the manual switcher: installs the selected flavor (defaults to recommended)", () => {
    const { getByTestId } = render(<SettingsScreen onBack={() => undefined} />);
    emit("s2c/torch_overlay/status", {
      recommended: "cuda",
      installed: ["cpu"],
      active: "cpu",
    });
    // No flavor change => the select sits on the recommended flavor.
    fireEvent.click(getByTestId("gpu-acceleration-install"));
    expect(send).toHaveBeenCalledWith("c2s/torch_overlay/install", {
      flavor: "cuda",
    });
  });

  it("shows the relaunch message after an activated install", () => {
    const { getByTestId, queryByTestId } = render(
      <SettingsScreen onBack={() => undefined} />,
    );
    emit("s2c/torch_overlay/status", {
      recommended: "cuda",
      installed: ["cpu"],
      active: "cpu",
    });
    fireEvent.click(getByTestId("gpu-acceleration-install"));
    emit("s2c/torch_overlay/status", {
      recommended: "cuda",
      installed: ["cpu", "cuda"],
      active: "cuda",
      activated: true,
    });
    expect(queryByTestId("torch-overlay-relaunch")).not.toBeNull();
  });

  it("auto_gpu_provision toggle autosaves the right settings update", async () => {
    const { getByTestId } = render(<SettingsScreen onBack={() => undefined} />);
    // Defaults on; turn it off.
    const toggle = getByTestId("gpu-auto-provision-toggle") as HTMLInputElement;
    expect(toggle.checked).toBe(true);
    fireEvent.click(toggle);
    // Autosave debounces ~350 ms; the image draft (carrying
    // auto_gpu_provision) rides the same patch.
    await new Promise((r) => setTimeout(r, 450));
    const call = vi
      .mocked(send)
      .mock.calls.find(
        (c) =>
          c[0] === "c2s/settings/update" &&
          (c[1] as { patch: { image?: { auto_gpu_provision?: boolean } } }).patch
            .image?.auto_gpu_provision === false,
      );
    expect(call).toBeTruthy();
  });
});
