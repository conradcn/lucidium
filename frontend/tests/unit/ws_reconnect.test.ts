import { describe, expect, it, afterEach, beforeEach, vi } from "vitest";

import { WsClient } from "../../src/ws/client";

class FakeSocket {
  static instances: FakeSocket[] = [];
  static OPEN = 1;
  readyState = 0;
  onopen: ((event: unknown) => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: ((event: unknown) => void) | null = null;
  onclose: ((event: unknown) => void) | null = null;
  sent: string[] = [];

  constructor(public readonly url: string) {
    FakeSocket.instances.push(this);
    queueMicrotask(() => {
      this.readyState = FakeSocket.OPEN;
      this.onopen?.({});
    });
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void {
    this.onclose?.({});
  }
}

beforeEach(() => {
  FakeSocket.instances = [];
  (globalThis as unknown as { WebSocket: typeof FakeSocket }).WebSocket = FakeSocket;
  vi.useFakeTimers();
});

afterEach(() => {
  // Undo the ``Math.random`` pin below so it can't leak into whatever
  // else shares this worker.
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("WsClient reconnect", () => {
  it("emits closed status when the underlying socket closes", async () => {
    const statuses: string[] = [];
    const client = new WsClient({
      url: "ws://test",
      onStatus: (status) => statuses.push(status),
    });
    client.connect();
    await Promise.resolve();
    await Promise.resolve();
    FakeSocket.instances[0]?.close();
    expect(statuses).toContain("closed");
    client.close();
  });

  it("schedules a reconnect after a close (backoff window)", async () => {
    // The first backoff is ``2000ms + Math.random() * 0.3 * 2000`` — i.e.
    // anywhere in [2000, 2600). Pinning ``Math.random`` to 1 puts the
    // reconnect at the very top of that window, so advancing to 2600
    // deterministically covers it. Without the pin this test advanced
    // 2500ms and failed on roughly one run in six, whenever the jitter
    // happened to exceed 500ms.
    vi.spyOn(Math, "random").mockReturnValue(1);
    const client = new WsClient({ url: "ws://test" });
    client.connect();
    await Promise.resolve();
    expect(FakeSocket.instances.length).toBe(1);
    FakeSocket.instances[0]?.close();
    await vi.advanceTimersByTimeAsync(2_600);
    expect(FakeSocket.instances.length).toBeGreaterThanOrEqual(2);
    client.close();
  });
});
