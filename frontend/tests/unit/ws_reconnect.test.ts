import { describe, expect, it, beforeEach, vi } from "vitest";

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
    const client = new WsClient({ url: "ws://test" });
    client.connect();
    await Promise.resolve();
    expect(FakeSocket.instances.length).toBe(1);
    FakeSocket.instances[0]?.close();
    await vi.advanceTimersByTimeAsync(2_500);
    expect(FakeSocket.instances.length).toBeGreaterThanOrEqual(2);
    client.close();
  });
});
