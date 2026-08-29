import { describe, expect, it, vi, beforeEach } from "vitest";

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
});

describe("WsClient", () => {
  it("sends c2s/hello on connect", async () => {
    const client = new WsClient({ url: "ws://test" });
    client.connect();
    await new Promise<void>((resolve) => queueMicrotask(() => resolve()));
    const socket = FakeSocket.instances.at(-1);
    expect(socket?.sent.length).toBe(1);
    const envelope = JSON.parse(socket?.sent[0] ?? "{}");
    expect(envelope.type).toBe("c2s/hello");
    client.close();
  });

  it("dispatches inbound envelopes to typed listeners", async () => {
    const client = new WsClient({ url: "ws://test" });
    const listener = vi.fn();
    client.on("s2c/hello", listener);
    client.connect();
    await new Promise<void>((resolve) => queueMicrotask(() => resolve()));
    const socket = FakeSocket.instances.at(-1);
    socket?.onmessage?.({
      data: JSON.stringify({ type: "s2c/hello", payload: { has_save: true }, protocol_version: 1 }),
    });
    expect(listener).toHaveBeenCalledWith({ has_save: true });
    client.close();
  });

  it("ignores malformed inbound messages", async () => {
    const client = new WsClient({ url: "ws://test" });
    const listener = vi.fn();
    client.on("s2c/hello", listener);
    client.connect();
    await new Promise<void>((resolve) => queueMicrotask(() => resolve()));
    const socket = FakeSocket.instances.at(-1);
    socket?.onmessage?.({ data: "{not json" });
    expect(listener).not.toHaveBeenCalled();
    client.close();
  });
});
