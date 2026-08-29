/**
 * Typed WebSocket client.
 *
 * Single transport. Reconnects with jittered backoff. Inbound messages
 * fan out to listeners registered by message type. Outbound messages are
 * envelope-wrapped to match `backend/src/lucidium/api/messages.py`.
 */

import type {
  C2SMessageType,
  C2SPayloadByType,
  S2CMessageType,
  S2CPayloadByType,
} from "../shared/generated/messages";

const PROTOCOL_VERSION = 1;

/** Listener for one server->client message type, receiving that type's
 *  generated payload interface. Erased to the untyped form in the
 *  internal registry — see ``listeners`` below. */
export type WsListener<T extends S2CMessageType = S2CMessageType> = (
  payload: S2CPayloadByType[T],
) => void;

type AnyListener = (payload: unknown) => void;

export interface Envelope {
  type: C2SMessageType | S2CMessageType;
  // Deliberately ``unknown``: outbound this holds a generated C2S
  // payload interface, inbound it is whatever JSON.parse produced.
  // ``on``/``send`` are where the per-type shape is enforced.
  payload: unknown;
  protocol_version?: number;
}

/** What actually arrives on the socket: only ever an ``s2c/...`` type. */
interface InboundEnvelope extends Envelope {
  type: S2CMessageType;
}

export interface WsClientOptions {
  url: string;
  onStatus?: (status: WsStatus) => void;
}

export type WsStatus = "connecting" | "open" | "closed" | "errored";

export class WsClient {
  private socket: WebSocket | null = null;
  // The registry is heterogeneous by construction: each key holds
  // listeners for a different payload type. ``on`` is the typed door
  // in; inside, listeners are stored in their erased form.
  private listeners = new Map<S2CMessageType, Set<AnyListener>>();
  private reconnectAttempt = 0;
  private closed = false;
  // Mutable so a backend that restarts on a DIFFERENT port can be
  // pointed at without recreating the client (which would drop every
  // registered listener). Seeded from the constructor URL.
  private url: string;

  constructor(private readonly options: WsClientOptions) {
    this.url = options.url;
  }

  /** Point the client at a new URL and reconnect immediately. Used when
   * the Electron main process restarts a dead backend on a fresh port:
   * the existing socket (still retrying the old, dead port) is torn down
   * without triggering its auto-reconnect, and a connection to the new
   * URL is opened right away. A no-op-safe reconnect even when the URL is
   * unchanged (a same-port restart still needs a fresh socket). */
  reconnectTo(url: string): void {
    this.url = url;
    this.closed = false;
    this.reconnectAttempt = 0;
    const stale = this.socket;
    this.socket = null;
    if (stale) {
      // Drop the handler so this manual swap doesn't schedule a
      // competing backoff reconnect to the old URL.
      stale.onclose = null;
      try {
        stale.close();
      } catch {
        /* already closing/closed */
      }
    }
    this.connect();
  }

  /** Subscribe to one server->client message type. The listener's
   *  payload is typed from the generated dispatch map, so an unknown
   *  message type or a mistyped field read is a compile error. */
  on<T extends S2CMessageType>(messageType: T, listener: WsListener<T>): () => void {
    let set = this.listeners.get(messageType);
    if (!set) {
      set = new Set();
      this.listeners.set(messageType, set);
    }
    // Erasing to AnyListener is the one unavoidable step: TypeScript
    // has no way to express a Map whose value type depends on its key.
    // ``on``'s signature is what keeps the two sides in agreement.
    const erased = listener as AnyListener;
    set.add(erased);
    return () => set?.delete(erased);
  }

  /** Send a client->server message. ``messageType`` must be one of the
   *  generated ``c2s/...`` literals and ``payload`` must match that
   *  type's generated interface — a typo in either is a compile error. */
  send<T extends C2SMessageType>(messageType: T, payload: C2SPayloadByType[T]): void {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      // We can't deliver the command. Throwing would tear down React
      // effects, so log it AND fan a synthetic ``s2c/error`` out to
      // the listeners — that releases ``useOptimisticAction``'s
      // pending guard so the affected button reverts to enabled
      // instead of staying locked while the banner says
      // "reconnecting". Without this synthetic, a click during a
      // disconnect locks the choices forever and looks like the
      // dialog "reset to the last choice" without explanation.
       
      console.warn(`[ws] dropping ${messageType}: not connected`);
      const errListeners = this.listeners.get("s2c/error");
      if (errListeners) {
        const fake: S2CPayloadByType["s2c/error"] = {
          code: "ws_disconnected",
          message: `not connected — dropped ${messageType}`,
          recoverable: true,
        };
        for (const listener of errListeners) listener(fake);
      }
      return;
    }
    const envelope: Envelope = {
      type: messageType,
      payload,
      protocol_version: PROTOCOL_VERSION,
    };
    this.socket.send(JSON.stringify(envelope));
  }

  connect(): void {
    if (this.closed) return;
    this.options.onStatus?.("connecting");
    const socket = new WebSocket(this.url);
    this.socket = socket;

    socket.onopen = () => {
      this.reconnectAttempt = 0;
      this.options.onStatus?.("open");
      this.send("c2s/hello", { protocol_version: PROTOCOL_VERSION });
    };

    socket.onmessage = (event) => {
      const raw = typeof event.data === "string" ? event.data : "";
      // Inbound is always server->client, so the type narrows to the
      // s2c half of the union — which is what the listener map is keyed
      // on. An unrecognised type simply finds no listeners below.
      let envelope: InboundEnvelope;
      try {
        envelope = JSON.parse(raw) as InboundEnvelope;
      } catch {
        return;
      }
      const listeners = this.listeners.get(envelope.type);
      if (!listeners) return;
      for (const listener of listeners) listener(envelope.payload);
    };

    socket.onerror = () => {
      this.options.onStatus?.("errored");
    };

    socket.onclose = () => {
      this.options.onStatus?.("closed");
      this.socket = null;
      if (!this.closed) this.scheduleReconnect();
    };
  }

  close(): void {
    this.closed = true;
    this.socket?.close();
    this.socket = null;
  }

  private scheduleReconnect(): void {
    const attempt = ++this.reconnectAttempt;
    const base = Math.min(1000 * 2 ** Math.min(attempt, 6), 30_000);
    const jitter = Math.random() * 0.3 * base;
    setTimeout(() => this.connect(), base + jitter);
  }
}
