/**
 * Singleton WebSocket client wired to the Zustand store.
 *
 * Exposed so screens can call ``send(...)`` directly without prop-
 * drilling. Module side effects are intentionally absent — call
 * ``ensureClient()`` at app startup.
 */

import { WsClient } from "../ws/client";
import { useLucidiumStore } from "../state/store";
import type {
  C2SMessageType,
  C2SPayloadByType,
} from "../shared/generated/messages";

let client: WsClient | null = null;

export function ensureClient(): WsClient {
  if (client) return client;
  const port = readWsPort();
  const url = wsUrl(port, readWsToken());
  client = new WsClient({
    url,
    onStatus: (status) => useLucidiumStore.getState().setStatus(status),
  });

  client.on("s2c/hello", (p) => {
    useLucidiumStore.getState().setHasSave(p.has_save);
  });

  client.on("s2c/saves/list", (p) => {
    // Keep hasSave fresh as the player commits new turns; otherwise
    // the StartScreen Continue button would only ever reflect the
    // boolean from the initial s2c/hello.
    useLucidiumStore.getState().setHasSave((p.saves ?? []).length > 0);
  });

  client.on("s2c/state/full", (p) => {
    useLucidiumStore.getState().applyFullState(p.game, p.settings);
    // Side-channel for e2e tests: expose the current node id so tests
    // can detect actual advances rather than typewriter growth. The
    // cast writes a property that exists only for the test harness,
    // so there is no declared type to narrow to.
    const w = window as unknown as { __lucidium_current_node_id?: unknown };
    w.__lucidium_current_node_id = p.game?.current_node_id;
  });

  client.on("s2c/state/patch", (p) => {
    useLucidiumStore.getState().applyPatch(p.ops);
    // Mirror current_node_id changes for the e2e test side-channel.
    for (const op of p.ops) {
      if (op.path === "/current_node_id") {
        // Same test-only side-channel as above — see the s2c/state/full
        // handler for why this is a bare structural cast.
        const w = window as unknown as { __lucidium_current_node_id?: unknown };
        w.__lucidium_current_node_id = op.value;
      }
    }
  });

  client.on("s2c/image/ready", (p) => {
    // Belt-and-braces: the backend already mutates character.images
    // and emits s2c/state/full whenever a render completes, but a
    // race between a slow state/full (built from a pre-render game
    // snapshot) and a fast image_ready event can leave the renderer
    // showing a stale portrait. We forward image_ready events into
    // the store so the most-recent render always wins, even if it
    // arrived for a "future" beat the player hasn't walked yet.
    useLucidiumStore.getState().applyImageReady(p);
  });

  client.on("s2c/error", (p) => {
    // Surface backend errors to the dev tools / Electron console; the
    // ConnectionBanner picks up provider_unreachable for the user.
     
    console.warn(`[ws] s2c/error ${p.code ?? "unknown"}: ${p.message ?? ""}`);
  });

  client.on("s2c/notice", (p) => {
    // Modal pop-up for safety / policy events. The store renders
    // the first queued notice as a blocking modal until the player
    // dismisses it. Currently the only producer is the minor-nudity
    // age-correction guard — see backend
    // ``orchestration/assets.py::check_minor_nudity_and_correct``.
    useLucidiumStore.getState().pushNotice({
      title: p.title,
      body: p.body,
      kind: p.kind === "warning" ? "warning" : "info",
    });
  });

  // When the Electron main process resurrects a backend that died
  // mid-session, it fires ``lucidium:backend-restarted`` with the newly
  // bound port. Point the client at it and reconnect immediately —
  // without this the client keeps retrying the old (now dead) port,
  // since it reads the port only once at boot.
  // The restarted backend minted a FRESH bearer token, so the reconnect
  // has to carry that one — the old token is dead with the old process.
  // Structural cast, justified: ``window.lucidium`` is injected by the
  // Electron preload at runtime and has no ambient declaration.
  const bridge = window as unknown as {
    lucidium?: {
      onBackendRestarted?: (
        cb: (info: { port: number; token: string | null }) => void,
      ) => void;
    };
  };
  bridge.lucidium?.onBackendRestarted?.((info) => {
    const nextUrl = info?.port ? wsUrl(info.port, info.token ?? null) : url;
    client?.reconnectTo(nextUrl);
  });

  client.connect();
  return client;
}

/** Build the backend URL, carrying the per-launch bearer token as a
 * query param. The WebSocket API gives no way to set request headers, so
 * the query string is the only channel a renderer has — the backend
 * accepts it there (and ``Authorization: Bearer`` from non-browser
 * clients). Without a token the handshake is answered with 401, which
 * surfaces as a failed connection in the banner. */
function wsUrl(port: number | null, token: string | null): string {
  const base = `ws://127.0.0.1:${port ?? 8765}`;
  return token ? `${base}/?token=${encodeURIComponent(token)}` : base;
}

function readWsPort(): number | null {
  // Primary: Electron's main.ts passes ``?wsPort=<port>`` on the URL it
  // loads. This is synchronous, contextIsolation-safe, and works for
  // both file:// and http:// schemes.
  try {
    const params = new URLSearchParams(window.location.search);
    const value = params.get("wsPort");
    if (value !== null) {
      const port = Number.parseInt(value, 10);
      if (Number.isFinite(port) && port > 0) return port;
    }
  } catch {
    /* window.location may not exist in some test runners */
  }
  // Fallback: Electron preload may have stashed the port on
  // ``window.lucidium.wsPortSync`` before the renderer scripts ran.
  // Structural cast, justified: preload-injected, see ``ensureClient``.
  const w = window as unknown as { lucidium?: { wsPortSync?: number } };
  if (typeof w.lucidium?.wsPortSync === "number" && w.lucidium.wsPortSync > 0) {
    return w.lucidium.wsPortSync;
  }
  return null;
}

/** Read the per-launch bearer token, same two channels as the port:
 * ``?wsToken=`` on the URL Electron loaded (synchronous, the normal
 * path), falling back to a value the preload stashed on ``window``. */
function readWsToken(): string | null {
  try {
    const params = new URLSearchParams(window.location.search);
    const value = params.get("wsToken");
    if (value) return value;
  } catch {
    /* window.location may not exist in some test runners */
  }
  // Structural cast, justified: preload-injected, see ``ensureClient``.
  const w = window as unknown as { lucidium?: { wsTokenSync?: string } };
  if (typeof w.lucidium?.wsTokenSync === "string" && w.lucidium.wsTokenSync) {
    return w.lucidium.wsTokenSync;
  }
  return null;
}

/**
 * Send a client->server message.
 *
 * ``messageType`` is constrained to the literals the backend actually
 * dispatches on (``C2S_PAYLOAD_BY_TYPE`` in ``api/messages.py``, via the
 * generated ``C2SPayloadByType``), and the payload must match that
 * type's generated interface. ``send("c2s/play/advnace", {})`` and
 * ``send("c2s/saves/load", {})`` are both compile errors.
 *
 * The variadic rest parameter makes the payload optional exactly when
 * every field of it is optional — ``send("c2s/saves/list")`` is fine,
 * ``send("c2s/saves/load")`` is not.
 */
export function send<T extends C2SMessageType>(
  messageType: T,
  ...rest: EmptyObject extends C2SPayloadByType[T]
    ? [payload?: C2SPayloadByType[T]]
    : [payload: C2SPayloadByType[T]]
): void {
  // The conditional tuple above is opaque inside the function body —
  // TypeScript cannot see that ``rest[0]`` is ``C2SPayloadByType[T] |
  // undefined`` for an unresolved ``T``. Narrowing it back is the one
  // assertion this signature costs, and it is sound by construction:
  // both branches of the conditional have that element type.
  const [payload] = rest as [C2SPayloadByType[T]?];
  ensureClient().send(messageType, payload ?? ({} as C2SPayloadByType[T]));
}

/** The empty object type. ``EmptyObject extends P`` holds exactly when
 *  every field of ``P`` is optional — that is the "may the caller omit
 *  the payload?" test. Named rather than written as a bare ``{}``,
 *  which reads as a typo and trips ``@typescript-eslint/ban-types``. */
type EmptyObject = Record<never, never>;
