import { contextBridge, ipcRenderer } from "electron";

const lucidium = {
  getWsPort: async (): Promise<number | null> => {
    return (await ipcRenderer.invoke("lucidium:get-ws-port")) as number | null;
  },
  /** The per-launch bearer token the backend announced on stdout. Every
   * WebSocket handshake must carry it (as ``?token=``) or the backend
   * answers 401 — that's what stops any other local process, or a page
   * in the player's browser, from driving the engine. */
  getWsToken: async (): Promise<string | null> => {
    return (await ipcRenderer.invoke("lucidium:get-ws-token")) as string | null;
  },
  /** Tear down the backend cleanly and restart the Electron process so
   * the next launch picks up a freshly-installed torch overlay (or
   * any other state that only takes effect at process start). Fire-
   * and-forget: the current process exits, so the returned promise
   * never resolves to anything callers should observe. */
  relaunchForOverlay: (): void => {
    ipcRenderer.send("lucidium:relaunch-for-overlay");
  },
  /** Subscribe to backend auto-restarts. The main process fires this
   * with the freshly-bound port AND the new launch's bearer token after
   * it resurrects a backend that died mid-session; the WS client swaps
   * to both and reconnects. The token matters as much as the port — a
   * restarted backend mints a new one, so reconnecting with the old
   * token would be rejected. Returns an unsubscribe function. */
  onBackendRestarted: (
    cb: (info: { port: number; token: string | null }) => void,
  ): (() => void) => {
    const listener = (
      _event: unknown,
      info: { port: number; token: string | null },
    ): void => cb(info);
    ipcRenderer.on("lucidium:backend-restarted", listener);
    return () => {
      ipcRenderer.removeListener("lucidium:backend-restarted", listener);
    };
  },
};

contextBridge.exposeInMainWorld("lucidium", lucidium);

declare global {
  interface Window {
    lucidium: typeof lucidium;
  }
}
