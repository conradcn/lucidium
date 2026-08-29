import type { JSX } from "react";
import { useEffect, useState } from "react";

import { ensureClient } from "./client";
import { useLucidiumStore, type WsStatus } from "../state/store";

/**
 * App-level banner that's visible whenever the WebSocket isn't open.
 *
 * Lives at every phase (start screen, interview, load, settings, main)
 * so a stuck connection cannot be mistaken for a "blank screen". Adds
 * itself to the DOM with ``data-testid="connection-banner"`` so tests
 * can assert visibility.
 */
export function ConnectionBanner(): JSX.Element | null {
  const status = useLucidiumStore((s) => s.status);
  const [providerError, setProviderError] = useState<string | null>(null);
  const [creditsError, setCreditsError] = useState<string | null>(null);
  const [stuck, setStuck] = useState(false);

  useEffect(() => {
    const client = ensureClient();
    return client.on("s2c/error", (p) => {
      if (p.code === "provider_credits") {
        // Distinct state — credits errors get their own banner
        // wording with a link to the top-up page. Clear any
        // generic provider banner so the player isn't seeing two
        // stacked alerts.
        setCreditsError(p.message);
        setProviderError(null);
        return;
      }
      if (p.code === "provider_unreachable") setProviderError(p.message);
      if (p.code === "schema_error" || p.code === "internal") setProviderError(p.message);
    });
  }, []);

  // Promote a long-lived connecting/closed state to "stuck" after 5 s
  // so users get a stronger cue than a flicker on initial connect.
  //
  // Clearing the flag on reconnect is a render-phase adjustment (the
  // "you might not need an effect" pattern) rather than a setState in
  // the effect body: it lands before the banner is painted, so a
  // reconnect never shows one frame of the stuck wording.
  const [seenStatus, setSeenStatus] = useState(status);
  if (seenStatus !== status) {
    setSeenStatus(status);
    if (status === "open") setStuck(false);
  }
  useEffect(() => {
    if (status === "open") return;
    const t = setTimeout(() => setStuck(true), 5_000);
    return () => clearTimeout(t);
  }, [status]);

  if (status === "open" && providerError === null && creditsError === null) {
    return null;
  }

  // Credits banner gets its own colour (warning amber) and a
  // clickable link to the top-up page so the player can act
  // without leaving the app.
  if (creditsError) {
    return (
      <div
        role="alert"
        data-testid="connection-banner"
        data-status="credits"
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 1000,
          background: "rgba(150, 100, 25, 0.95)",
          color: "#fff",
          padding: "0.6rem 1rem",
          textAlign: "center",
          font: "inherit",
        }}
      >
        {creditsError}{" "}
        <a
          href="https://openrouter.ai/credits"
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: "#ffe9b3", textDecoration: "underline" }}
        >
          Top up at openrouter.ai/credits →
        </a>
        <button
          onClick={() => setCreditsError(null)}
          style={{
            marginLeft: "1rem",
            background: "transparent",
            border: "1px solid rgba(255,255,255,0.35)",
            color: "#fff",
            padding: "0.1rem 0.6rem",
            cursor: "pointer",
          }}
        >
          Dismiss
        </button>
      </div>
    );
  }

  return (
    <div
      role="alert"
      data-testid="connection-banner"
      data-status={status}
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 1000,
        background: "rgba(120, 30, 30, 0.92)",
        color: "#fff",
        padding: "0.5rem 1rem",
        textAlign: "center",
        font: "inherit",
      }}
    >
      {bannerMessage(status, stuck, providerError)}
    </div>
  );
}

function bannerMessage(status: WsStatus, stuck: boolean, providerError: string | null): string {
  if (providerError) {
    return `Backend error: ${providerError}. Open Options to fix the connection.`;
  }
  if (status === "connecting") {
    return stuck
      ? "Still connecting to the backend… check that start.ps1 launched it."
      : "Connecting to the backend…";
  }
  if (status === "errored") {
    return "WebSocket connection errored. Restart the app or check your settings.";
  }
  if (status === "closed") {
    return "Backend disconnected. Reconnecting…";
  }
  return "";
}
