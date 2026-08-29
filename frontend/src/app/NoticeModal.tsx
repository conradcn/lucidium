/**
 * Modal pop-up for safety / policy events that the player needs to
 * read.
 *
 * Renders the FIRST queued notice from ``store.notices`` as a
 * fullscreen-blocking dialog. Dismissing it pops it off the queue
 * so the next one (if any) shows. Notices arrive via the
 * ``s2c/notice`` WebSocket message — see ``app/client.ts``.
 *
 * The modal intentionally takes focus and blocks the rest of the
 * UI: a safety-side age correction (the producer of the only
 * current notice) needs to register with the player before they
 * keep playing. A toast-style ephemeral banner would let the
 * notice scroll off-screen unread.
 */

import type { JSX } from "react";
import { useEffect, useRef } from "react";

import { useLucidiumStore } from "../state/store";

export function NoticeModal(): JSX.Element | null {
  const notices = useLucidiumStore((s) => s.notices);
  const dismiss = useLucidiumStore((s) => s.dismissNotice);
  const dismissButtonRef = useRef<HTMLButtonElement | null>(null);
  const current = notices[0] ?? null;

  // Pull focus to the dismiss button when a notice opens — keyboard
  // users get an immediate close affordance and an Escape press
  // closes the dialog.
  useEffect(() => {
    if (!current) return;
    dismissButtonRef.current?.focus();
  }, [current]);

  useEffect(() => {
    if (!current) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") {
        e.preventDefault();
        dismiss(current.id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [current, dismiss]);

  if (!current) return null;

  const accent =
    current.kind === "warning"
      ? "var(--accent-warning, #c97a3a)"
      : "var(--accent, #6aa6ff)";

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="notice-modal-title"
      data-testid="notice-modal"
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.55)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
      }}
    >
      <div
        style={{
          maxWidth: "32rem",
          background: "var(--bg, #1c1c1c)",
          color: "var(--text, #eee)",
          padding: "1.5rem",
          borderRadius: "0.5rem",
          borderTop: `4px solid ${accent}`,
          boxShadow: "0 16px 48px rgba(0,0,0,0.45)",
        }}
      >
        <h2
          id="notice-modal-title"
          style={{ margin: "0 0 0.75rem", fontSize: "1.15rem" }}
        >
          {current.title}
        </h2>
        <p style={{ margin: "0 0 1rem", lineHeight: 1.55, fontSize: "0.95rem" }}>
          {current.body}
        </p>
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button
            ref={dismissButtonRef}
            type="button"
            onClick={() => dismiss(current.id)}
            data-testid="notice-modal-dismiss"
          >
            OK
          </button>
        </div>
      </div>
    </div>
  );
}
