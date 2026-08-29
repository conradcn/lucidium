import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";

import { send } from "../../../app/client";
import { useLucidiumStore } from "../../../state/store";
import { AutoGrowTextarea } from "./AutoGrowTextarea";

interface DialogNode {
  id: string;
  text: string | null;
  speaker_id: string | null;
}

export function HistoryTab(): JSX.Element {
  const game = useLucidiumStore((s) => s.game) as
    | {
        current_node_id?: string | null;
        characters?: Record<string, { name: string }>;
        dialog_tree?: {
          nodes: Record<string, DialogNode>;
          committed_path: string[];
          root_id?: string | null;
        };
      }
    | null;
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  // Global retcon: one input that applies to all history at once.
  const [retconInstructions, setRetconInstructions] = useState("");
  const [retconPending, setRetconPending] = useState(false);

  // Per-node debounced autosave for inline history rewrites.
  // The retcon path is intentionally separate — it's an
  // LLM-driven rewrite of the whole history, expensive enough
  // that it stays gated behind an explicit "Apply retcon"
  // button so the player never accidentally fires it.
  //
  // The ref and its cleanup effect must sit ABOVE the "no game
  // loaded" early return so hook order is identical on every
  // render, including when a game loads or unloads.
  const autosaveTimersRef = useRef<Record<string, number>>({});
  useEffect(() => {
    // Snapshot the map identity now; the ref object may have been
    // swapped by the time the cleanup runs.
    const timers = autosaveTimersRef.current;
    return () => {
      Object.values(timers).forEach((id) => window.clearTimeout(id));
    };
  }, []);

  const scheduleAutosave = (nodeId: string, value: string): void => {
    const existing = autosaveTimersRef.current[nodeId];
    if (existing) window.clearTimeout(existing);
    autosaveTimersRef.current[nodeId] = window.setTimeout(() => {
      send("c2s/edit/history", { node_id: nodeId, new_text: value });
      delete autosaveTimersRef.current[nodeId];
      setDrafts((prev) => {
        const copy = { ...prev };
        delete copy[nodeId];
        return copy;
      });
    }, 500);
  };

  const onRetcon = (): void => {
    const instructions = retconInstructions.trim();
    if (instructions.length === 0) return;
    setRetconPending(true);
    send("c2s/edit/history/retcon", { instructions });
    // The handler runs an LLM call and emits state/full when done;
    // clear the textbox + pending flag on a fallback timer in case
    // the response is dropped.
    setTimeout(() => {
      setRetconInstructions("");
      setRetconPending(false);
    }, 60_000);
  };

  if (!game?.dialog_tree) return <p>(no game loaded)</p>;
  const path = game.dialog_tree.committed_path ?? [];

  return (
    <div>
      <div
        style={{
          padding: "0.5rem",
          marginBottom: "1rem",
          border: "1px dashed var(--border)",
        }}
      >
        <div style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
          Retcon — rewrite all history + character state to fit one instruction
        </div>
        <AutoGrowTextarea
          value={retconInstructions}
          onChange={setRetconInstructions}
          ariaLabel="global retcon instructions"
        />
        <button
          onClick={onRetcon}
          disabled={retconPending || retconInstructions.trim().length === 0}
          style={{ marginTop: "0.25rem" }}
        >
          {retconPending ? "Rewriting..." : "Apply retcon"}
        </button>
      </div>
      {path.map((id) => {
        const node = game.dialog_tree?.nodes[id];
        if (!node) return null;
        const speaker = node.speaker_id ? game.characters?.[node.speaker_id]?.name : "narrator";
        const draft = drafts[id] ?? node.text ?? "";
        const canDelete =
          id !== game.dialog_tree?.root_id && id !== game.current_node_id;
        return (
          <div key={id} style={{ marginBottom: "0.75rem" }}>
            <div
              style={{
                display: "flex",
                alignItems: "baseline",
                justifyContent: "space-between",
                gap: "0.5rem",
              }}
            >
              <div style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
                {speaker}
              </div>
              {canDelete ? (
                <button
                  onClick={() => send("c2s/edit/history/delete", { node_id: id })}
                  aria-label={`delete history beat ${id}`}
                  style={{
                    background: "transparent",
                    border: "none",
                    color: "var(--muted)",
                    cursor: "pointer",
                    fontSize: "0.85rem",
                    padding: "0 0.25rem",
                  }}
                >
                  ×
                </button>
              ) : null}
            </div>
            <AutoGrowTextarea
              value={draft}
              onChange={(next) => {
                setDrafts((prev) => ({ ...prev, [id]: next }));
                if (!retconPending) scheduleAutosave(id, next);
              }}
              ariaLabel={`history beat ${id}`}
            />
          </div>
        );
      })}
    </div>
  );
}
