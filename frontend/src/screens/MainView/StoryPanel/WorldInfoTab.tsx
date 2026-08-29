import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";

import { send } from "../../../app/client";
import { useLucidiumStore } from "../../../state/store";
import type { WorldState } from "../../../shared/generated/Game.schema";
import { AutoGrowTextarea } from "./AutoGrowTextarea";

/** The free-text world fields the panel exposes for editing. Typed as
 *  keys of the generated WorldState, so a Pydantic rename breaks the
 *  build here rather than silently rendering an empty box. */
const SCALAR_FIELDS: (keyof WorldState)[] = [
  "game_name",
  "setting",
  "genre",
  "visual_style",
  "overall_plot_direction",
  "summarizer_assessment",
];

export function WorldInfoTab(): JSX.Element {
  const world = useLucidiumStore((s) => s.game?.world ?? null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});

  // Per-field debounced autosave — same pattern as the other
  // story-panel tabs. Long pauses between keystrokes commit;
  // a single round-trip per field per typing burst.
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

  const scheduleAutosave = (field: keyof WorldState, value: string): void => {
    const existing = autosaveTimersRef.current[field];
    if (existing) window.clearTimeout(existing);
    autosaveTimersRef.current[field] = window.setTimeout(() => {
      send("c2s/edit/world", { field, value });
      delete autosaveTimersRef.current[field];
      setDrafts((prev) => {
        const copy = { ...prev };
        delete copy[field];
        return copy;
      });
    }, 400);
  };

  if (!world) return <p>(no game loaded)</p>;

  return (
    <div>
      {SCALAR_FIELDS.map((field) => {
        const current = String(world[field] ?? "");
        const draft = drafts[field] ?? current;
        return (
          <label key={field} style={{ display: "block", marginBottom: "0.75rem" }}>
            <div style={{ color: "var(--muted)", fontSize: "0.85rem" }}>{field}</div>
            <AutoGrowTextarea
              value={draft}
              onChange={(next) => {
                setDrafts((prev) => ({ ...prev, [field]: next }));
                scheduleAutosave(field, next);
              }}
              ariaLabel={`world ${field}`}
            />
          </label>
        );
      })}
      <h3>Active plot threads</h3>
      <ul>
        {(world.active_plot_threads ?? []).map((thread) => (
          <li key={thread.id}>
            <strong>{thread.title}</strong>
            <div style={{ color: "var(--muted)", fontSize: "0.85rem" }}>{thread.summary}</div>
          </li>
        ))}
      </ul>
      <h3>Dropped plot threads</h3>
      <ul>
        {(world.dropped_plot_threads ?? []).map((thread) => (
          <li key={thread.id}>
            <strong>{thread.title}</strong>
            <div style={{ color: "var(--muted)", fontSize: "0.85rem" }}>{thread.summary}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
