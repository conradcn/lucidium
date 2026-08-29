import type { JSX } from "react";
import { useEffect, useState } from "react";

import { ensureClient, send } from "../app/client";

interface SaveSummary {
  id: string;
  name: string;
  last_played_at: string;
  created_at: string;
  schema_version: number;
  summary: string;
}

interface Props {
  onBack: () => void;
}

export function LoadGameScreen({ onBack }: Props): JSX.Element {
  const [saves, setSaves] = useState<SaveSummary[]>([]);
  const [renaming, setRenaming] = useState<{ id: string; value: string } | null>(null);

  useEffect(() => {
    const client = ensureClient();
    const off = client.on("s2c/saves/list", (p) => {
      setSaves(p.saves ?? []);
    });
    send("c2s/saves/list");
    return off;
  }, []);

  const onLoad = (id: string): void => {
    send("c2s/saves/load", { save_id: id });
  };

  const onDelete = (id: string): void => {
    send("c2s/saves/delete", { save_id: id });
  };

  const onCommitRename = (): void => {
    if (!renaming) return;
    send("c2s/saves/rename", { save_id: renaming.id, new_name: renaming.value });
    setRenaming(null);
  };

  return (
    <div
      className="interview load-game-screen"
      data-testid="load-game-screen"
    >
      <div className="page-frame">
        <div className="page-content">
          <h2 className="page-title">Load game</h2>
          {saves.length === 0 ? (
            <p className="page-subtitle">No saves yet.</p>
          ) : (
            <ul className="save-list">
              {saves.map((save) => (
                <li key={save.id}>
                  {renaming?.id === save.id ? (
                    <>
                      <input
                        type="text"
                        value={renaming.value}
                        onChange={(e) =>
                          setRenaming({ id: save.id, value: e.target.value })
                        }
                      />
                      <button onClick={onCommitRename}>Save</button>
                      <button onClick={() => setRenaming(null)}>Cancel</button>
                    </>
                  ) : (
                    <>
                      <div className="save-meta">
                        <div className="save-name">{save.name}</div>
                        <div className="save-summary">
                          {save.summary || "(no summary)"}
                        </div>
                        <div className="save-time">
                          {formatSaveTime(save.last_played_at)}
                        </div>
                      </div>
                      {/* Buttons grouped so they wrap as a unit on
                          narrow viewports — meta keeps its column,
                          actions drop to a new row beneath when
                          there isn't horizontal room. */}
                      <div className="save-actions">
                        <button onClick={() => onLoad(save.id)}>Load</button>
                        <button
                          onClick={() =>
                            setRenaming({ id: save.id, value: save.name })
                          }
                        >
                          Rename
                        </button>
                        <button onClick={() => onDelete(save.id)}>
                          Delete
                        </button>
                      </div>
                    </>
                  )}
                </li>
              ))}
            </ul>
          )}
          <div className="button-row">
            <button onClick={onBack}>Back</button>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatSaveTime(iso: string): string {
  if (!iso) return "";
  const ts = Date.parse(iso);
  if (Number.isNaN(ts)) return iso;
  const date = new Date(ts);
  const diffMs = Date.now() - ts;
  const dayMs = 24 * 60 * 60 * 1000;
  if (diffMs < 60_000) return "just now";
  if (diffMs < 3_600_000) {
    const mins = Math.round(diffMs / 60_000);
    return `${mins} min ago`;
  }
  if (diffMs < dayMs) {
    const hrs = Math.round(diffMs / 3_600_000);
    return `${hrs} hr ago`;
  }
  if (diffMs < 7 * dayMs) {
    const days = Math.round(diffMs / dayMs);
    return `${days} day${days === 1 ? "" : "s"} ago`;
  }
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
