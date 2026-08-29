import type { JSX } from "react";
import { useState } from "react";

import { send } from "../../../app/client";
import { useLucidiumStore } from "../../../state/store";
import { AutoGrowTextarea } from "./AutoGrowTextarea";

interface WorldShape {
  music_path?: string | null;
  music_prompt?: string;
  music_prompt_hash?: string;
}

export function MusicTab(): JSX.Element {
  const game = useLucidiumStore((s) => s.game) as
    | { world?: WorldShape }
    | null;
  const [draftPrompt, setDraftPrompt] = useState<string | null>(null);

  if (!game?.world) return <p>(no game loaded)</p>;
  const world = game.world;
  const livePrompt = world.music_prompt ?? "";
  const promptValue = draftPrompt ?? livePrompt;
  const filename = world.music_path
    ? world.music_path.split(/[\\/]/).pop() ?? world.music_path
    : null;

  const onRegenerate = (): void => {
    const prompt = (draftPrompt ?? "").trim();
    send("c2s/music/regenerate", {
      prompt: prompt && prompt !== livePrompt ? prompt : "",
    });
    // Clear the draft so the field reverts to live state once
    // the backend echoes the new prompt back.
    setDraftPrompt(null);
  };

  return (
    <div>
      <p style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
        Background music for the active scene. ACE-Step renders one
        looping track at a time; regenerating replaces it.
      </p>
      <div
        style={{
          padding: "0.5rem",
          marginBottom: "1rem",
          border: "1px dashed var(--border)",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            gap: "0.5rem",
            alignItems: "baseline",
          }}
        >
          <strong>{filename ?? "(no track yet)"}</strong>
          {world.music_prompt_hash ? (
            <span style={{ color: "var(--muted)", fontSize: "0.75rem" }}>
              {world.music_prompt_hash.slice(0, 8)}
            </span>
          ) : null}
        </div>
        <div
          style={{
            color: "var(--muted)",
            fontSize: "0.85rem",
            marginTop: "0.4rem",
          }}
        >
          Prompt
        </div>
        <AutoGrowTextarea
          value={promptValue}
          onChange={(next) => setDraftPrompt(next)}
          ariaLabel="music prompt"
        />
        <div
          style={{ display: "flex", gap: "0.5rem", marginTop: "0.4rem" }}
        >
          <button
            onClick={onRegenerate}
            disabled={
              draftPrompt !== null && draftPrompt.trim().length === 0
            }
          >
            Regenerate
          </button>
          {draftPrompt !== null && draftPrompt !== livePrompt ? (
            <button onClick={() => setDraftPrompt(null)}>Reset</button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
