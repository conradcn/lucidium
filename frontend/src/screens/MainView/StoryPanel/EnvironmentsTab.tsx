import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";

import { assetUrl } from "../../../app/assetUrl";
import { send } from "../../../app/client";
import { useLucidiumStore, type Game } from "../../../state/store";
import type { Environment } from "../../../shared/generated/Game.schema";
import type { C2SEditEnvironment } from "../../../shared/generated/C2SEditEnvironment.schema";
import { AutoGrowTextarea } from "./AutoGrowTextarea";

export function EnvironmentsTab(): JSX.Element {
  const game = useLucidiumStore((s) => s.game);

  const [drafts, setDrafts] = useState<Record<string, Partial<Environment>>>({});

  // Autosave per environment field with a 400 ms debounce
  // window — same pattern as CharactersTab. Per-(env, field)
  // timer keys so editing the prompt doesn't postpone the
  // location_label commit. The draft entry clears once the
  // patch fires and the input snaps to whatever the backend
  // round-tripped back.
  //
  // The ref and its cleanup effect must sit ABOVE the "no game
  // loaded" early return: hooks have to run in the same order on
  // every render, and this component unmounts/remounts its content
  // as games load and unload.
  const autosaveTimersRef = useRef<Record<string, number>>({});
  useEffect(() => {
    // Snapshot the map identity now; the ref object may have been
    // swapped by the time the cleanup runs.
    const timers = autosaveTimersRef.current;
    return () => {
      Object.values(timers).forEach((id) => window.clearTimeout(id));
    };
  }, []);

  const draftFor = (envId: string): Partial<Environment> => drafts[envId] ?? {};

  const scheduleAutosave = (
    envId: string,
    field: keyof Environment,
    value: C2SEditEnvironment["value"],
  ): void => {
    const key = `${envId}|${String(field)}`;
    const existing = autosaveTimersRef.current[key];
    if (existing) window.clearTimeout(existing);
    autosaveTimersRef.current[key] = window.setTimeout(() => {
      send("c2s/edit/environment", {
        environment_id: envId,
        field,
        value,
      });
      delete autosaveTimersRef.current[key];
      setDrafts((prev) => {
        const copy = { ...prev };
        const envDraft = copy[envId];
        if (envDraft) {
          const envCopy = { ...envDraft };
          delete envCopy[field];
          copy[envId] = envCopy;
        }
        return copy;
      });
    }, 400);
  };

  if (!game?.environments) return <p>(no game loaded)</p>;

  const activeId = activeEnvironmentId(game);

  return (
    <div>
      {/* Iterate entries, not values: the environment id is the map
          key, and the generated Environment marks its own ``id`` optional
          (Pydantic default_factory). The key is always present. */}
      {Object.entries(game.environments).map(([envId, env]) => {
        const draft = draftFor(envId);
        const isActive = envId === activeId;
        return (
          <div
            key={envId}
            style={{
              borderLeft: isActive ? "3px solid var(--accent)" : "3px solid transparent",
              padding: "0.5rem",
              marginBottom: "0.75rem",
            }}
          >
            <div style={{ color: isActive ? "var(--accent)" : "var(--muted)", fontSize: "0.85rem" }}>
              {isActive ? "ACTIVE" : ""}
            </div>
            <input
              type="text"
              value={draft.location_label ?? env.location_label}
              onChange={(e) => {
                const next = e.target.value;
                setDrafts((prev) => ({
                  ...prev,
                  [envId]: { ...draftFor(envId), location_label: next },
                }));
                scheduleAutosave(envId, "location_label", next);
              }}
            />
            <AutoGrowTextarea
              value={draft.prompt ?? env.prompt}
              onChange={(next) => {
                setDrafts((prev) => ({
                  ...prev,
                  [envId]: { ...draftFor(envId), prompt: next },
                }));
                scheduleAutosave(envId, "prompt", next);
              }}
              ariaLabel={`environment ${envId} prompt`}
            />
            <div style={{ display: "flex", gap: "0.25rem", marginTop: "0.25rem" }}>
              <button
                onClick={() =>
                  send("c2s/edit/environment/rerender", {
                    environment_id: envId,
                  })
                }
                title="Generate a fresh background with a new seed"
              >
                Rerender
              </button>
              {/* Apply: switch the current scene's backdrop to this
                  environment. Hidden on the already-active env (its
                  image is what the player sees right now) and on
                  envs without a rendered image yet (no backdrop to
                  apply). Backend handler mutates the current node's
                  location_id; the renderer's location-walk picks up
                  this env on the next state/patch tick. */}
              {!isActive && env.image_path ? (
                <button
                  onClick={() =>
                    send("c2s/edit/environment/apply", {
                      environment_id: envId,
                    })
                  }
                  title="Switch the current scene's backdrop to this environment"
                  data-testid={`apply-env-${envId}`}
                >
                  Apply as backdrop
                </button>
              ) : null}
            </div>
            {env.image_path ? (
              <img
                src={assetUrl(env.image_path) ?? ""}
                alt={env.location_label}
                style={{ width: "100%", marginTop: "0.5rem" }}
              />
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function activeEnvironmentId(game: Game): string | null {
  if (!game.dialog_tree) return null;
  const path = game.dialog_tree.committed_path ?? [];
  const nodes = game.dialog_tree.nodes ?? {};
  for (let i = path.length - 1; i >= 0; i--) {
    const id = path[i] as string;
    const node = nodes[id];
    if (node?.location_id) return node.location_id;
  }
  return null;
}
