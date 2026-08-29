import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";

import { assetUrl } from "../../../app/assetUrl";
import { send } from "../../../app/client";
import { useLucidiumStore } from "../../../state/store";
import { AutoGrowTextarea } from "./AutoGrowTextarea";

// ``seed`` is editable so a player who wants a different draw of
// the same character can change the seed manually — rerender no
// longer bumps the seed because the seed is part of character
// identity and bumping it on every rerender broke face
// consistency across automatic regenerations. Numeric fields are
// listed here too so the save path coerces the input value to a
// Number; an unhandled string would either silently fail
// pydantic validation or get coerced unexpectedly.
// ``physical_description`` ships above the human-anatomy block so a
// player who flips a character to ``kind=nonhuman`` lands on the
// freeform field they're now expected to fill, rather than scrolling
// past stale gender/hair/eye lines that no longer apply to the
// renderer's chosen pipeline.
const FIELDS: { key: string; multiline?: boolean; numeric?: boolean }[] = [
  { key: "name" },
  { key: "description", multiline: true },
  { key: "physical_description", multiline: true },
  { key: "gender" },
  // Pronouns sit next to gender so the player can pick a
  // pronoun set independent of gender. Free text — covers
  // "she/her", "they/them", neopronouns, setting-specific
  // forms. Empty falls back to the storyteller deriving
  // pronouns from gender / context (legacy behaviour).
  { key: "pronouns" },
  { key: "age", numeric: true },
  { key: "ethnicity" },
  { key: "skin" },
  { key: "hair_color" },
  { key: "hairstyle" },
  { key: "eye_color" },
  { key: "build" },
  { key: "bust" },
  { key: "outfit" },
  { key: "pose" },
  { key: "expression" },
  // Permanent skin markings (tattoos / scars / birthmarks). Multiline
  // so the player can describe several. Edits flow through the same
  // debounced c2s/edit/character path as every other field and are
  // folded into the portrait prompt on the next render / rerender.
  { key: "decals", multiline: true },
  { key: "seed", numeric: true },
];

interface Character {
  id: string;
  name: string;
  description: string;
  age: number;
  seed: number;
  is_player: boolean;
  removed: boolean;
  removed_reason: string;
  // Branches the portrait pipeline. ``human`` (default) uses the
  // structured anatomy fields; ``nonhuman`` uses the freeform
  // ``physical_description`` and routes through the environment
  // SDXL checkpoint with FaceDetailer skipped.
  kind: "human" | "nonhuman";
  physical_description: string;
  images: { id: string; path: string }[];
  facts: { id: string; text: string; confidence: string }[];
}

interface DialogNodeForSort {
  id: string;
  speaker_id?: string | null;
  entering_character_ids: string[];
  leaving_character_ids: string[];
}

/**
 * Build a map from character id → committed_path index of the
 * latest beat that REFERENCED them — entering, speaking, or being
 * on stage at the time. Walks the path forward, simulating the
 * on_stage set per beat by applying each node's entering / leaving
 * lists; for every beat, every member of the resulting on-stage
 * set AND that beat's speaker get the current index stamped on
 * them.
 *
 * "Most recently referenced" is what the player wants for cast
 * ordering — a character who joined ten beats ago and is still
 * standing in the scene should rank above one mentioned in
 * passing yesterday and gone since. ``speaker_id`` also bumps the
 * recency: a character can technically have left the stage but
 * still be the named speaker in a follow-up beat, and that's a
 * reference the player would expect to count.
 */
function lastReferenceIndex(
  committedPath: string[],
  nodes: Record<string, DialogNodeForSort>,
): Map<string, number> {
  const out = new Map<string, number>();
  const onStage = new Set<string>();
  for (let i = 0; i < committedPath.length; i++) {
    const nodeId = committedPath[i];
    if (nodeId === undefined) continue;
    const node = nodes[nodeId];
    if (!node) continue;
    for (const cid of node.entering_character_ids ?? []) {
      onStage.add(cid);
    }
    for (const cid of node.leaving_character_ids ?? []) {
      onStage.delete(cid);
    }
    for (const cid of onStage) {
      out.set(cid, i);
    }
    const speaker = (node.speaker_id ?? "").trim();
    if (speaker && speaker !== "narrator") {
      out.set(speaker, i);
    }
  }
  return out;
}

/**
 * Return ``drafts`` minus every field whose canonical value has
 * caught up with the typed draft. Returns the original object
 * identity when nothing settled, so React can bail out of the
 * re-render.
 */
function pruneSettledDrafts(
  drafts: Record<string, Record<string, string>>,
  characters: Record<string, Character> | null,
): Record<string, Record<string, string>> {
  let changed = false;
  const next: Record<string, Record<string, string>> = {};
  for (const charId of Object.keys(drafts)) {
    const charDraft = drafts[charId];
    if (!charDraft) continue;
    const canonical = characters?.[charId] as
      | Record<string, unknown>
      | undefined;
    const remaining: Record<string, string> = {};
    for (const field of Object.keys(charDraft)) {
      const draftValue = charDraft[field] ?? "";
      const canonicalValue = canonical ? String(canonical[field] ?? "") : "";
      if (canonicalValue === draftValue) {
        changed = true;
        continue;
      }
      remaining[field] = draftValue;
    }
    if (Object.keys(remaining).length > 0) {
      next[charId] = remaining;
    } else {
      changed = true;
    }
  }
  return changed ? next : drafts;
}

export function CharactersTab(): JSX.Element {
  const game = useLucidiumStore((s) => s.game) as
    | {
        characters?: Record<string, Character>;
        on_stage?: string[];
        dialog_tree?: {
          nodes: Record<string, DialogNodeForSort>;
          committed_path: string[];
        };
      }
    | null;
  const characters = game?.characters ?? null;
  const onStageNow = new Set(game?.on_stage ?? []);
  const [drafts, setDrafts] = useState<Record<string, Record<string, string>>>({});
  // Cards collapse by default — the cast tab can fill the panel
  // with full anatomy / facts / autosave fields, and the player
  // mostly wants to scan WHO is in the cast and which of them is
  // currently on stage. Each card opens to the full editor when
  // its header is clicked. ``expanded`` keeps an explicit
  // ``true`` for cards the player has opened; missing → use the
  // default-collapse behavior (player card excepted, see render).
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  // Drop drafts whose canonical value has caught up. This is the
  // "adjust state when the data changes" render-phase pattern
  // (https://react.dev/learn/you-might-not-need-an-effect) rather
  // than an effect: React re-runs this component immediately with
  // the pruned drafts, before anything is painted, so there is no
  // cascading render and no flash of the stale draft. While a
  // round-trip is in flight, canonical lags behind the draft and we
  // keep the draft so the input reflects what the player typed.
  const [seenCharacters, setSeenCharacters] = useState(characters);
  if (seenCharacters !== characters) {
    setSeenCharacters(characters);
    setDrafts((prev) => pruneSettledDrafts(prev, characters));
  }

  // The autosave ref and its effects must sit ABOVE the "no game
  // loaded" early return: hooks have to run in the same order on
  // every render, and this component's game can load or unload
  // while it is mounted.
  const autosaveTimersRef = useRef<Record<string, number>>({});
  // Flush every pending autosave on unmount so the player's
  // last edit doesn't get dropped when they switch tabs.
  useEffect(() => {
    // Snapshot the map identity now; the ref object may have been
    // swapped by the time the cleanup runs.
    const timers = autosaveTimersRef.current;
    return () => {
      Object.values(timers).forEach((id) => window.clearTimeout(id));
    };
  }, []);

  if (!characters) return <p>(no game loaded)</p>;

  // Sort: player first (always pinned to the top), then NPCs by
  // last-reference committed_path index descending — most recent
  // reference (entering, speaking, or on-stage at that beat) wins.
  // A character who joined twenty beats ago and is still standing
  // in the scene ranks above one mentioned in passing five beats
  // ago but gone since. Characters never referenced — e.g.,
  // off-stage NPCs the storyteller has only set up in a fact —
  // fall to the bottom in a stable id-based tiebreak.
  const lastReference = lastReferenceIndex(
    game?.dialog_tree?.committed_path ?? [],
    game?.dialog_tree?.nodes ?? {},
  );
  const sortedCharacters = [...Object.values(characters)].sort((a, b) => {
    if (a.is_player !== b.is_player) return a.is_player ? -1 : 1;
    const ai = lastReference.get(a.id);
    const bi = lastReference.get(b.id);
    if (ai !== undefined && bi !== undefined) return bi - ai;
    if (ai !== undefined) return -1;
    if (bi !== undefined) return 1;
    return a.id.localeCompare(b.id);
  });

  // Autosave: each field's draft fires a debounced
  // ``c2s/edit/character`` patch when the player stops typing
  // for ~400 ms. Per-field timers (keyed by ``charId|field``)
  // so editing one field doesn't postpone another's commit;
  // long-pause typing on a single field is one round-trip,
  // not one-per-keystroke.
  //
  // CRITICAL: the draft is NOT cleared when the autosave timer
  // fires. If we cleared it there, the input ``value`` would
  // briefly snap from the typed string back to the old
  // canonical value (state/patch from the backend hasn't
  // arrived yet), which forces React to set ``input.value``
  // programmatically, which moves the cursor to the end of
  // the text. After the state/patch arrives the cursor jumps
  // again. Net effect for the player: typing in the MIDDLE of
  // a word silently lands at the END instead, because every
  // autosave fire-and-echo cycle resets the cursor to the
  // tail. Drafts are cleared by the effect below — only once
  // the canonical character state has caught up to the draft
  // value — so the input always renders the in-flight string
  // and React never has to forcibly rewrite input.value.
  const scheduleAutosave = (charId: string, field: string, raw: string): void => {
    const key = `${charId}|${field}`;
    const existing = autosaveTimersRef.current[key];
    if (existing) window.clearTimeout(existing);
    autosaveTimersRef.current[key] = window.setTimeout(() => {
      const fieldDef = FIELDS.find((f) => f.key === field);
      const value = fieldDef?.numeric ? Number(raw) : raw;
      send("c2s/edit/character", { character_id: charId, field, value });
      delete autosaveTimersRef.current[key];
    }, 400);
  };
  // Flush every pending autosave for one character RIGHT NOW,
  // bypassing the 400 ms debounce, and return once each edit has
  // been sent. The Rerender button calls this before it fires
  // ``c2s/edit/character/rerender``: a player who edits ``pose``
  // (or outfit / hair / …) and immediately clicks Rerender would
  // otherwise race the still-pending debounced ``c2s/edit/character``
  // — the rerender reaches the backend FIRST, runs against the
  // stale attributes, computes the unchanged prompt_hash, and gets
  // short-circuited by ``_portrait_already_current`` so no new
  // portrait is produced. Sending the edits first (WebSocket
  // preserves order; the backend processes one message fully
  // before the next) guarantees the rerender sees the new pose.
  const flushAutosaves = (charId: string): void => {
    const prefix = `${charId}|`;
    const charDraft = drafts[charId] ?? {};
    for (const key of Object.keys(autosaveTimersRef.current)) {
      if (!key.startsWith(prefix)) continue;
      window.clearTimeout(autosaveTimersRef.current[key]);
      delete autosaveTimersRef.current[key];
      const field = key.slice(prefix.length);
      const fieldDef = FIELDS.find((f) => f.key === field);
      const raw = charDraft[field] ?? "";
      const value = fieldDef?.numeric ? Number(raw) : raw;
      send("c2s/edit/character", { character_id: charId, field, value });
    }
  };
  return (
    <div>
      {sortedCharacters.map((char) => {
        const charDraft = drafts[char.id] ?? {};
        const portrait = char.images?.[0];
        // Player card defaults to OPEN (it's the one card the
        // player most often looks at to confirm the protagonist
        // line); NPC cards default to CLOSED. Either can be
        // toggled by clicking the header. Once toggled, the
        // explicit value wins.
        const explicit = expanded[char.id];
        const isOpen =
          explicit !== undefined ? explicit : char.is_player;
        const isOnStage = onStageNow.has(char.id) && !char.removed;
        return (
          <div
            key={char.id}
            className={char.removed ? "character-card removed" : "character-card"}
            style={{
              padding: "0.5rem",
              marginBottom: "0.75rem",
              borderTop: "1px solid var(--border)",
              opacity: char.removed ? 0.5 : 1,
            }}
          >
            <h3
              style={{
                marginTop: 0,
                cursor: "pointer",
                userSelect: "none",
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
              }}
              onClick={() =>
                setExpanded((prev) => ({ ...prev, [char.id]: !isOpen }))
              }
              title={isOpen ? "Click to collapse" : "Click to expand"}
            >
              <span
                aria-hidden="true"
                style={{
                  display: "inline-block",
                  width: "0.7rem",
                  textAlign: "center",
                  color: "var(--muted)",
                }}
              >
                {isOpen ? "▾" : "▸"}
              </span>
              <span>{char.name}</span>
              {isOnStage ? (
                <span
                  className="on-stage-badge"
                  title="Currently on stage"
                  aria-label="on stage"
                  style={{
                    fontSize: "0.7rem",
                    padding: "0.1rem 0.4rem",
                    borderRadius: "999px",
                    background: "color-mix(in oklab, var(--accent) 30%, transparent)",
                    border: "1px solid color-mix(in oklab, var(--accent) 60%, transparent)",
                    color: "var(--fg)",
                    letterSpacing: "0.05em",
                    textTransform: "uppercase",
                  }}
                >
                  on stage
                </span>
              ) : null}
              {char.is_player ? (
                <span style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
                  (player)
                </span>
              ) : null}
              {char.removed ? (
                <span style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
                  ({char.removed_reason || "removed"})
                </span>
              ) : null}
            </h3>
            {!isOpen ? (
              <div style={{ clear: "both" }} />
            ) : null}
            {isOpen && portrait ? (
              <img
                src={assetUrl(portrait.path) ?? ""}
                alt={char.name}
                style={{
                  width: "8rem",
                  float: "right",
                  filter: char.removed ? "grayscale(1)" : undefined,
                }}
              />
            ) : null}
            {isOpen && !char.is_player ? (
              <div
                style={{
                  display: "flex",
                  gap: "0.5rem",
                  marginBottom: "0.5rem",
                }}
              >
                <button
                  onClick={() => {
                    // Commit any in-flight field edits (e.g. a pose
                    // the player just typed) BEFORE the rerender, so
                    // the backend renders the new attributes rather
                    // than racing the debounced autosave.
                    flushAutosaves(char.id);
                    send("c2s/edit/character/rerender", { character_id: char.id });
                  }}
                  title="Generate a fresh portrait"
                >
                  Rerender
                </button>
                {char.removed ? (
                  <button
                    onClick={() =>
                      send("c2s/edit/character/show", { character_id: char.id })
                    }
                    title="Bring this character back into the scene"
                  >
                    Show
                  </button>
                ) : (
                  <button
                    onClick={() =>
                      send("c2s/edit/character/dismiss", {
                        character_id: char.id,
                        reason: "dismissed",
                      })
                    }
                    title="Remove this character from the scene and future turns"
                  >
                    Dismiss
                  </button>
                )}
              </div>
            ) : null}
            {/* Kind toggle. Sits above the structured fields so the
                pipeline switch is the FIRST decision the player
                sees — flipping it informs which fields below
                actually drive the render. ``human`` keeps the
                anatomy fields canonical; ``nonhuman`` routes
                through the freeform physical_description (and the
                environment SDXL pipeline). Click writes
                immediately — there's no draft state for an enum
                value because there's nothing to type. */}
            {isOpen ? (
            <label
              style={{
                display: "flex",
                gap: "0.5rem",
                alignItems: "center",
                marginBottom: "0.6rem",
              }}
            >
              <span style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
                kind
              </span>
              <select
                value={char.kind ?? "human"}
                onChange={(e) =>
                  send("c2s/edit/character", {
                    character_id: char.id,
                    field: "kind",
                    value: e.target.value,
                  })
                }
                title={
                  char.kind === "nonhuman"
                    ? "Renders via the freeform physical_description on the environment SDXL pipeline (FaceDetailer skipped)."
                    : "Renders via the structured anatomy fields on the character SDXL pipeline."
                }
              >
                <option value="human">human</option>
                <option value="nonhuman">nonhuman</option>
              </select>
            </label>
            ) : null}
            {isOpen && FIELDS.map((field) => {
              // Structural cast, justified: ``field.key`` is a runtime
              // string from the FIELDS table driving this generic editor,
              // so the read is a genuine dynamic index into the character.
              const current = String((char as unknown as Record<string, unknown>)[field.key] ?? "");
              const value = charDraft[field.key] ?? current;
              const onFieldChange = (next: string): void => {
                setDrafts((prev) => ({
                  ...prev,
                  [char.id]: { ...(prev[char.id] ?? {}), [field.key]: next },
                }));
                scheduleAutosave(char.id, field.key, next);
              };
              return (
                <label key={field.key} style={{ display: "block", marginBottom: "0.5rem" }}>
                  <div style={{ color: "var(--muted)", fontSize: "0.85rem" }}>{field.key}</div>
                  {field.multiline ? (
                    <AutoGrowTextarea
                      value={value}
                      onChange={onFieldChange}
                      ariaLabel={`character ${char.id} ${field.key}`}
                    />
                  ) : (
                    <input
                      type="text"
                      value={value}
                      onChange={(e) => onFieldChange(e.target.value)}
                    />
                  )}
                </label>
              );
            })}
            {isOpen ? (
              <>
                <h4>Facts</h4>
                <ul>
                  {(char.facts ?? []).map((fact) => (
                    <li key={fact.id}>
                      <em>[{fact.confidence}]</em> {fact.text}
                    </li>
                  ))}
                </ul>
              </>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
