import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";

import { send } from "../../app/client";
import { useOptimisticAction } from "../../app/useOptimisticAction";

interface InterviewSnapshot {
  setting?: string;
  genre?: string;
  visual_style?: string;
  character_description?: string;
  character_name?: string;
  pronouns?: string;
  side_characters?: Record<string, { id: string; name: string }>;
}

interface SideCharacterRowProps {
  id: string;
  name: string;
  disabled: boolean;
}

function SideCharacterRow({
  id,
  name,
  disabled,
}: SideCharacterRowProps): JSX.Element {
  const [draft, setDraft] = useState(name);
  // Optimistic guard keyed on the canonical name — the moment the
  // backend echoes a rename via state/patch, the guard resets and
  // the buttons re-enable for the next interaction.
  const [pending, run] = useOptimisticAction(name);
  // Drop the draft and adopt the new canonical name when the
  // backend's state/patch lands (or any sibling action changes
  // the rendered name out from under us). Render-phase adjustment
  // rather than an effect (see
  // https://react.dev/learn/you-might-not-need-an-effect) so the row
  // never paints one frame of the stale draft.
  const [seenName, setSeenName] = useState(name);
  if (seenName !== name) {
    setSeenName(name);
    setDraft(name);
  }

  const trimmed = draft.trim();
  const isDirty = trimmed.length > 0 && trimmed !== name;

  const onSave = (): void => {
    if (!isDirty) return;
    run(() => {
      send("c2s/new_game/edit_side_character", {
        character_id: id,
        description: trimmed,
      });
    });
  };

  const onDelete = (): void => {
    run(() => {
      send("c2s/new_game/delete_side_character", { character_id: id });
    });
  };

  return (
    <li className="side-character-row" data-testid={`side-character-row-${id}`}>
      <input
        type="text"
        className="side-character-row__name"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") onSave();
        }}
        onBlur={() => {
          // Auto-save on blur so the player doesn't have to think
          // about an explicit Save button — same ergonomic as the
          // review fields above.
          if (isDirty) onSave();
        }}
        disabled={disabled || pending}
        aria-label={`Side character name (${name})`}
      />
      <button
        type="button"
        onClick={onDelete}
        disabled={disabled || pending}
        aria-label={`Remove ${name}`}
        className="side-character-row__delete"
      >
        ×
      </button>
    </li>
  );
}

interface Props {
  snapshot: InterviewSnapshot;
  onConfirmed: () => void;
  // ``Go back`` on the Review step jumps straight back to the main
  // menu. The parent screen owns the navigation; ConfirmStep just
  // fires the backend cancel message + invokes this callback.
  // Optional so tests that don't exercise the cancel path compile
  // unchanged; falls through to a no-op when undefined. The
  // explicit ``| undefined`` matches the parent's conditional-
  // forward pattern under ``exactOptionalPropertyTypes``.
  onCancel?: (() => void) | undefined;
}

// Allow-list of editable fields on the Review step. Maps the
// player-facing label to the interview-state key the backend
// expects. The order here is also the on-screen order.
const EDITABLE_FIELDS: Array<{
  key:
    | "setting"
    | "visual_style"
    | "genre"
    | "character_description"
    | "character_name"
    | "pronouns";
  label: string;
  multiline?: boolean;
}> = [
  { key: "setting", label: "Setting", multiline: true },
  { key: "genre", label: "Genre" },
  { key: "visual_style", label: "Visual style", multiline: true },
  { key: "character_description", label: "Character", multiline: true },
  { key: "character_name", label: "Name" },
  { key: "pronouns", label: "Pronouns" },
];

export function ConfirmStep({
  snapshot,
  onConfirmed,
  onCancel,
}: Props): JSX.Element {
  const [sideDescription, setSideDescription] = useState("");
  const sideCharacters = Object.values(snapshot.side_characters ?? {});
  // Begin is permanent — once clicked we transition out of the
  // interview, so a single optimistic guard covers the whole step.
  const [committed, run] = useOptimisticAction();
  // The Add-side-character button is per-add, not a one-shot. We
  // track its own pending state and let it reset whenever the side
  // characters list changes (which happens when the backend
  // acknowledges via a state/patch).
  const [addPending, addRun] = useOptimisticAction(sideCharacters.length);
  // Local drafts for review-screen edits. Each editable field
  // holds its in-flight typed value; on save the draft clears
  // and the displayed value falls back to the snapshot
  // (which the backend echoes back via state/patch). Same
  // pattern as CharactersTab's autosave to avoid the
  // typing-cursor-jumps-to-end bug. Drafts also get cleared
  // by the effect below once canonical state catches up.
  const [drafts, setDrafts] = useState<
    Partial<Record<typeof EDITABLE_FIELDS[number]["key"], string>>
  >({});
  // Per-field debounced save timers — same key shape as the
  // CharactersTab autosave so the pattern stays consistent.
  const saveTimersRef = useRef<Record<string, number>>({});

  // When the backend echoes the edit back via state/patch, the
  // matching draft is no longer needed. Clear drafts whose canonical
  // (snapshot) value matches. Render-phase adjustment rather than an
  // effect (https://react.dev/learn/you-might-not-need-an-effect):
  // React re-runs this component with the pruned drafts before
  // painting, so there is no cascading committed render.
  const [seenSnapshot, setSeenSnapshot] = useState(snapshot);
  if (seenSnapshot !== snapshot) {
    setSeenSnapshot(snapshot);
    setDrafts((prev) => {
      let changed = false;
      const next: typeof prev = {};
      for (const k of Object.keys(prev) as Array<
        typeof EDITABLE_FIELDS[number]["key"]
      >) {
        const draftValue = prev[k] ?? "";
        const canonical = snapshot[k] ?? "";
        if (canonical === draftValue) {
          changed = true;
          continue;
        }
        next[k] = draftValue;
      }
      return changed ? next : prev;
    });
  }

  useEffect(() => {
    // Snapshot the map identity now; the ref object may have been
    // swapped by the time the cleanup runs.
    const timers = saveTimersRef.current;
    return () => {
      Object.values(timers).forEach((id) => window.clearTimeout(id));
    };
  }, []);

  const scheduleSave = (
    field: typeof EDITABLE_FIELDS[number]["key"],
    value: string,
  ): void => {
    const existing = saveTimersRef.current[field];
    if (existing) window.clearTimeout(existing);
    saveTimersRef.current[field] = window.setTimeout(() => {
      send("c2s/new_game/edit_review", { field, value });
      delete saveTimersRef.current[field];
    }, 350);
  };

  const onFieldChange = (
    field: typeof EDITABLE_FIELDS[number]["key"],
    value: string,
  ): void => {
    setDrafts((prev) => ({ ...prev, [field]: value }));
    scheduleSave(field, value);
  };

  const flushPendingEdits = (): void => {
    // Before Begin, fire any debounced edit immediately so the
    // backend has the freshest values when it consumes the
    // world_init prefetch. Without this, a player who types
    // and IMMEDIATELY clicks Begin would lose the last few
    // characters of their edit because the 350ms debounce
    // hadn't fired yet.
    for (const field of Object.keys(saveTimersRef.current) as Array<
      typeof EDITABLE_FIELDS[number]["key"]
    >) {
      const timerId = saveTimersRef.current[field];
      if (timerId !== undefined) {
        window.clearTimeout(timerId);
        delete saveTimersRef.current[field];
        const draftValue = drafts[field];
        if (draftValue !== undefined) {
          send("c2s/new_game/edit_review", { field, value: draftValue });
        }
      }
    }
  };

  const addSide = (): void => {
    const text = sideDescription.trim();
    if (text.length === 0) return;
    addRun(() => {
      send("c2s/new_game/add_side_character", { description: text });
      setSideDescription("");
    });
  };

  const confirm = (): void => {
    run(() => {
      // Flush any debounced field edits BEFORE confirming so
      // the backend's world_init prefetch sees the latest
      // values. The backend's edit_review handler cancels +
      // re-fires the prefetch on each edit; flushing here
      // makes sure the LAST edit is on the wire before
      // confirm tries to consume it.
      flushPendingEdits();
      // Auto-add an unconfirmed typed description before
      // confirming. Players type a side character, click Begin,
      // and reasonably expect that NPC to be in their story —
      // forcing them to remember to click Add first is a
      // footgun. Trim + length-check matches the Add button's
      // own gate so we don't dispatch on whitespace-only text.
      const pending = sideDescription.trim();
      if (pending.length > 0) {
        send("c2s/new_game/add_side_character", { description: pending });
        setSideDescription("");
      }
      send("c2s/new_game/confirm", { overrides: {} });
      onConfirmed();
    });
  };

  const goBack = (): void => {
    // Flush any in-flight edits BEFORE we tell the backend to wipe
    // the interview, so a debounced edit can't race the cancel and
    // re-populate ``session.interview`` with a stale value the
    // user just discarded. ``c2s/new_game/go_back`` resets the
    // server-side state + cancels every prefetch. ``onCancel``
    // navigates the renderer back to the start phase optimistically.
    flushPendingEdits();
    send("c2s/new_game/go_back", {});
    if (onCancel) onCancel();
  };

  return (
    <div className="page-frame">
      <div className="page-content interview-step-content">
        <h2 className="page-title">Review</h2>
        <p
          className="page-subtitle"
          style={{ marginTop: 0, marginBottom: "0.6rem" }}
        >
          Edit any field below before clicking Begin.
        </p>
        <dl className="review-grid">
          {EDITABLE_FIELDS.map(({ key, label, multiline }) => {
            const draft = drafts[key];
            const value = draft ?? snapshot[key] ?? "";
            const inputId = `review-edit-${key}`;
            return (
              <div style={{ display: "contents" }} key={key}>
                <dt>
                  <label htmlFor={inputId}>{label}</label>
                </dt>
                <dd>
                  {multiline ? (
                    <textarea
                      id={inputId}
                      className="review-edit-input"
                      rows={2}
                      value={value}
                      onChange={(e) => onFieldChange(key, e.target.value)}
                      disabled={committed}
                      data-testid={`review-edit-${key}`}
                    />
                  ) : (
                    <input
                      id={inputId}
                      type="text"
                      className="review-edit-input"
                      value={value}
                      onChange={(e) => onFieldChange(key, e.target.value)}
                      disabled={committed}
                      data-testid={`review-edit-${key}`}
                    />
                  )}
                </dd>
              </div>
            );
          })}
        </dl>
        <h3 className="review-section-title">Side characters</h3>
        {sideCharacters.length > 0 ? (
          <ul className="side-character-list">
            {sideCharacters.map((side) => (
              <SideCharacterRow
                key={side.id}
                id={side.id}
                name={side.name}
                disabled={committed}
              />
            ))}
          </ul>
        ) : null}
        <div className="free-text-row" style={{ marginTop: 0 }}>
          <input
            type="text"
            placeholder="One-line description of a side character..."
            value={sideDescription}
            onChange={(e) => setSideDescription(e.target.value)}
            disabled={committed || addPending}
            data-testid="side-character-description"
          />
          <button
            onClick={addSide}
            disabled={committed || addPending || sideDescription.trim().length === 0}
          >
            Add
          </button>
        </div>
        <div className="confirm-actions" style={{ marginTop: "1rem" }}>
          <button
            type="button"
            onClick={goBack}
            disabled={committed}
            className="confirm-actions__back"
            data-testid="confirm-go-back"
          >
            Go back
          </button>
          <button
            type="button"
            onClick={confirm}
            disabled={committed}
            className="confirm-actions__begin"
            data-testid="confirm-begin"
          >
            Begin
          </button>
        </div>
      </div>
    </div>
  );
}
