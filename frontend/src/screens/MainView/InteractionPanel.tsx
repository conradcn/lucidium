import type { JSX } from "react";
import { useState } from "react";

import { send } from "../../app/client";
import { useOptimisticAction } from "../../app/useOptimisticAction";
import { useLucidiumStore } from "../../state/store";
import { Typewriter } from "./Typewriter";

interface DialogOption {
  id: string;
  text: string;
}

interface DialogNode {
  id: string;
  text: string | null;
  speaker_id: string | null;
  options: DialogOption[];
  parent_id: string | null;
  chosen_option_id: string | null;
  state?: string;
}

interface Character {
  id: string;
  name: string;
}

function hasContinueChild(
  nodes: Record<string, DialogNode> | undefined,
  parentId: string,
): boolean {
  if (!nodes) return false;
  for (const candidate of Object.values(nodes)) {
    if (
      candidate.parent_id === parentId &&
      candidate.chosen_option_id === null &&
      candidate.state !== "invalidated"
    ) {
      return true;
    }
  }
  return false;
}

interface PanelProps {
  flushSignal?: number;
  /** Ren'Py-style rollback offset: 0 = current beat (default
   *  interaction), 1+ = displaying the Nth-most-recent committed
   *  beat instead. While > 0 the panel is in read-only rollback
   *  mode — options + free-text are suppressed and a banner tells
   *  the player to scroll down to return. */
  rollbackOffset?: number;
}

export function InteractionPanel({
  flushSignal,
  rollbackOffset = 0,
}: PanelProps = {}): JSX.Element {
  const game = useLucidiumStore((s) => s.game) as
    | {
        current_node_id?: string | null;
        characters?: Record<string, Character>;
        dialog_tree?: {
          nodes: Record<string, DialogNode>;
          committed_path?: string[];
        };
      }
    | null;
  const settings = useLucidiumStore((s) => s.settings) as
    | { typewriter_speed_chars_per_sec?: number }
    | null;

  const [freeText, setFreeText] = useState("");
  // Reset on every dialog-node transition AND on every fresh game
  // object the store hands us. The store builds a new game ref on
  // every state/full and every state/patch that touches the game,
  // so any backend response that arrives after a click resets the
  // guard. Without the second key, when a c2s/play/advance leaves
  // current_node_id unchanged (backend hit a no-op or the same
  // speculative chain twice) the buttons would stay disabled
  // forever — a freeze the user reported in production.
  const currentNodeId = game?.current_node_id ?? "";
  const [pending, run] = useOptimisticAction(currentNodeId);

  // Rollback resolution: at offset 0 we render the current beat
  // (full interaction). At offset N>0 we walk the committed_path
  // backward and render that historical beat — the live beat's
  // node still exists in the tree, but the panel pretends the
  // story is at the older one. The backend's current_node_id
  // never moves; rollback is purely a renderer-side view shift.
  const isRolledBack = rollbackOffset > 0;
  const node = (() => {
    if (!game?.dialog_tree) return null;
    const path = game.dialog_tree.committed_path ?? [];
    if (isRolledBack && path.length > 0) {
      const targetIndex = path.length - 1 - rollbackOffset;
      const id = targetIndex >= 0 ? path[targetIndex] : path[0];
      if (typeof id === "string") {
        return game.dialog_tree.nodes[id] ?? null;
      }
    }
    if (!game.current_node_id) return null;
    return game.dialog_tree.nodes[game.current_node_id] ?? null;
  })();

  if (!node) {
    return (
      <div className="interaction">
        <div className="text">…</div>
      </div>
    );
  }

  const onChooseOption = (optionId: string | null): void => {
    run(() => send("c2s/play/advance", { option_id: optionId }));
  };

  const onSubmitFreeText = (): void => {
    const text = freeText.trim();
    if (text.length === 0) return;
    run(() => {
      send("c2s/play/free_text", { text });
      setFreeText("");
    });
  };

  const speaker = node.speaker_id ? game?.characters?.[node.speaker_id] : null;

  const hasOptions = node.options.length > 0;
  const hasWalkableChild = hasContinueChild(game?.dialog_tree?.nodes, node.id);
  // Spinner shows iff no choices are present AND clicking wouldn't
  // walk to an existing beat (so the player has to wait on a fresh
  // generation). The button stays clickable either way — the backend
  // awaits in-flight speculation or kicks off a new chain.
  const showSpinner = !hasOptions && !hasWalkableChild;

  return (
    <div
      className={`interaction${isRolledBack ? " interaction--rollback" : ""}`}
      data-testid="interaction-panel"
      data-rollback={isRolledBack ? "true" : "false"}
    >
      {isRolledBack ? (
        <div
          className="rollback-banner"
          role="status"
          data-testid="rollback-banner"
        >
          Rolled back {rollbackOffset} beat{rollbackOffset === 1 ? "" : "s"}
          {" — scroll down to return"}
        </div>
      ) : null}
      {speaker ? (
        <div className="speaker" data-testid="speaker-tag">
          {speaker.name}
        </div>
      ) : null}
      <div className="text">
        {isRolledBack ? (
          // No typewriter on history beats — show them instantly
          // since the player isn't reading them for the first time.
          <span>{node.text ?? ""}</span>
        ) : (
          <Typewriter
            text={node.text ?? ""}
            speedCharsPerSec={settings?.typewriter_speed_chars_per_sec ?? 60}
            flushSignal={flushSignal}
          />
        )}
      </div>
      {isRolledBack ? null : (
        <div className="options">
          {!hasOptions ? (
            <button
              className="continue-glyph"
              disabled={pending}
              onClick={() => onChooseOption(null)}
              aria-label="Continue"
              data-state={showSpinner ? "loading" : "ready"}
            >
              {showSpinner ? (
                <span
                  aria-hidden
                  className="continue-glyph__spinner"
                  data-testid="continue-glyph-spinner"
                />
              ) : (
                <span aria-hidden className="continue-glyph__triangle" />
              )}
            </button>
          ) : (
            node.options.map((opt) => (
              <button
                key={opt.id}
                disabled={pending}
                onClick={() => onChooseOption(opt.id)}
              >
                {opt.text}
              </button>
            ))
          )}
        </div>
      )}
      {isRolledBack ? null : <div className="free-text">
        <input
          type="text"
          placeholder="Or do something else..."
          value={freeText}
          onChange={(e) => setFreeText(e.target.value)}
          onKeyDown={(e) => {
            // Don't let typed keys bubble to the window-level
            // dialog-advance handler in MainView. The
            // ``inEditableField`` guard there relies on
            // ``document.activeElement.tagName``, which is
            // technically correct, but the user reported space
            // still advancing while typing — most likely a race
            // between disabled-state focus loss and a keypress
            // landing on the body. Stopping propagation here
            // closes the gap defensively: as long as THIS input
            // owns the keydown, the rest of the app never sees
            // it. Enter still triggers submit before propagation
            // stops, since we handle that on this same event.
            if (e.key === "Enter") {
              onSubmitFreeText();
            }
            e.stopPropagation();
          }}
          disabled={pending}
        />
        <button
          onClick={onSubmitFreeText}
          disabled={pending || freeText.trim().length === 0}
        >
          Submit
        </button>
      </div>}
    </div>
  );
}
