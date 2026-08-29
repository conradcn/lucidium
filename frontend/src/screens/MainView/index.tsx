import type { JSX } from "react";
import { useEffect, useState } from "react";

import { ensureClient, send } from "../../app/client";
import { useOptimisticAction } from "../../app/useOptimisticAction";
import { useLucidiumStore } from "../../state/store";
import { Background } from "./Background";
import { CharacterStage } from "./CharacterStage";
import { InteractionPanel } from "./InteractionPanel";
import { StoryPanel } from "./StoryPanel";

interface Props {
  onOpenStory: () => void;
  onOpenMenu: () => void;
  onOpenSettings: () => void;
}

export function MainView({ onOpenStory, onOpenMenu, onOpenSettings }: Props): JSX.Element {
  const status = useLucidiumStore((s) => s.status);
  const game = useLucidiumStore((s) => s.game);
  // The renderer doesn't see the backend's undo stack directly, so it
  // optimistically enables Undo whenever there's a current node — the
  // backend returns a not_found error if the stack is empty, which we
  // surface in the same banner as other provider errors.
  const canUndo = Boolean(game?.current_node_id);
  const [storyOpen, setStoryOpen] = useState(false);
  // Confirmation gate on the Menu button. Returning to the start
  // screen is reversible (Continue picks the run back up), but
  // accidentally clicking Menu mid-scene reads as "I lost my
  // place" — the carousel dethrones the active backdrop, the
  // music swaps to the menu theme, and the player has to find
  // their way back in. The modal turns the click into a
  // deliberate action.
  const [menuConfirmOpen, setMenuConfirmOpen] = useState(false);
  // Hide-text mode: lets the player see the full image without the
  // dialog box covering the bottom third. Toggled from the top bar
  // and via the "h" hotkey. State lives only on the renderer — the
  // backend keeps generating beats either way.
  const [textHidden, setTextHidden] = useState(false);
  // Flush signal: bumped on every click-to-advance so the
  // Typewriter jumps to the FULL current line before the next beat
  // swaps in. Without this, a player who clicks faster than the
  // typewriter can finish never reads the beats they're skipping.
  const [flushSignal, setFlushSignal] = useState(0);
  const [providerError, setProviderError] = useState<string | null>(null);
  // Track advances per-node so a single click only fires once even if
  // it bubbles through multiple layers; reset when the node changes.
  const currentNodeId = game?.current_node_id ?? "";
  const [advancePending, runAdvance] = useOptimisticAction(currentNodeId);
  // Ren'Py-style rollback. Wheel-up over the game view scrolls back
  // through the committed_path one beat per wheel event; wheel-down
  // returns toward the live beat. While rollbackOffset > 0 the
  // InteractionPanel is in read-only mode (no options, no free
  // text, no Continue) — the player is reviewing past prose, not
  // making decisions. Reaching the live beat (offset 0) returns
  // the panel to its normal interactive state. The backend's
  // current_node_id is never moved; rollback is a renderer-side
  // view shift, so speculation, music, and image scheduling all
  // continue running on the live state in the background.
  const [rollbackOffset, setRollbackOffset] = useState(0);
  const committedPathLen = game?.dialog_tree?.committed_path?.length ?? 0;
  // Reset rollback whenever the live beat advances — otherwise a
  // player rolled back 3 beats would suddenly find themselves
  // looking 4 beats back the moment a new beat lands. Reset is
  // also the natural way to exit rollback if the player picks an
  // option / advances normally.
  //
  // Render-phase adjustment rather than an effect (see
  // https://react.dev/learn/you-might-not-need-an-effect): the reset
  // lands before the new beat is painted, so there is no frame in
  // which the new beat is rendered at the old rollback offset.
  const [seenNodeId, setSeenNodeId] = useState(currentNodeId);
  if (seenNodeId !== currentNodeId) {
    setSeenNodeId(currentNodeId);
    setRollbackOffset(0);
  }

  useEffect(() => {
    const client = ensureClient();
    const off = client.on("s2c/error", (p) => {
      if (p.code === "provider_unreachable") {
        setProviderError(p.message);
      }
    });
    return off;
  }, []);

  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      // Drop every keystroke if the window isn't focused. Without
      // this the renderer can pick up keys routed from a sibling
      // BrowserWindow (DevTools detached, a child window the user
      // opened) or from a background-mode synthesised event,
      // advancing the dialog while the user thinks they're typing
      // somewhere else entirely. ``document.hasFocus()`` returns
      // true only when this document is the OS-active focus
      // target, so the gate is a single primitive check that
      // closes every "key fired from elsewhere" path.
      if (!document.hasFocus()) return;
      // "h" toggles the dialog box. Skip when an input is focused so
      // typing "h" in free-text doesn't accidentally hide the UI.
      // Resolve the editable-field check via BOTH ``event.target``
      // and ``document.activeElement``: target is more reliable
      // mid-keystroke (activeElement can briefly fall to body on
      // disabled-input transitions), but activeElement catches
      // cases where the event target is the document/window
      // itself (e.g., synthesised keypress events from automation).
      const targetTag = ((event.target as Element | null)?.tagName ?? "").toLowerCase();
      const activeTag = (document.activeElement?.tagName ?? "").toLowerCase();
      const targetEditable = (event.target as HTMLElement | null)?.isContentEditable ?? false;
      const activeEditable = (document.activeElement as HTMLElement | null)?.isContentEditable ?? false;
      const inEditableField =
        targetTag === "input" || targetTag === "textarea" || targetTag === "select" ||
        activeTag === "input" || activeTag === "textarea" || activeTag === "select" ||
        targetEditable || activeEditable;
      if (event.key === "h" || event.key === "H") {
        if (inEditableField) return;
        setTextHidden((v) => !v);
      }
      // Space advances the dialog when the player isn't typing —
      // matches the visual-novel convention that the spacebar
      // "moves the script forward". Same gating as the implicit
      // continue: only when the current node has NO explicit
      // options (otherwise we'd silently bypass a real fork).
      // Skipped while focused in a free-text / search input so
      // typing actual spaces still works.
      if (event.code === "Space" || event.key === " ") {
        if (inEditableField) return;
        if (storyOpen) return;
        if (advancePending) return;
        // Space during rollback exits rollback rather than
        // advancing — matches the click-handler behaviour.
        if (rollbackOffset > 0) {
          event.preventDefault();
          setRollbackOffset(0);
          return;
        }
        if (!canImplicitContinue) return;
        event.preventDefault();
        setFlushSignal((s) => s + 1);
        runAdvance(() => send("c2s/play/advance", { option_id: null }));
      }
      // PageUp / PageDown as keyboard mirrors of wheel
      // rollback — accessibility for trackpads / non-wheel mice.
      if (event.key === "PageUp") {
        if (inEditableField || storyOpen) return;
        const max = Math.max(0, committedPathLen - 1);
        setRollbackOffset((o) => Math.min(o + 1, max));
        event.preventDefault();
      }
      if (event.key === "PageDown") {
        if (inEditableField || storyOpen) return;
        setRollbackOffset((o) => Math.max(o - 1, 0));
        event.preventDefault();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // Click-anywhere-to-continue: visual-novel convention. Only fire
  // when the current node has no explicit options (the only path
  // forward is the implicit Continue). When the LLM returns a choice
  // list, the player must click a specific button — silently
  // advancing past a real fork would lose information.
  const node = game?.current_node_id
    ? game.dialog_tree?.nodes?.[game.current_node_id]
    : null;
  const canImplicitContinue =
    Boolean(node) && (node?.options?.length ?? 0) === 0;
  const onStageWheel = (event: React.WheelEvent<HTMLDivElement>): void => {
    if (storyOpen) return;
    // Ignore wheel inside the story panel / overlays / scrollable
    // text inputs — the target's closest scrollable should always
    // win. The story panel sets its own overflow, so its wheel
    // events don't bubble to us in practice; this guards against
    // free-text textareas that might.
    const target = event.target as Element | null;
    if (target?.closest("input, textarea")) return;
    if (event.deltaY < 0) {
      // Scroll up = rollback one beat. Cap at one less than the
      // committed path so we never roll past the first beat.
      const max = Math.max(0, committedPathLen - 1);
      setRollbackOffset((o) => Math.min(o + 1, max));
    } else if (event.deltaY > 0) {
      setRollbackOffset((o) => Math.max(o - 1, 0));
    }
  };

  const onStageClick = (event: React.MouseEvent<HTMLDivElement>): void => {
    if (storyOpen) return;
    const target = event.target as Element | null;
    if (!target) return;
    // Click during rollback returns to the live beat instead of
    // advancing — matches Ren'Py, which treats any click in
    // rollback-mode as "exit rollback first". This check sits
    // BEFORE the canImplicitContinue gate so clicks while a live
    // beat has options (no implicit continue) still exit
    // rollback. Buttons inside the panel still short-circuit
    // through the closest("button,...") guard below.
    if (rollbackOffset > 0) {
      if (target.closest("button, input, textarea, select, a, [role='button']")) {
        return;
      }
      const onStageRegion = target.closest(
        ".background, .stage, .character, .interaction",
      );
      if (!onStageRegion) return;
      setRollbackOffset(0);
      return;
    }
    // Whitelist: ONLY clicks on the actual game stage qualify as
    // "the player clicked the dialog itself". Buttons / inputs
    // inside the panel short-circuit so an option click doesn't
    // double as a stage click.
    if (target.closest("button, input, textarea, select, a, [role='button']")) {
      return;
    }
    const onStage = target.closest(
      ".background, .stage, .character, .interaction",
    );
    if (!onStage) return;
    // Flush the typewriter on EVERY stage click — even when
    // options are present, or an advance is already in flight,
    // or this beat won't implicit-continue. The player asked to
    // finish reading; we honour that regardless of whether the
    // click can also advance.
    setFlushSignal((s) => s + 1);
    if (advancePending) return;
    if (!canImplicitContinue) return;
    runAdvance(() => send("c2s/play/advance", { option_id: null }));
  };

  return (
    <div
      className={`main-view${storyOpen ? " story-open" : ""}${
        canImplicitContinue && !storyOpen ? " can-advance" : ""
      }`}
      data-testid="main-view"
      onClick={onStageClick}
      onWheel={onStageWheel}
    >
      {status !== "open" ? (
        <div className="banner" role="alert">
          Connection {status}. Reconnecting…
        </div>
      ) : providerError ? (
        <div className="banner" role="alert">
          {providerError} —{" "}
          <button onClick={onOpenSettings} style={{ marginLeft: "0.5rem" }}>
            Open Settings
          </button>
        </div>
      ) : null}
      <div className="top-bar">
        <button
          onClick={() => {
            setStoryOpen((v) => !v);
            onOpenStory();
          }}
        >
          Story
        </button>
        <button
          disabled={!canUndo}
          onClick={() => send("c2s/play/undo", {})}
          title="Go back to the last choice"
        >
          Undo
        </button>
        <button
          onClick={() => setTextHidden((v) => !v)}
          title="Show or hide the dialog box (hotkey: H)"
        >
          {textHidden ? "Show text" : "Hide text"}
        </button>
        <button onClick={() => setMenuConfirmOpen(true)}>Menu</button>
      </div>
      <Background />
      <CharacterStage hideText={textHidden} />
      {textHidden ? null : (
        <InteractionPanel
          flushSignal={flushSignal}
          rollbackOffset={rollbackOffset}
        />
      )}
      {storyOpen ? (
        <StoryPanel
          onClose={() => setStoryOpen(false)}
          onOpenSettings={onOpenSettings}
        />
      ) : null}
      {menuConfirmOpen ? (
        <MenuConfirm
          onCancel={() => setMenuConfirmOpen(false)}
          onConfirm={() => {
            setMenuConfirmOpen(false);
            onOpenMenu();
          }}
        />
      ) : null}
    </div>
  );
}

/**
 * Modal confirmation for the Menu button on the game screen.
 * Sits on top of the main view via a fixed overlay; clicking
 * outside the panel cancels (matches the convention from
 * StartupWarning). The save is auto-committed on every turn so
 * "lose progress" is not a real concern, but the trip back to
 * the start screen still wipes the in-flight scene context —
 * the player loses the typewriter mid-line and the music
 * crossfade. A tiny prompt makes the navigation deliberate.
 */
function MenuConfirm({
  onCancel,
  onConfirm,
}: {
  onCancel: () => void;
  onConfirm: () => void;
}): JSX.Element {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="menu-confirm-title"
      onClick={(e) => {
        // Clicking the backdrop (not the panel) cancels.
        if (e.target === e.currentTarget) onCancel();
      }}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0, 0, 0, 0.55)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
      }}
    >
      <div
        style={{
          background: "var(--bg, #15131a)",
          border: "1px solid var(--border-soft, rgba(255,255,255,0.1))",
          borderRadius: "var(--r-md, 0.5rem)",
          padding: "1.4rem 1.6rem",
          maxWidth: "26rem",
          boxShadow: "0 8px 32px rgba(0, 0, 0, 0.6)",
        }}
      >
        <h3 id="menu-confirm-title" style={{ margin: "0 0 0.6rem" }}>
          Return to the main menu?
        </h3>
        <p style={{ margin: "0 0 1.2rem", color: "var(--muted)" }}>
          Your progress is auto-saved — Continue from the start
          screen will resume this run exactly where you left off.
        </p>
        <div style={{ display: "flex", gap: "0.5rem", justifyContent: "flex-end" }}>
          <button onClick={onCancel} autoFocus>
            Stay
          </button>
          <button onClick={onConfirm}>Return to menu</button>
        </div>
      </div>
    </div>
  );
}
