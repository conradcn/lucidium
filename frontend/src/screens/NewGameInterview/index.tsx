import type { JSX } from "react";
import { useEffect, useState } from "react";

import { send } from "../../app/client";
import { type MenuPair } from "../../app/menuManifest.generated";
import { useLucidiumStore } from "../../state/store";
import type { InterviewStep } from "../../shared/generated/C2SNewGameAnswer.schema";
import { CharacterStep } from "./CharacterStep";
import { ConfirmStep } from "./ConfirmStep";
import { GenreStep } from "./GenreStep";
import { NameStep } from "./NameStep";
import { SettingStep } from "./SettingStep";
import { VisualStyleStep } from "./VisualStyleStep";

/** The interview steps the backend accepts an answer for, plus the
 *  renderer-only ``confirm`` review screen (which submits through
 *  ``c2s/new_game/confirm``, not ``answer``). */
type Step = InterviewStep | "confirm";

interface Props {
  // The dream guide pair frozen at the moment the player clicked
  // New Game. Used to (a) keep the same companion visible through
  // the interview's early steps and (b) give the backend a starting
  // slug for prompt look-ups. The mount effect deps off this so
  // it MUST be stable across renders — recomputing it every render
  // would re-fire the effect and reset the interview to step 1.
  frozenPair: MenuPair | null;
  // The pair the carousel is CURRENTLY showing — ticks every ~10 s
  // as the carousel cycles. Used only for the interview's border
  // colour so it tracks the visible backdrop, fading along with
  // the carousel. Falling back to ``frozenPair`` when undefined
  // keeps existing call sites and tests happy.
  livePair?: MenuPair | null;
  // Called whenever the backend pushes a new preview asset — the
  // background after the player picks Visual Style, the player-
  // character portrait after they pick Character Description. App
  // uses these to gradually swap the cycling carousel for the
  // player's own chosen-style imagery; ``guide_path === null`` means
  // "background only — hide the cycling dream guide".
  onPreviewAssets: (
    assets: { background_path: string | null; guide_path: string | null } | null,
  ) => void;
  onLeftToMainView: () => void;
  // Optional: route back to the start-screen phase when the player
  // hits ``Go back`` on the Review step. Optional so existing
  // callers (and tests that don't render the Review step) compile
  // unchanged; ConfirmStep falls back to a no-op when undefined.
  // ``| undefined`` keeps the conditional-forward in the JSX
  // happy under ``exactOptionalPropertyTypes``.
  onCancel?: (() => void) | undefined;
}

/** The ``InterviewState`` fields holding a step's option list. Typed
 *  as a key union rather than ``string`` so indexing the snapshot with
 *  it stays type-checked — adding a step without its options field is
 *  now a compile error. */
type OptionsField =
  | "setting_options"
  | "genre_options"
  | "visual_style_options"
  | "character_description_options"
  | "name_options";

const STEP_OPTIONS_FIELD: Record<Exclude<Step, "confirm">, OptionsField> = {
  setting: "setting_options",
  genre: "genre_options",
  visual_style: "visual_style_options",
  character_description: "character_description_options",
  name: "name_options",
};

// One line per interview step, voiced in-character by the dream
// guide. Sets the tone for each question without giving away the
// answer.
const GUIDE_DIALOG: Record<Exclude<Step, "confirm">, string> = {
  setting: "Where shall we begin?",
  genre: "And what shape should this story take?",
  visual_style: "Through what lens shall you see it?",
  character_description: "Who, then, shall you be?",
  name: "And how shall you be known?",
};

const CONFIRM_DIALOG = "It is set, then. Shall we cross the threshold?";

// Defaults shown on the character-description / name screens
// while the LLM-generated options are still en route. Picking
// any of these is a real answer — the backend treats it the
// same as picking a freshly-generated option — so the user
// can move on without waiting if they're happy with the
// default. When the LLM round-trip lands, the renderer swaps
// to the real options via state/patch.
//
// Tuned to be GENERIC ENOUGH to fit any setting / genre, since
// these don't know what the player picked at earlier steps:
//   * Archetypes only — no setting-specific roles like
//     "starship navigator" or "court mage" that fit some
//     genres but read as imposed-genre-flavor in others.
//   * Names lean neutral / cross-cultural so they work as
//     placeholders against ink-noir, anime, fresco, etc.
const DEFAULT_CHARACTER_DESCRIPTIONS: string[] = [
  "wry archivist",
  "retired bounty hunter",
  "stoic apprentice",
  "cynical detective",
  "wandering scholar",
];

const DEFAULT_NAMES: string[] = [
  "Iris Vale",
  "Hale Stone",
  "Mira Quill",
  "Arden Locke",
  "Vance Reed",
];

const DEFAULT_OPTIONS: Partial<Record<Exclude<Step, "confirm">, string[]>> = {
  character_description: DEFAULT_CHARACTER_DESCRIPTIONS,
  name: DEFAULT_NAMES,
};

export function NewGameInterview({
  frozenPair,
  livePair,
  onPreviewAssets,
  onLeftToMainView,
  onCancel,
}: Props): JSX.Element {
  const [waitedOnce, setWaitedOnce] = useState(false);
  // The single source of truth for onboarding state — the ``/interview``
  // patches the backend pushes land here through the store's ``applyOp``,
  // and this subscription re-renders the moment they do.
  const snapshot = useLucidiumStore((s) => s.interview);
  const resetInterview = useLucidiumStore((s) => s.resetInterview);

  useEffect(() => {
    // Discard any stale interview snapshot from a previously aborted
    // onboarding so the user never momentarily sees the prior run's
    // selected options before the backend's reset patch arrives.
    resetInterview();
    // Hand the backend the frozen pair's character body + face so it
    // can re-render the same dream guide in whatever visual style
    // the player picks at step 3 (along with a background of their
    // chosen setting). The pre-render fires the moment we know
    // setting + visual_style.
    const startPayload: Record<string, unknown> = {};
    if (frozenPair) {
      startPayload["preview_character_slug"] = frozenPair.slug;
    }
    send("c2s/new_game/start", startPayload);
    const slowTimer = setTimeout(() => setWaitedOnce(true), 2000);
    return () => clearTimeout(slowTimer);
  }, [frozenPair, resetInterview]);

  // Forward the backend's preview assets to App as they land. Two
  // fields flow through here:
  //   * preview_background_path — set after Visual Style; when
  //     present, the dream guide gets hidden and the chosen-style
  //     backdrop replaces the cycling background.
  //   * pc_portrait_path — set after Character Description; when
  //     present, the player-character portrait replaces the hidden
  //     guide layer.
  // Selecting the two fields (rather than the whole slice) keeps this
  // effect from re-firing on every unrelated interview patch, and lets
  // App progressively reveal as more becomes available.
  const previewBackground = useLucidiumStore(
    (s) => s.interview.preview_background_path,
  );
  const pcPortrait = useLucidiumStore((s) => s.interview.pc_portrait_path);
  useEffect(() => {
    const bg = previewBackground || null;
    const portrait = pcPortrait || null;
    if (!bg && !portrait) return;
    onPreviewAssets({ background_path: bg, guide_path: portrait });
  }, [previewBackground, pcPortrait, onPreviewAssets]);

  // NOTE: the override is intentionally NOT cleared on unmount —
  // App handles the lifecycle. When the player clicks Begin the
  // interview unmounts but the loading phase needs the same
  // background + portrait visible behind its spinner. App's phase
  // effect resets them when the player lands on main / start /
  // load-game, so the next visit starts fresh.

  const step = (snapshot.step ?? "setting") as Step;

  const answer = (
    stepName: InterviewStep,
    value: string,
    isFreeText: boolean,
    pronouns: string = "",
  ): void => {
    send("c2s/new_game/answer", {
      step: stepName,
      answer: value,
      is_free_text: isFreeText,
      pronouns,
    });
  };

  // The dream guide's caption mirrors the active step. Confirm gets
  // its own line; option-loading screens reuse the step's prompt.
  const guideLine: string =
    step === "confirm" ? CONFIRM_DIALOG : GUIDE_DIALOG[step];

  // Apply the frozen pair's typography to the interview's question
  // card so it carries the same visual identity as the menu it
  // came from. The BORDER colour comes from ``livePair`` instead
  // — that ticks every ~10 s as the carousel cycles through pairs,
  // so the question-card outline fades along with whatever the
  // backdrop is currently highlighting. ``--border-accent`` is
  // wired into ``.interview > .page-frame .page-content`` (and
  // the guide-dialog border) in styles.css with a ``transition:
  // border-color ...`` so the colour change blends rather than
  // snaps. Fonts and letter-spacing stay frozen (changing them
  // mid-interview would flash across every label / option).
  const borderPair = livePair ?? frozenPair;
  const themeStyle: React.CSSProperties | undefined = frozenPair
    ? ({
        "--menu-accent": frozenPair.title_color,
        "--menu-font": frozenPair.title_font,
        "--menu-letter-spacing": frozenPair.title_letter_spacing,
        "--border-accent": (borderPair?.title_color) ?? frozenPair.title_color,
      } as React.CSSProperties)
    : undefined;

  // Resolve options for the current step. Steps with hardcoded
  // server-side options (setting, visual_style, genre) populate
  // synchronously; the LLM-driven steps (character_description,
  // name) used to block on the prefetch and show a "Generating
  // options…" splash. Now those steps render with built-in
  // defaults the moment the player arrives, plus a small
  // "loading" badge — they can pick a default and move on,
  // OR wait a moment for fresh suggestions to swap in.
  let optionsForStep: string[] = [];
  let usingDefaults = false;
  if (step !== "confirm") {
    const field = STEP_OPTIONS_FIELD[step];
    const fromSnapshot = snapshot[field] ?? [];
    if (fromSnapshot.length > 0) {
      optionsForStep = fromSnapshot;
    } else {
      const fallback = DEFAULT_OPTIONS[step];
      if (fallback && fallback.length > 0) {
        optionsForStep = fallback;
        usingDefaults = true;
      }
    }
    // No defaults available for this step AND no snapshot
    // options — fall back to the original "Generating options…"
    // splash. Should be unreachable in practice (every step
    // either has hardcoded backend options or sits in
    // DEFAULT_OPTIONS).
    if (optionsForStep.length === 0) {
      return (
        <div
          className="interview"
          data-testid="interview-loading-wrap"
          style={themeStyle}
        >
          <GuideDialog text={guideLine} />
          <div className="loading-splash" data-testid="interview-loading">
            <div className="glyph" />
            <h2>Generating options…</h2>
            <p className="hint">
              The engine is asking the LLM for {step.replace("_", " ")} choices.
            </p>
            {waitedOnce ? (
              <p className="hint">
                Still waiting. If this hangs, open <strong>Options</strong> from
                the start screen and check that an LLM endpoint and API key are
                configured.
              </p>
            ) : null}
          </div>
        </div>
      );
    }
  }

  return (
    <div
      className="interview"
      data-testid={`interview-${step}`}
      style={themeStyle}
    >
      <GuideDialog text={guideLine} />
      {step === "setting" && (
        <SettingStep
          options={snapshot.setting_options ?? []}
          onAnswer={(value, freeText) => answer("setting", value, freeText)}
        />
      )}
      {step === "genre" && (
        <GenreStep
          options={snapshot.genre_options ?? []}
          onAnswer={(value, freeText) => answer("genre", value, freeText)}
        />
      )}
      {step === "visual_style" && (
        <VisualStyleStep
          options={snapshot.visual_style_options ?? []}
          onAnswer={(value, freeText) => answer("visual_style", value, freeText)}
        />
      )}
      {step === "character_description" && (
        <CharacterStep
          options={optionsForStep}
          loading={usingDefaults}
          onAnswer={(value, freeText) => answer("character_description", value, freeText)}
        />
      )}
      {step === "name" && (
        <NameStep
          options={optionsForStep}
          loading={usingDefaults}
          onAnswer={(value, freeText, pronouns) =>
            answer("name", value, freeText, pronouns)
          }
        />
      )}
      {step === "confirm" && (
        <ConfirmStep
          snapshot={snapshot}
          onConfirmed={onLeftToMainView}
          onCancel={onCancel}
        />
      )}
    </div>
  );
}

/**
 * One-line dream-guide caption that sits above the figure. The
 * caption keys off its text so React remounts and the fade-in
 * animation re-fires whenever the line changes between steps.
 */
function GuideDialog({ text }: { text: string }): JSX.Element {
  return (
    <div
      key={text}
      className="interview-guide-dialog"
      data-testid="guide-dialog"
    >
      <span className="interview-guide-dialog__text">{text}</span>
    </div>
  );
}
