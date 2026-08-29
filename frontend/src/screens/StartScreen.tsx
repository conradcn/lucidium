import type { JSX } from "react";
import { useEffect, useState } from "react";

import { send } from "../app/client";
import {
  MENU_PAIRS,
  type MenuPair,
} from "../app/menuManifest.generated";
import { type MenuCycleState } from "../app/useMenuCycle";
import { useDelayedValue } from "../app/useDelayedValue";
import {
  BUTTON_STAGGER_MS,
  useMenuTransition,
} from "../app/useMenuTransition";
import { useOptimisticAction } from "../app/useOptimisticAction";
import { useLucidiumStore } from "../state/store";
import { OrchestatedText } from "./OrchestatedText";
import { StaggeredText } from "./StaggeredText";

interface StartScreenProps {
  cycle: MenuCycleState;
  onNewGame: () => void;
  onLoadGame: () => void;
  onSettings: () => void;
  onContinue?: () => void;
  onSurpriseMe?: () => void;
}

interface UserProfileShape {
  likes?: unknown;
  dislikes?: unknown;
  notes?: unknown;
  summarizer_likes?: unknown;
  summarizer_dislikes?: unknown;
  summarizer_notes?: unknown;
}

export function StartScreen({
  cycle,
  onNewGame,
  onLoadGame,
  onSettings,
  onContinue: requestContinue,
  onSurpriseMe: requestSurpriseMe,
}: StartScreenProps): JSX.Element {
  const hasSave = useLucidiumStore((s) => s.hasSave);
  // Surprise Me builds the scenario from the cross-save player
  // profile. With NO recorded preferences the LLM has nothing to
  // tailor against and produces a generic story indistinguishable
  // from the regular onboarding flow — so we gate the action on
  // the profile actually being non-empty.
  const settings = useLucidiumStore((s) => s.settings) as
    | { user_profile?: UserProfileShape }
    | null;
  const [showProfileNudge, setShowProfileNudge] = useState(false);
  const [pending, run] = useOptimisticAction();

  const total = MENU_PAIRS.length;
  const guideSlot = cycle.guide;
  // The title sits on top of every layer; its style follows the
  // dream-guide's pair so it matches whatever silhouette is on stage.
  const titleCurrent: MenuPair | null =
    total > 0 ? (MENU_PAIRS[guideSlot.pair[guideSlot.current]] ?? null) : null;
  const titlePrevious: MenuPair | null =
    total > 0
      ? (MENU_PAIRS[guideSlot.pair[guideSlot.current === 0 ? 1 : 0]] ?? null)
      : null;

  // Four-phase transition orchestrator. Drives:
  //   - the title's split erase/rewrite (phases A and C)
  //   - the menu accent crossfade (flips at phase B)
  //   - the per-button rewrite cascade at phase D, with each button
  //     ``index * BUTTON_STAGGER_MS`` behind the previous one.
  const transition = useMenuTransition(guideSlot.bump);
  // Until phase B starts the menu accent stays on the OLD pair so
  // button borders, hover colour and the title shade don't swap
  // ahead of the background.
  const accentPair: MenuPair | null = transition.showNewAccent
    ? titleCurrent
    : (titlePrevious ?? titleCurrent);

  useEffect(() => {
    send("c2s/saves/list");
  }, []);

  const onContinue = (): void => {
    requestContinue?.();
    send("c2s/saves/continue");
  };

  const onSurpriseMe = (): void => {
    // Empty-profile guard. With nothing in either the player-typed
    // or summarizer-inferred buckets, the scenario LLM call gets
    // no signal to tailor against and produces a story that's
    // indistinguishable from what the regular onboarding flow
    // would. Pop a nudge directing the player to Settings instead
    // of running the round-trip.
    if (!hasAnyProfileEntry(settings?.user_profile)) {
      setShowProfileNudge(true);
      return;
    }
    // Hand control to App.tsx so the loading-phase snapshot lands
    // BEFORE the backend's state/full arrives (without that the
    // new game's current_node_id matches the snapshot for the
    // prior session and the auto-route stays put). Then send the
    // request — backend runs scenario LLM call + world_init in
    // sequence and emits state/full when the opening node is ready.
    requestSurpriseMe?.();
    send("c2s/new_game/surprise_me");
  };

  const onExit = (): void => {
    send("c2s/app/exit");
    window.close();
  };

  const titleStyleFor = (pair: MenuPair | null): React.CSSProperties | undefined =>
    pair
      ? ({
          color: pair.title_color,
          fontFamily: pair.title_font,
          letterSpacing: pair.title_letter_spacing,
          fontWeight: pair.title_weight,
          // Per-pair stroke + shadow tint so dark titles get a
          // light shade and light titles get a dark shade. Read
          // out via ``var(--menu-title-shade)`` in styles.css.
          "--menu-title-shade": pair.title_shade_color,
        } as React.CSSProperties)
      : undefined;

  return (
    <div className="start-screen" data-testid="start-screen">
      {/* Carousel itself is rendered by App so it survives the
          start → interview → start round trip. StartScreen only
          owns the menu panel + title now. */}
      <div
        className="start-screen__menu"
        data-testid="start-menu"
        data-transition-phase={transition.phase}
        style={
          accentPair
            ? ({
                // Menu wrap inherits cues from whichever pair the
                // accent is currently locked to (old until phase B,
                // new from B onward). The CSS transition on
                // ``--menu-accent`` smooths the change across the
                // 1.5 s of phase B.
                "--menu-accent": accentPair.title_color,
                "--menu-font": accentPair.title_font,
                "--menu-letter-spacing": accentPair.title_letter_spacing,
              } as React.CSSProperties)
            : undefined
        }
      >
        <div className="start-screen__title-wrap" data-testid="start-title-wrap">
          {/* Single h1 carries the accessible name; the visible
              animation is per-letter underneath via OrchestatedText
              so phase A backspaces in the OLD style and phase C
              rewrites in the NEW one. The h1 itself stays a stable
              LUCIDIUM for screen readers and Playwright role queries. */}
          <h1
            className="start-screen__title start-screen__title--current"
            style={titleStyleFor(titleCurrent)}
            data-active="true"
            data-slug={titleCurrent?.slug ?? ""}
          >
            <span className="visually-hidden">LUCIDIUM</span>
            <OrchestatedText
              text="LUCIDIUM"
              styleCurrent={titleStyleFor(titleCurrent) ?? {}}
              stylePrevious={titleStyleFor(titlePrevious)}
              eraseBump={transition.titleEraseBump}
              rewriteBump={transition.titleRewriteBump}
              stepMs={70}
            />
          </h1>
        </div>
        <MenuButton
          label="Continue"
          visible={hasSave}
          onClick={onContinue}
          pending={pending}
          run={run}
          current={titleCurrent}
          previous={titlePrevious}
          buttonStaggerBump={transition.buttonStaggerBump}
          buttonIndex={0}
        />
        <MenuButton
          label="New Game"
          onClick={onNewGame}
          pending={pending}
          run={run}
          current={titleCurrent}
          previous={titlePrevious}
          buttonStaggerBump={transition.buttonStaggerBump}
          buttonIndex={1}
        />
        <MenuButton
          label="Surprise Me"
          onClick={onSurpriseMe}
          pending={pending}
          run={run}
          current={titleCurrent}
          previous={titlePrevious}
          buttonStaggerBump={transition.buttonStaggerBump}
          buttonIndex={2}
        />
        <MenuButton
          label="Load Game"
          onClick={onLoadGame}
          pending={pending}
          run={run}
          current={titleCurrent}
          previous={titlePrevious}
          buttonStaggerBump={transition.buttonStaggerBump}
          buttonIndex={3}
        />
        <MenuButton
          label="Options"
          onClick={onSettings}
          pending={pending}
          run={run}
          current={titleCurrent}
          previous={titlePrevious}
          buttonStaggerBump={transition.buttonStaggerBump}
          buttonIndex={4}
        />
        <MenuButton
          label="Exit"
          onClick={onExit}
          pending={pending}
          run={run}
          current={titleCurrent}
          previous={titlePrevious}
          buttonStaggerBump={transition.buttonStaggerBump}
          buttonIndex={5}
        />
      </div>
      {showProfileNudge ? (
        <SurpriseMeProfileNudge
          onClose={() => setShowProfileNudge(false)}
          onOpenSettings={() => {
            setShowProfileNudge(false);
            onSettings();
          }}
        />
      ) : null}
    </div>
  );
}


function hasAnyProfileEntry(profile: UserProfileShape | undefined): boolean {
  if (!profile) return false;
  // Surprise Me reads the merged profile (player-typed UNION
  // summarizer-inferred), so any entry in any bucket counts as
  // signal. We deliberately ignore ``dismissed_*`` here — those
  // are deletions, not preferences.
  const buckets: (keyof UserProfileShape)[] = [
    "likes",
    "dislikes",
    "notes",
    "summarizer_likes",
    "summarizer_dislikes",
    "summarizer_notes",
  ];
  for (const key of buckets) {
    const val = profile[key];
    if (Array.isArray(val) && val.some((v) => typeof v === "string" && v.trim().length > 0)) {
      return true;
    }
  }
  return false;
}


/**
 * Modal shown when the player clicks Surprise Me before the
 * cross-save player profile has any entries. Without preferences
 * the scenario-LLM call has nothing to tailor against and
 * produces generic output, so nudge the player into Settings to
 * fill at least one bucket first.
 */
function SurpriseMeProfileNudge({
  onClose,
  onOpenSettings,
}: {
  onClose: () => void;
  onOpenSettings: () => void;
}): JSX.Element {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="surprise-nudge-title"
      className="surprise-me-nudge"
    >
      <div className="surprise-me-nudge__panel">
        <h2 id="surprise-nudge-title">Tell me what you like first</h2>
        <p>
          Surprise Me builds the scenario from your saved likes, dislikes, and
          notes — so the story is something you actually want to play. There's
          nothing recorded yet, so a generated scenario wouldn't have anything
          to lean on.
        </p>
        <p>
          Open Settings, fill in a few entries under <em>Player profile</em>,
          then come back and click Surprise Me.
        </p>
        <div className="surprise-me-nudge__buttons">
          <button onClick={onOpenSettings}>Open Settings</button>
          <button onClick={onClose}>Cancel</button>
        </div>
      </div>
    </div>
  );
}

interface MenuButtonProps {
  label: string;
  visible?: boolean;
  onClick: () => void;
  pending: boolean;
  run: (action: () => void) => void;
  current: MenuPair | null;
  previous: MenuPair | null;
  buttonStaggerBump: number;
  buttonIndex: number;
}

function MenuButton({
  label,
  visible = true,
  onClick,
  pending,
  run,
  current,
  previous,
  buttonStaggerBump,
  buttonIndex,
}: MenuButtonProps): JSX.Element | null {
  // Each button rewrites ``buttonIndex * BUTTON_STAGGER_MS`` behind
  // the previous one. ``useDelayedValue`` mirrors the global tick
  // into a per-button bump that StaggeredText reacts to.
  const localBump = useDelayedValue(
    buttonStaggerBump,
    buttonIndex * BUTTON_STAGGER_MS,
  );
  if (!visible) return null;
  return (
    <button
      disabled={pending}
      onClick={() => run(onClick)}
      aria-label={label}
    >
      {renderMenuLabel(label, current, previous, localBump)}
    </button>
  );
}

/**
 * Render a menu button's label as a per-letter staggered backspace+
 * rewrite when ``bump`` ticks. Reuses ``StaggeredText`` because
 * buttons run a single chained cycle (no empty-pause between phases
 * the way the title does).
 *
 * Both ``fontFamily`` AND ``letterSpacing`` are set on the inner
 * span (not the button) so they only swap at StaggeredText's empty
 * moment — between the backspace and rewrite halves of the cycle,
 * when no letters are visible. The earlier impl drove letter-spacing
 * from a button-level CSS variable that snapped to the new value
 * the instant the pair changed, producing a visible kerning shift
 * under the still-readable old label. ``0.4`` matches the historic
 * scale factor that converted the per-pair title letter-spacing
 * into a tighter button-text spacing.
 */
function renderMenuLabel(
  text: string,
  current: MenuPair | null,
  previous: MenuPair | null,
  bump: number,
): JSX.Element {
  const styleCurrent: React.CSSProperties = current
    ? {
        fontFamily: current.title_font,
        letterSpacing: `calc(${current.title_letter_spacing} * 0.4)`,
      }
    : {};
  const stylePrevious: React.CSSProperties | undefined = previous
    ? {
        fontFamily: previous.title_font,
        letterSpacing: `calc(${previous.title_letter_spacing} * 0.4)`,
      }
    : undefined;
  return (
    <span className="start-screen__menu-label-wrap" aria-hidden>
      <StaggeredText
        text={text}
        styleCurrent={styleCurrent}
        stylePrevious={stylePrevious}
        bumpKey={bump}
        stepMs={40}
      />
    </span>
  );
}
