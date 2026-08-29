import type { JSX } from "react";
import {
  MENU_BASE,
  MENU_PAIRS,
  type MenuPair,
} from "../app/menuManifest.generated";
import type { MenuCycleState } from "../app/useMenuCycle";

interface OverridePair {
  // Progressive override: ``background`` is set after the player
  // picks Visual Style, ``guide`` after they pick Character
  // Description (the player-character portrait replaces the dream
  // guide). When ``background`` is set the carousel stops cycling.
  // When ``guide`` is null the guide layer is hidden entirely —
  // the player has chosen a backdrop but no portrait yet.
  background: string | null;
  guide: string | null;
  slug: string;
}

interface Props {
  cycle: MenuCycleState;
  override?: OverridePair | null;
}

/**
 * Renders the cycling background + dream-guide layers used by both
 * the start screen and the new-game interview. The cycle itself is
 * driven by ``useMenuCycle()`` at the App level so the position
 * persists across phase transitions.
 *
 * When ``override`` is set the carousel stops cycling — the override
 * pair becomes the only thing rendered. This is how the player's
 * chosen-style preview imagery replaces the menu loop after the
 * pre-render finishes.
 */
export function MenuCarousel({ cycle, override }: Props): JSX.Element {
  if (override && override.background) {
    // Background-known case. Stop cycling and pin the chosen-style
    // backdrop. The guide layer is rendered ONLY if a portrait has
    // also arrived — otherwise the dream guide is hidden entirely
    // (the player has chosen a setting + style; the next thing
    // they'll see in this layer is their own character).
    return (
      <>
        <div
          className="start-screen__bg-stack"
          data-testid="start-bg-stack"
          aria-hidden
        >
          <div
            key={`bg-override-${override.slug}`}
            className="start-screen__bg start-screen__bg--current"
            style={{ backgroundImage: `url("${override.background}")` }}
            data-active="true"
            data-slug={override.slug}
          />
        </div>
        {override.guide ? (
          <div
            className="start-screen__guide-stack"
            data-testid="start-guide-stack"
            aria-hidden
          >
            <img
              key={`guide-override-${override.slug}`}
              className="start-screen__guide start-screen__guide--current"
              src={override.guide}
              alt=""
              data-active="true"
              data-slug={override.slug}
            />
          </div>
        ) : null}
      </>
    );
  }

  // bump === 0 is the initial mount: both slots hold the SAME pair
  // index (see ``initialState`` in useMenuCycle), so the radial /
  // wipe animations would fire visibly even though there's nothing
  // to transition between. Suppress them on the first frame so the
  // game opens already settled on the first pair; the next tick
  // (PAIR_DURATION_MS later) re-keys both slots and runs the full
  // four-phase cycle.
  const bgInitial = cycle.bg.bump === 0;
  const guideInitial = cycle.guide.bump === 0;
  const suppressBg: React.CSSProperties = { animation: "none" };
  // Suppressing the guide-current animation alone leaves --wipe-edge
  // pinned at the keyframe's ``from`` value (-10%), which renders
  // the mask fully transparent — the guide vanishes. Override the
  // edge to 110% (the keyframe's ``to`` value) so the guide is fully
  // visible on the first frame without any transition running.
  const suppressGuide: React.CSSProperties = {
    animation: "none",
    ["--wipe-edge" as string]: "110%",
  };
  return (
    <>
      <div
        className="start-screen__bg-stack"
        data-testid="start-bg-stack"
        aria-hidden
      >
        {([0, 1] as const).map((slot) => {
          const pair = MENU_PAIRS[cycle.bg.pair[slot]]!;
          const role = slot === cycle.bg.current ? "current" : "previous";
          const reactKey =
            role === "current" ? `bg-cur-${cycle.bg.bump}` : `bg-prev`;
          const baseStyle: React.CSSProperties = {
            backgroundImage: `url("${MENU_BASE}${pair.background}")`,
          };
          const style =
            bgInitial && role === "current"
              ? { ...baseStyle, ...suppressBg }
              : baseStyle;
          return (
            <div
              key={reactKey}
              className={`start-screen__bg start-screen__bg--${role}`}
              style={style}
              data-active={role === "current" ? "true" : "false"}
              data-slug={pair.slug}
            />
          );
        })}
      </div>
      <div
        className="start-screen__guide-stack"
        data-testid="start-guide-stack"
        aria-hidden
      >
        {(() => {
          const currentIdx = cycle.guide.pair[cycle.guide.current];
          const previousIdx =
            cycle.guide.pair[cycle.guide.current === 0 ? 1 : 0];
          const currentPair = MENU_PAIRS[currentIdx]!;
          const previousPair = MENU_PAIRS[previousIdx]!;
          return (
            <>
              {/* On the initial mount both slots hold the same pair, so
                  the wipe-down animation would erase a guide that's
                  already correct. Skip rendering the previous slot
                  entirely on the first frame. */}
              {guideInitial ? null : (
                <img
                  key={`guide-prev-${cycle.guide.bump}`}
                  className="start-screen__guide start-screen__guide--previous"
                  src={`${MENU_BASE}${previousPair.guide}`}
                  alt=""
                  data-active="false"
                  data-slug={previousPair.slug}
                />
              )}
              <img
                key={`guide-cur-${cycle.guide.bump}`}
                className="start-screen__guide start-screen__guide--current"
                src={`${MENU_BASE}${currentPair.guide}`}
                alt=""
                style={guideInitial ? suppressGuide : undefined}
                data-active="true"
                data-slug={currentPair.slug}
              />
            </>
          );
        })()}
      </div>
    </>
  );
}

/**
 * The "frozen" pair the player saw at the moment they clicked New
 * Game. Computed from the cycle state so the interview can carry
 * the same companion through every step until the preview render
 * lands.
 */
export function activeGuidePair(cycle: MenuCycleState): MenuPair | null {
  return MENU_PAIRS[cycle.guide.pair[cycle.guide.current]] ?? null;
}
