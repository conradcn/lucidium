import type { JSX } from "react";
import { assetUrl } from "../../app/assetUrl";
import { useLucidiumStore } from "../../state/store";
import { PortraitSilhouette } from "./silhouettes";

interface CharacterImage {
  id: string;
  path: string;
}

interface Character {
  id: string;
  name: string;
  is_player: boolean;
  gender?: string;
  images: CharacterImage[];
}

interface Props {
  /** When true, hide every dialog-text element on the stage —
   *  including character name labels above portraits. The "h"
   *  hotkey toggles this so the player can take a clean
   *  screenshot of the rendered scene without the floating
   *  speaker tags. The portraits and silhouettes still show;
   *  only typographic UI hides. */
  hideText?: boolean;
}

export function CharacterStage({ hideText = false }: Props): JSX.Element {
  const game = useLucidiumStore((s) => s.game) as
    | { on_stage?: string[]; characters?: Record<string, Character> }
    | null;

  const onStage = game?.on_stage ?? [];
  const characters = game?.characters ?? {};

  // Count the characters that will actually render — the player
  // is excluded per FR-034a, and missing characters are skipped.
  // The stage's CSS uses this count to compute negative margin
  // so 3+ figures slide partially over each other instead of
  // overflowing the viewport. Two figures fit naturally; the
  // spacing math falls back to a normal gap below the overlap
  // threshold.
  const visibleCount = onStage.reduce((n, id) => {
    const c = characters[id];
    return c && !c.is_player ? n + 1 : n;
  }, 0);

  return (
    <div
      className="stage"
      style={{ "--character-count": visibleCount } as React.CSSProperties}
    >
      {onStage.map((id) => {
        const char = characters[id];
        if (!char) return null;
        // Per FR-034a: the player character is the lens, not a stage
        // actor. Skip them even if the backend somehow listed them.
        if (char.is_player) return null;
        const portrait = char.images?.[0];
        return (
          <div className="character" key={id}>
            {hideText ? null : (
              <span className="name">{char.name}</span>
            )}
            {portrait ? (
              <img src={assetUrl(portrait.path) ?? ""} alt={char.name} />
            ) : (
              <div
                className="placeholder"
                aria-label={`portrait pending for ${char.name}`}
              >
                <PortraitSilhouette gender={char.gender} />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
