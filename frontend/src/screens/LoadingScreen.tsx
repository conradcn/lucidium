/**
 * Shown between the New Game confirm click and the first playable
 * frame (FR-031a). The MenuCarousel rendered behind us already
 * carries the player's chosen-style background + dream-guide /
 * player-character portrait from the interview, so the LoadingScreen
 * itself is just a centered spinner + a short framing line that
 * floats over the carousel.
 *
 * The framing line is a UX nudge for first-time players: an LLM
 * storyteller will say "no" to things you'd assumed weren't on
 * the table (a bookshop's hidden basement, a detective's cane,
 * a grandfather who's actually waiting on the porch). Reminding
 * the player up front that this is THEIR dream — and that
 * declaring something true via the free-text box is a valid move
 * — makes the engine feel more like a co-author and less like an
 * adversarial DM gating their imagination.
 */
import type { JSX } from "react";

export function LoadingScreen(): JSX.Element {
  return (
    <div className="loading-overlay" data-testid="loading-screen" aria-live="polite">
      <span className="loading-overlay__spinner" role="status" aria-label="Loading" />
      <p className="loading-overlay__hint">
        Remember that this is your dream. If you want something to be true,
        act as though it is.
      </p>
    </div>
  );
}
