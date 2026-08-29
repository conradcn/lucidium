import type { JSX } from "react";
import { assetUrl } from "../../app/assetUrl";

interface Props {
  whiteRoomPath?: string | undefined;
  dreamGuidePath?: string | undefined;
}

/**
 * Background dressing for the onboarding interview (FR-027). The white
 * room image fills the page as a soft backdrop, and the dream guide
 * portrait sits on the left edge so the player has a presence to talk
 * to. Both come bundled with the app and the backend hands their
 * absolute disk paths down through the interview snapshot.
 *
 * Both layers are positioned absolutely behind the page-frame content
 * via z-index — nothing here intercepts pointer events.
 */
export function InterviewStage({ whiteRoomPath, dreamGuidePath }: Props): JSX.Element | null {
  const bg = assetUrl(whiteRoomPath);
  const guide = assetUrl(dreamGuidePath);
  if (!bg && !guide) return null;
  return (
    <div className="interview-stage" aria-hidden>
      {bg ? (
        <div
          className="interview-stage__backdrop"
          style={{ backgroundImage: `url("${bg}")` }}
          data-image-path={whiteRoomPath}
          data-testid="interview-white-room"
        />
      ) : null}
      {guide ? (
        <img
          className="interview-stage__guide"
          src={guide}
          alt="dream guide"
          data-image-path={dreamGuidePath}
          data-testid="interview-dream-guide"
        />
      ) : null}
    </div>
  );
}
